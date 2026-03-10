import asyncio
import io
import json
import logging
import time

from starlette.applications import Starlette
from starlette.responses import StreamingResponse, JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect
from starlette.requests import Request

from google.cloud import texttospeech

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tts-api")

# ── Google TTS client ────────────────────────────────────────────────────────
logger.info("Initializing Google TTS client...")
client = texttospeech.TextToSpeechClient.from_service_account_json("key.json")
logger.info("Google TTS client initialized successfully")

CHUNK_SIZE = 4096


def synthesize(text: str) -> bytes:
    logger.debug(f"[SYNTHESIZE] Starting synthesis | text='{text[:80]}...' length={len(text)}")
    t0 = time.perf_counter()

    synthesis_input = texttospeech.SynthesisInput(text=text)
    logger.debug("[SYNTHESIZE] SynthesisInput created")

    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL,
    )
    logger.debug(f"[SYNTHESIZE] Voice params | language=en-US gender=NEUTRAL")

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    logger.debug("[SYNTHESIZE] Audio config | encoding=MP3")

    logger.info("[SYNTHESIZE] Calling Google Cloud TTS API...")
    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config,
    )

    elapsed = time.perf_counter() - t0
    audio_size = len(response.audio_content)
    logger.info(f"[SYNTHESIZE] Done | size={audio_size} bytes elapsed={elapsed:.3f}s")

    return response.audio_content


def audio_chunk_generator(audio: bytes, chunk_size: int = CHUNK_SIZE):
    total = len(audio)
    sent = 0
    buffer = io.BytesIO(audio)
    logger.debug(f"[CHUNK_GEN] Starting | total={total} bytes chunk_size={chunk_size}")

    while True:
        chunk = buffer.read(chunk_size)
        if not chunk:
            break
        sent += len(chunk)
        logger.debug(f"[CHUNK_GEN] Yielding chunk | size={len(chunk)} sent={sent}/{total}")
        yield chunk

    logger.debug(f"[CHUNK_GEN] Done | total_sent={sent} bytes")


# HTTP API
async def tts_api(request: Request):
    request_id = id(request)
    logger.info(f"[HTTP:{request_id}] POST /tts received")

    # Step 1: Parse body
    try:
        body = await request.json()
        logger.debug(f"[HTTP:{request_id}] Body parsed | keys={list(body.keys())}")
    except Exception as e:
        logger.warning(f"[HTTP:{request_id}] JSON parse failed | error={e}")
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    # Step 2: Validate text
    text = body.get("text", "").strip()
    logger.debug(f"[HTTP:{request_id}] Text extracted | length={len(text)} preview='{text[:60]}'")
    if not text:
        logger.warning(f"[HTTP:{request_id}] Empty text rejected")
        return JSONResponse({"error": "text is required"}, status_code=400)

    # Step 3: Synthesize
    try:
        logger.info(f"[HTTP:{request_id}] Dispatching synthesize() to thread...")
        t0 = time.perf_counter()
        audio = await asyncio.to_thread(synthesize, text)
        elapsed = time.perf_counter() - t0
        logger.info(f"[HTTP:{request_id}] Synthesis complete | size={len(audio)} bytes elapsed={elapsed:.3f}s")

        # Step 4: Build buffer
        buffer = io.BytesIO(audio)
        buffer.seek(0)
        logger.debug(f"[HTTP:{request_id}] BytesIO buffer ready | position={buffer.tell()}")

        # Step 5: Return response
        logger.info(f"[HTTP:{request_id}] Returning StreamingResponse | media_type=audio/mpeg")
        return StreamingResponse(
            buffer,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "attachment; filename=tts.mp3",
                "Content-Length": str(len(audio)),
            }
        )

    except Exception as e:
        logger.exception(f"[HTTP:{request_id}] Synthesis failed | error={e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# WebSocket API 
async def websocket_tts(websocket: WebSocket):
    ws_id = id(websocket)
    logger.info(f"[WS:{ws_id}] New connection | client={websocket.client}")
    await websocket.accept()
    logger.info(f"[WS:{ws_id}] Connection accepted")

    try:
        while True:
            # Step 1: Receive
            logger.debug(f"[WS:{ws_id}] Waiting for message...")
            data = await websocket.receive_text()
            logger.debug(f"[WS:{ws_id}] Message received | length={len(data)}")

            # Step 2: Parse
            try:
                payload = json.loads(data)
                text = payload.get("text", "").strip()
                logger.debug(f"[WS:{ws_id}] JSON parsed | text_length={len(text)}")
            except json.JSONDecodeError as e:
                logger.warning(f"[WS:{ws_id}] JSON decode failed, treating as raw text | error={e}")
                text = data.strip()

            # Step 3: Validate
            if not text:
                logger.warning(f"[WS:{ws_id}] Empty text, sending error")
                await websocket.send_json({"error": "text is required"})
                continue

            # Step 4: Synthesize
            try:
                logger.info(f"[WS:{ws_id}] Dispatching synthesize() to thread...")
                t0 = time.perf_counter()
                audio = await asyncio.to_thread(synthesize, text)
                elapsed = time.perf_counter() - t0
                logger.info(f"[WS:{ws_id}] Synthesis complete | size={len(audio)} bytes elapsed={elapsed:.3f}s")

                # Step 5: Stream chunks
                logger.info(f"[WS:{ws_id}] Streaming audio chunks...")
                chunk_count = 0
                for chunk in audio_chunk_generator(audio):
                    await websocket.send_bytes(chunk)
                    chunk_count += 1

                logger.info(f"[WS:{ws_id}] All chunks sent | total_chunks={chunk_count}")

                # Step 6: Done signal
                await websocket.send_json({"status": "done"})
                logger.info(f"[WS:{ws_id}] Done signal sent")

            except Exception as e:
                logger.exception(f"[WS:{ws_id}] Synthesis failed | error={e}")
                await websocket.send_json({"error": str(e)})

    except WebSocketDisconnect:
        logger.info(f"[WS:{ws_id}] Client disconnected")
    except Exception as e:
        logger.exception(f"[WS:{ws_id}] Unexpected error | error={e}")


routes = [
    Route("/tts", tts_api, methods=["POST"]),
    WebSocketRoute("/ws", websocket_tts),
]

app = Starlette(routes=routes)