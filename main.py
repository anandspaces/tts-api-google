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

# ── Supported gender values ───────────────────────────────────────────────────
GENDER_MAP = {
    "neutral": texttospeech.SsmlVoiceGender.NEUTRAL,
    "male":    texttospeech.SsmlVoiceGender.MALE,
    "female":  texttospeech.SsmlVoiceGender.FEMALE,
}

# ── Default voice options ─────────────────────────────────────────────────────
DEFAULT_LANGUAGE = "en-US"
DEFAULT_GENDER    = "neutral"


def synthesize(
    text: str,
    language_code: str = DEFAULT_LANGUAGE,
    gender: str = DEFAULT_GENDER,
    voice_name: str | None = None,
) -> bytes:
    """
    Synthesize speech from text.

    Parameters
    ----------
    text          : The text to convert to speech.
    language_code : BCP-47 language tag, e.g. "en-US", "fr-FR", "hi-IN".
    gender        : "neutral" | "male" | "female"  (ignored when voice_name is set).
    voice_name    : Specific voice name, e.g. "en-US-Neural2-F".
                    When provided, language_code is still required.
    """
    # When a specific voice_name is given, omit ssml_gender entirely —
    # Neural2 / Journey / Studio voices reject NEUTRAL and the name already
    # encodes the gender, so letting Google infer it avoids a 400 error.
    if voice_name:
        ssml_gender = texttospeech.SsmlVoiceGender.SSML_VOICE_GENDER_UNSPECIFIED
    else:
        ssml_gender = GENDER_MAP.get(gender.lower(), texttospeech.SsmlVoiceGender.NEUTRAL)

    logger.debug(
        f"[SYNTHESIZE] Starting | lang={language_code} gender={gender} "
        f"voice_name={voice_name} length={len(text)}"
    )
    t0 = time.perf_counter()

    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice_params = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        ssml_gender=ssml_gender,
        **({"name": voice_name} if voice_name else {}),
    )
    logger.debug(f"[SYNTHESIZE] Voice params | {voice_params}")

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    logger.info("[SYNTHESIZE] Calling Google Cloud TTS API...")
    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice_params,
        audio_config=audio_config,
    )

    elapsed = time.perf_counter() - t0
    audio_size = len(response.audio_content)
    logger.info(f"[SYNTHESIZE] Done | size={audio_size} bytes elapsed={elapsed:.3f}s")

    return response.audio_content


def audio_chunk_generator(audio: bytes, chunk_size: int = CHUNK_SIZE):
    total  = len(audio)
    sent   = 0
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


# ── HTTP API ──────────────────────────────────────────────────────────────────
async def tts_api(request: Request):
    """
    POST /tts

    JSON body:
      {
        "text":          "Hello, world!",          # required
        "language_code": "en-US",                  # optional, default "en-US"
        "gender":        "neutral|male|female",    # optional, default "neutral"
        "voice_name":    "en-US-Neural2-F"         # optional, overrides gender
      }

    Returns: audio/mpeg stream
    """
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
    if not text:
        logger.warning(f"[HTTP:{request_id}] Empty text rejected")
        return JSONResponse({"error": "text is required"}, status_code=400)

    # Step 3: Extract voice options
    language_code = body.get("language_code", DEFAULT_LANGUAGE).strip()
    gender        = body.get("gender",        DEFAULT_GENDER).strip().lower()
    voice_name    = body.get("voice_name",    None)

    if gender not in GENDER_MAP:
        return JSONResponse(
            {"error": f"Invalid gender '{gender}'. Must be one of: {list(GENDER_MAP.keys())}"},
            status_code=400,
        )

    logger.debug(
        f"[HTTP:{request_id}] Options | lang={language_code} gender={gender} voice_name={voice_name}"
    )

    # Step 4: Synthesize
    try:
        logger.info(f"[HTTP:{request_id}] Dispatching synthesize() to thread...")
        t0 = time.perf_counter()
        audio = await asyncio.to_thread(synthesize, text, language_code, gender, voice_name)
        elapsed = time.perf_counter() - t0
        logger.info(f"[HTTP:{request_id}] Synthesis complete | size={len(audio)} bytes elapsed={elapsed:.3f}s")

        buffer = io.BytesIO(audio)
        buffer.seek(0)

        logger.info(f"[HTTP:{request_id}] Returning StreamingResponse | media_type=audio/mpeg")
        return StreamingResponse(
            buffer,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "attachment; filename=tts.mp3",
                "Content-Length":      str(len(audio)),
            },
        )

    except Exception as e:
        logger.exception(f"[HTTP:{request_id}] Synthesis failed | error={e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ── WebSocket API ─────────────────────────────────────────────────────────────
async def websocket_tts(websocket: WebSocket):
    """
    WebSocket /ws

    Send JSON message:
      {
        "text":          "Hello, world!",          # required
        "language_code": "en-US",                  # optional, default "en-US"
        "gender":        "neutral|male|female",    # optional, default "neutral"
        "voice_name":    "en-US-Neural2-F"         # optional, overrides gender
      }

    Receives: binary audio chunks, then {"status": "done"}
    """
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
                text          = payload.get("text",          "").strip()
                language_code = payload.get("language_code", DEFAULT_LANGUAGE).strip()
                gender        = payload.get("gender",        DEFAULT_GENDER).strip().lower()
                voice_name    = payload.get("voice_name",    None)
                logger.debug(
                    f"[WS:{ws_id}] Parsed | lang={language_code} gender={gender} "
                    f"voice_name={voice_name} text_len={len(text)}"
                )
            except json.JSONDecodeError as e:
                logger.warning(f"[WS:{ws_id}] JSON decode failed, treating as raw text | error={e}")
                text          = data.strip()
                language_code = DEFAULT_LANGUAGE
                gender        = DEFAULT_GENDER
                voice_name    = None

            # Step 3: Validate
            if not text:
                logger.warning(f"[WS:{ws_id}] Empty text, sending error")
                await websocket.send_json({"error": "text is required"})
                continue

            if gender not in GENDER_MAP:
                await websocket.send_json(
                    {"error": f"Invalid gender '{gender}'. Must be one of: {list(GENDER_MAP.keys())}"}
                )
                continue

            # Step 4: Synthesize
            try:
                logger.info(f"[WS:{ws_id}] Dispatching synthesize() to thread...")
                t0 = time.perf_counter()
                audio = await asyncio.to_thread(synthesize, text, language_code, gender, voice_name)
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