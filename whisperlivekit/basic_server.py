import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from whisperlivekit import (
    AudioProcessor,
    TranscriptionEngine,
    get_inline_ui_html,
    parse_args,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

args = parse_args()
transcription_engines = {}  # Cache engines by language

# Supported languages for the UI
SUPPORTED_LANGUAGES = {
    "auto": "Auto Detect",
    "en": "English",
    "de": "German",
    "fr": "French",
    "it": "Italian",
    "es": "Spanish",
    "pt": "Portuguese",
    "nl": "Dutch",
    "pl": "Polish",
    "cs": "Czech",
    "sk": "Slovak",
    "hu": "Hungarian",
    "ro": "Romanian",
    "bg": "Bulgarian",
    "hr": "Croatian",
    "sl": "Slovenian",
    "el": "Greek",
    "da": "Danish",
    "sv": "Swedish",
    "no": "Norwegian",
    "fi": "Finnish",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "hi": "Hindi",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "ar": "Arabic",
    "he": "Hebrew",
    "tr": "Turkish",
    "fa": "Persian",
    "ru": "Russian",
    "uk": "Ukrainian",
    "ca": "Catalan",
    "eu": "Basque",
    "gl": "Galician",
}


def get_transcription_engine(language: str) -> TranscriptionEngine:
    """Get or create a transcription engine for the specified language."""
    if language not in transcription_engines:
        logger.info(f"Creating new TranscriptionEngine for language: {language}")
        engine_args = vars(args).copy()
        engine_args["lan"] = language  # 'lan' is the actual arg name
        transcription_engines[language] = TranscriptionEngine(**engine_args)
    return transcription_engines[language]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load the default engine
    default_lang = getattr(args, "lan", "auto") or "auto"
    get_transcription_engine(default_lang)
    yield
    # Cleanup engines on shutdown
    transcription_engines.clear()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def get():
    return HTMLResponse(get_inline_ui_html())


@app.get("/api/config")
async def get_config():
    """Return server configuration including supported languages and features."""
    return JSONResponse(
        {
            "supported_languages": SUPPORTED_LANGUAGES,
            "default_language": args.lan or "auto",
            "model": getattr(args, "model_size", "base"),
            "backend": getattr(args, "backend", "auto"),
            "features": {
                "diarization": getattr(args, "diarization", False),
                "diarization_backend": getattr(
                    args, "diarization_backend", "sortformer"
                ),
                "translation": bool(getattr(args, "target_language", "")),
                "vad": getattr(args, "vad", True),
                "confidence_scores": True,  # Now supported
                "word_timestamps": True,
                "language_detection": args.lan == "auto",
            },
            "version": "1.1.0",
        }
    )


@app.get("/api/languages")
async def get_languages():
    """Return list of supported languages for transcription."""
    return JSONResponse(
        {
            "languages": [
                {"code": code, "name": name}
                for code, name in SUPPORTED_LANGUAGES.items()
            ]
        }
    )


async def handle_websocket_results(websocket, results_generator):
    """Consumes results from the audio processor and sends them via WebSocket."""
    try:
        async for response in results_generator:
            await websocket.send_json(response.to_dict())
        # when the results_generator finishes it means all audio has been processed
        logger.info("Results generator finished. Sending 'ready_to_stop' to client.")
        await websocket.send_json({"type": "ready_to_stop"})
    except WebSocketDisconnect:
        logger.info(
            "WebSocket disconnected while handling results (client likely closed connection)."
        )
    except Exception as e:
        logger.exception(f"Error in WebSocket results handler: {e}")


@app.websocket("/asr")
async def websocket_endpoint(
    websocket: WebSocket,
    language: Optional[str] = Query(
        default=None, description="Language code for transcription"
    ),
):
    # Use language from query param, or fall back to server default
    effective_language = (
        language
        if language and language in SUPPORTED_LANGUAGES
        else (args.lan or "auto")
    )
    logger.info(f"WebSocket connection with language: {effective_language}")

    transcription_engine = get_transcription_engine(effective_language)

    audio_processor = AudioProcessor(
        transcription_engine=transcription_engine,
    )
    await websocket.accept()
    logger.info("WebSocket connection opened.")

    try:
        await websocket.send_json(
            {
                "type": "config",
                "useAudioWorklet": bool(args.pcm_input),
                "language": effective_language,
                "supported_languages": list(SUPPORTED_LANGUAGES.keys()),
            }
        )
    except Exception as e:
        logger.warning(f"Failed to send config to client: {e}")

    results_generator = await audio_processor.create_tasks()
    websocket_task = asyncio.create_task(
        handle_websocket_results(websocket, results_generator)
    )

    try:
        while True:
            message = await websocket.receive_bytes()
            await audio_processor.process_audio(message)
    except KeyError as e:
        if "bytes" in str(e):
            logger.warning(f"Client has closed the connection.")
        else:
            logger.error(
                f"Unexpected KeyError in websocket_endpoint: {e}", exc_info=True
            )
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client during message receiving loop.")
    except Exception as e:
        logger.error(
            f"Unexpected error in websocket_endpoint main loop: {e}", exc_info=True
        )
    finally:
        logger.info("Cleaning up WebSocket endpoint...")
        if not websocket_task.done():
            websocket_task.cancel()
        try:
            await websocket_task
        except asyncio.CancelledError:
            logger.info("WebSocket results handler task was cancelled.")
        except Exception as e:
            logger.warning(f"Exception while awaiting websocket_task completion: {e}")

        await audio_processor.cleanup()
        logger.info("WebSocket endpoint cleaned up successfully.")


def main():
    """Entry point for the CLI command."""
    import uvicorn

    uvicorn_kwargs = {
        "app": "whisperlivekit.basic_server:app",
        "host": args.host,
        "port": args.port,
        "reload": False,
        "log_level": "info",
        "lifespan": "on",
    }

    ssl_kwargs = {}
    if args.ssl_certfile or args.ssl_keyfile:
        if not (args.ssl_certfile and args.ssl_keyfile):
            raise ValueError(
                "Both --ssl-certfile and --ssl-keyfile must be specified together."
            )
        ssl_kwargs = {
            "ssl_certfile": args.ssl_certfile,
            "ssl_keyfile": args.ssl_keyfile,
        }

    if ssl_kwargs:
        uvicorn_kwargs = {**uvicorn_kwargs, **ssl_kwargs}
    if args.forwarded_allow_ips:
        uvicorn_kwargs = {
            **uvicorn_kwargs,
            "forwarded_allow_ips": args.forwarded_allow_ips,
        }

    uvicorn.run(**uvicorn_kwargs)


if __name__ == "__main__":
    main()
