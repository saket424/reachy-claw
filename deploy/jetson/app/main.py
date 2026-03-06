"""FastAPI speech service: SenseVoice ASR + Kokoro TTS."""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Jetson Speech Service", version="1.0.0")


class TTSRequest(BaseModel):
    text: str
    sid: int | None = None
    speed: float = 1.0


@app.on_event("startup")
async def startup():
    import asr_service, tts_service

    logger.info("Pre-loading ASR model...")
    asr_service.get_recognizer()

    logger.info("Pre-loading TTS model...")
    tts_service.preload()

    logger.info("Speech service ready.")


@app.get("/health")
async def health():
    import asr_service, tts_service

    return {
        "asr": asr_service.is_ready(),
        "tts": tts_service.is_ready(),
    }


@app.post("/asr")
async def asr(
    file: UploadFile = File(...),
    language: str = Query("auto"),
):
    import asr_service

    audio_bytes = await file.read()
    text = asr_service.transcribe_audio(audio_bytes, language=language)
    return {"text": text}


@app.post("/tts")
async def tts(req: TTSRequest):
    import tts_service

    wav_bytes, meta = tts_service.synthesize(
        text=req.text,
        speaker_id=req.sid,
        speed=req.speed,
    )
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-Audio-Duration": str(meta["duration"]),
            "X-Inference-Time": str(meta["inference_time"]),
            "X-RTF": str(meta["rtf"]),
        },
    )
