"""
api/main.py — FastAPI backend for image enhancement
Exposes /enhance endpoint that delegates to infer.py
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import base64
import io
import time
from infer import EnhancementInfer

app = FastAPI(title="Image Enhancement API", version="1.0.0")

# Allow Angular dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-load model once at startup
infer_engine: EnhancementInfer = None

@app.on_event("startup")
async def startup_event():
    global infer_engine
    infer_engine = EnhancementInfer()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": infer_engine is not None}


@app.post("/enhance")
async def enhance(
    file: UploadFile = File(...),
    model: str = Form(default="esrgan"),          # "esrgan" | "pix2pix"
    scale: int = Form(default=4),                  # upscale factor (ESRGAN)
    strength: float = Form(default=1.0),           # enhancement strength 0–1
):
    """
    Accepts an image file, runs enhancement via infer.py,
    returns base64-encoded result + attention heatmap.
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=400, detail="Unsupported image type")

    image_bytes = await file.read()
    if len(image_bytes) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large (max 20 MB)")

    try:
        t0 = time.time()
        result = infer_engine.run(
            image_bytes=image_bytes,
            model=model,
            scale=scale,
            strength=strength,
        )
        elapsed = round(time.time() - t0, 3)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    def to_b64(img_bytes: bytes) -> str:
        return base64.b64encode(img_bytes).decode()

    return JSONResponse({
        "enhanced_image": to_b64(result["enhanced"]),
        "heatmap": to_b64(result["heatmap"]),
        "model": model,
        "scale": scale,
        "inference_time_s": elapsed,
        "original_size": result["original_size"],
        "enhanced_size": result["enhanced_size"],
    })
