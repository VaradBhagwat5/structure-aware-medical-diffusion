# Teammate C — Backend + Frontend

## Project structure
```
api/
  main.py        ← FastAPI app, /enhance endpoint
  infer.py       ← ESRGAN / Pix2Pix inference wrapper
  requirements.txt

frontend/src/app/
  services/enhancement.service.ts   ← HTTP client
  components/enhancer/
    enhancer.component.ts            ← logic
    enhancer.component.html          ← UI: upload · slider · heatmap
    enhancer.component.scss          ← dark-theme styles
  app.module.ts
```

---

## Backend setup

```bash
cd api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Health check → http://localhost:8000/health

### Connecting real model checkpoints
Open `infer.py` and replace the `TODO` lines with your actual checkpoint paths:

```python
# ESRGAN
m.load_state_dict(torch.load("checkpoints/esrgan_x4.pth", map_location=self.device))

# Pix2Pix
m.load_state_dict(torch.load("checkpoints/pix2pix.pth", map_location=self.device))
```

---

## Frontend setup

```bash
cd frontend
npm install
ng serve          # dev server → http://localhost:4200
```

### UI features
| Feature | Description |
|---|---|
| Drag-and-drop upload | Drop zone or click-to-browse |
| Model selector | ESRGAN (super-res) · Pix2Pix (translation) |
| Scale buttons | ×2 / ×4 / ×8 (ESRGAN) |
| Strength slider | Blend 0–100% enhancement |
| Before/after comparison | Draggable divider slider |
| Attention heatmap | Toggle canvas overlay |
| Download | Save enhanced PNG |

---

## API reference

### `POST /enhance`
| Field | Type | Default | Description |
|---|---|---|---|
| `file` | image/* | — | Input image (max 20 MB) |
| `model` | string | `esrgan` | `esrgan` or `pix2pix` |
| `scale` | int | `4` | Upscale factor (ESRGAN only) |
| `strength` | float | `1.0` | Enhancement blend (0.0–1.0) |

**Response**
```json
{
  "enhanced_image": "<base64 PNG>",
  "heatmap":        "<base64 PNG>",
  "model":          "esrgan",
  "scale":          4,
  "inference_time_s": 0.342,
  "original_size":  [512, 512],
  "enhanced_size":  [2048, 2048]
}
```
