from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

app = FastAPI(title="Metrics API", version="1.0.0")

# Ajusta esto a tus orígenes reales (tu sitio estático)
ALLOWED_ORIGINS = [
    "https://redmocaaltama.org",      # tu dominio
    "https://www.redmocaaltama.org",  # variante www
    "*"  # en desarrollo; en producción es mejor ser específico
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MetricsRequest(BaseModel):
    y_true: List[float]
    y_pred: List[float]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/api/metrics")
def metrics_get(
    y_true: str = Query(..., description="CSV de reales, ej: 3,-0.5,2,7"),
    y_pred: str = Query(..., description="CSV de predicciones, ej: 2.5,0,2,8"),
):
    try:
        yt = [float(x.strip()) for x in y_true.split(",") if x.strip() != ""]
        yp = [float(x.strip()) for x in y_pred.split(",") if x.strip() != ""]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al parsear: {e}")

    if len(yt) == 0 or len(yp) == 0:
        raise HTTPException(status_code=400, detail="Listas vacías.")
    if len(yt) != len(yp):
        raise HTTPException(status_code=400, detail="Listas con diferente tamaño.")

    df = pd.DataFrame({"y_true": yt, "y_pred": yp})
    mae = mean_absolute_error(df["y_true"], df["y_pred"])
    r2 = r2_score(df["y_true"], df["y_pred"])

    return {
        "mae": float(mae),
        "r2": float(r2),
        "n": len(yt),
        "data": {
            "y_true": yt,
            "y_pred": yp,
        }
    }

@app.post("/api/metrics")
def metrics_post(payload: MetricsRequest):
    yt = payload.y_true
    yp = payload.y_pred
    if len(yt) == 0 or len(yp) == 0:
        raise HTTPException(status_code=400, detail="Listas vacías.")
    if len(yt) != len(yp):
        raise HTTPException(status_code=400, detail="Listas con diferente tamaño.")
    df = pd.DataFrame({"y_true": yt, "y_pred": yp})
    mae = mean_absolute_error(df["y_true"], df["y_pred"])
    r2 = r2_score(df["y_true"], df["y_pred"])
    return {
        "mae": float(mae),
        "r2": float(r2),
        "n": len(yt),
        "data": {"y_true": yt, "y_pred": yp}
    }
