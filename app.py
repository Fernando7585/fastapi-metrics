from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics import mean_absolute_error, r2_score

# Crear la aplicación FastAPI
app = FastAPI(
    title="API de métricas",
    description="Calcula MAE y R² usando scikit-learn",
    version="1.0.0"
)

# 🚨 Configuración de CORS
# Permite que la página estática en redmocaaltama.org pueda hacer fetch a esta API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://redmocaaltama.org",
        "https://www.redmocaaltama.org",
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Endpoint de prueba para ver si la API está viva
@app.get("/health")
def health():
    return {"status": "ok"}

# Endpoint principal de métricas
@app.get("/metrics")
def metrics(
    y_true: str = Query(..., description="Valores reales separados por coma"),
    y_pred: str = Query(..., description="Valores predichos separados por coma"),
):
    """
    Ejemplo de uso:
    /metrics?y_true=3,-0.5,2,7&y_pred=2.5,0,2,8
    """
    try:
        yt = [float(x) for x in y_true.split(",") if x.strip()]
        yp = [float(x) for x in y_pred.split(",") if x.strip()]

        if len(yt) != len(yp):
            raise ValueError("Las listas deben tener el mismo tamaño.")

        mae = float(mean_absolute_error(yt, yp))
        r2 = float(r2_score(yt, yp))

        return {
            "mae": mae,
            "r2": r2,
            "n": len(yt),
            "data": {"y_true": yt, "y_pred": yp}
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
