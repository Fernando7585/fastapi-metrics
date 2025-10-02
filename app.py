from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="API de métricas (mínima)")

# CORS: deja abierto para probar. Luego restringe a tu dominio.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # <-- para probar
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

def _parse_list(s: str):
    vals = [x.strip() for x in s.split(",")]
    vals = [float(x) for x in vals if x != ""]
    return vals

def _mean(a):
    return sum(a) / len(a)

def _mae(y, yhat):
    return sum(abs(a - b) for a, b in zip(y, yhat)) / len(y)

def _r2(y, yhat):
    ybar = _mean(y)
    ss_res = sum((a - b) ** 2 for a, b in zip(y, yhat))
    ss_tot = sum((a - ybar) ** 2 for a in y)
    return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0

@app.get("/metrics")
def metrics(
    y_true: str = Query(..., description="CSV: 3,-0.5,2,7"),
    y_pred: str = Query(..., description="CSV: 2.5,0,2,8"),
):
    try:
        yt = _parse_list(y_true)
        yp = _parse_list(y_pred)
        if not yt or not yp:
            raise ValueError("Listas vacías.")
        if len(yt) != len(yp):
            raise ValueError("Listas con diferente tamaño.")
        mae = _mae(yt, yp)
        r2 = _r2(yt, yp)
        return {"mae": mae, "r2": r2, "n": len(yt), "data": {"y_true": yt, "y_pred": yp}}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
