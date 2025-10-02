from fastapi import FastAPI
from sklearn.metrics import mean_absolute_error, r2_score

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hola Mundo desde FastAPI en Render"}

@app.get("/metrics")
def calc_metrics(y_true: str, y_pred: str):
    try:
        y_true = [float(x) for x in y_true.split(",")]
        y_pred = [float(x) for x in y_pred.split(",")]

        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        return {
            "mae": mae,
            "r2": r2,
            "y_true": y_true,
            "y_pred": y_pred
        }
    except Exception as e:
        return {"error": str(e)}
