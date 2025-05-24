"""Prediccion script for the MLflow model.

This script loads a model from MLflow and makes predictions on a dataset.

$ python3 make_predictions.py


"""

import mlflow
import pandas as pd

FILE_PATH = "data/winequality-red.csv"

df = pd.read_csv(FILE_PATH)
y = df["quality"]
x = df.drop(columns=["quality"])

## Debe verificarse el run_id del modelo que se quiere cargar
## Se puede obtener el run_id desde la interfaz de MLflow
logged_model = "runs:/3c967d714e784eb58ae1ca66e837d1c5/model"
loaded_model = mlflow.pyfunc.load_model(logged_model)
y = loaded_model.predict(x)

print(y)