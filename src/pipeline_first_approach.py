import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
import mlflow
from mlflow.tracking import MlflowClient
import joblib
from sklearn.metrics import f1_score, recall_score, precision_score, log_loss
import time


class FraudDetectionModel:
    def __init__(self, data, target, model_type='logistic', mlflow_uri="http://localhost:5000"):
        self.data = data
        self.target = target
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.model = None
        self.model_type = model_type
        mlflow.set_tracking_uri(mlflow_uri)  # Asegurarse de que esto apunte a tu servidor de MLflow
        self.experiment_name = "Fraud_Detection_Experiment"
        mlflow.set_experiment(self.experiment_name)

    def prepare_data(self):
        # Identificar columnas categóricas
        categorical_cols = self.data.select_dtypes(include=['object']).columns.tolist()

        # Definir el clasificador basado en el tipo de modelo seleccionado
        if self.model_type == 'logistic':
            classifier = LogisticRegression()
        elif self.model_type == 'tree':
            classifier = DecisionTreeClassifier()
        elif self.model_type == 'lgbm':
            classifier = lgb.LGBMClassifier()
        else:
            raise ValueError("Invalid model type provided. Choose 'logistic', 'tree', or 'lgbm'")

        # Crear un pipeline para preprocesamiento y modelo
        self.pipeline = ImbPipeline(steps=[
            ('preprocessor', ColumnTransformer(
                transformers=[
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
                ])),
            ('sampler', SMOTE(random_state=42)),
            ('classifier', classifier)
        ])

    def split_data(self, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data, self.target, test_size=test_size, random_state=random_state
        )

    def train_model(self, model_name):
        start_time = time.time()

        with mlflow.start_run():
            self.model = self.pipeline.fit(self.X_train, self.y_train)
            train_time = time.time() - start_time

            y_pred = self.model.predict(self.X_test)
            y_proba = self.model.predict_proba(self.X_test)[:, 1]

            f1 = f1_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            logloss = log_loss(self.y_test, y_proba)


            mlflow.sklearn.log_model(self.model, "model", registered_model_name=model_name)

            # Registrar parámetros y métricas como ejemplo
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("train_time", train_time)
            accuracy = self.model.score(self.X_test, self.y_test)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("log_loss", logloss)          


    def load_model(self, model_name, version=None, stage=None):
        client = MlflowClient()
        if version is not None:
            model_uri = f"models:/{model_name}/{version}"
        elif stage is not None:
            model_uri = f"models:/{model_name}/{stage}"
        else:
            model_uri = f"models:/{model_name}/latest"  # Si no se especifica, carga la última versión
        return mlflow.pyfunc.load_model(model_uri)

    def predict(self, X):
        return self.model.predict(X)