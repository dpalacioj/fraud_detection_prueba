import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
import mlflow
from mlflow.tracking import MlflowClient
import joblib
from sklearn.metrics import f1_score, recall_score, precision_score, log_loss
import time
from group_time_series_split import GroupTimeSeriesSplit
import matplotlib.pyplot as plt


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

    def prepare_data(self, balance='None'):
        # Identificar columnas categóricas
        categorical_cols = self.data.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = self.data.select_dtypes(include=['number']).columns.tolist()

        # Definir el clasificador basado en el tipo de modelo seleccionado
        if self.model_type == 'logistic':
            classifier = LogisticRegression()
        elif self.model_type == 'tree':
            classifier = DecisionTreeClassifier()
        elif self.model_type == 'lgbm':
            classifier = lgb.LGBMClassifier()
        else:
            raise ValueError("Invalid model type provided. Choose 'logistic', 'tree', or 'lgbm'")

        # Configurar el paso de balance de clases si es necesario
        if balance == 'SMOTE':
            sampler = SMOTE(random_state=42)
        elif balance == 'ADASYN':
            sampler = ADASYN(random_state=42)
        elif balance == 'None':
            sampler = None
        else:
            raise ValueError("Invalid balance option. Choose 'SMOTE', 'ADASYN', or 'None'")

        # Crear el ColumnTransformer para el preprocesamiento
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ],
            remainder='passthrough')

        # Configurar el pipeline, incluyendo el sampler si no es None
        steps = [('preprocessor', preprocessor)]
        if sampler:
            steps.append(('sampler', sampler))
        steps.append(('classifier', classifier))
        
        self.pipeline = ImbPipeline(steps)

    def split_data(self, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data, self.target, test_size=test_size, random_state=random_state
        )

    def train_model(self, model_name):
        start_time = time.time()
        groups = self.data['step']
        gkf = GroupTimeSeriesSplit(n_splits=2)  # Ajusta el número de divisiones según sea necesario

        with mlflow.start_run():
            # Listas para almacenar métricas de todos los pliegues
            accuracies, f1_scores, precisions, recalls, log_losses = [], [], [], [], []

            for fold, (train_idx, test_idx) in enumerate(gkf.split(self.data, self.target, groups)):
                print('-' * 40)
                print(f'Fold: {fold}')
                print('Groups for training:')
                print(self.data.loc[train_idx, 'step'].unique())
                print('Group for test:')
                print(self.data.loc[test_idx, 'step'].unique())
                print('-' * 40)

                X_train, X_test = self.data.iloc[train_idx], self.data.iloc[test_idx]
                y_train, y_test = self.target.iloc[train_idx], self.target.iloc[test_idx]

                # Entrenar el modelo
                self.model = self.pipeline.fit(X_train, y_train)

                # Predecir y evaluar
                y_pred = self.model.predict(X_test)
                y_proba = self.model.predict_proba(X_test)[:, 1]
                accuracy = self.model.score(X_test, y_test)
                f1 = f1_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                logloss = log_loss(y_test, y_proba)

                # Almacenar métricas de este pliegue
                accuracies.append(accuracy)
                f1_scores.append(f1)
                precisions.append(precision)
                recalls.append(recall)
                log_losses.append(logloss)

                # Registrar métricas para este pliegue en MLflow
                mlflow.log_metric(f"fold_{fold}_accuracy", accuracy)
                mlflow.log_metric(f"fold_{fold}_f1_score", f1)
                mlflow.log_metric(f"fold_{fold}_precision", precision)
                mlflow.log_metric(f"fold_{fold}_recall", recall)
                mlflow.log_metric(f"fold_{fold}_log_loss", logloss)

            # Calcular métricas promedio
            avg_accuracy = sum(accuracies) / len(accuracies)
            avg_f1 = sum(f1_scores) / len(f1_scores)
            avg_precision = sum(precisions) / len(precisions)
            avg_recall = sum(recalls) / len(recalls)
            avg_log_loss = sum(log_losses) / len(log_losses)

            # Registrar métricas promedio en MLflow
            mlflow.log_metric("average_accuracy", avg_accuracy)
            mlflow.log_metric("average_f1_score", avg_f1)
            mlflow.log_metric("average_precision", avg_precision)
            mlflow.log_metric("average_recall", avg_recall)
            mlflow.log_metric("average_log_loss", avg_log_loss)


            # Registrar el método de balance usado
            balance_method = self.pipeline.named_steps.get('sampler', 'None')
            if balance_method != 'None':
                balance_method = type(balance_method).__name__
            mlflow.log_param("balance_method", balance_method)

            # Registrar información adicional como parámetros o el nombre del modelo
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("n_splits", gkf.n_splits)
            mlflow.log_param("total_training_time", time.time() - start_time)

            # Si se desea, también se puede registrar el modelo
            mlflow.sklearn.log_model(self.model, "model", registered_model_name=model_name)

            print(f"Average Metrics - Accuracy: {avg_accuracy}, F1: {avg_f1}, Precision: {avg_precision}, Recall: {avg_recall}, Log Loss: {avg_log_loss}")

        # Tiempo total de entrenamiento
        total_time = time.time() - start_time
        print(f"Total training time: {total_time:.2f} seconds")

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