
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv('Preprocessed_Balanced_dataset.csv')

X = df.drop(columns=[
    'Attack_sub_category',
    'Label',
    'Attack_Category'
])

y = df['Attack_sub_category']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

mlflow.set_experiment("Darknet_Classification")

with mlflow.start_run():

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(
        y_test,
        y_pred,
        average='weighted'
    )
    rec = recall_score(
        y_test,
        y_pred,
        average='weighted'
    )
    f1 = f1_score(
        y_test,
        y_pred,
        average='weighted'
    )

    mlflow.log_param(
        "n_estimators",
        100
    )

    mlflow.log_metric(
        "accuracy",
        acc
    )

    mlflow.log_metric(
        "precision",
        prec
    )

    mlflow.log_metric(
        "recall",
        rec
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )

    mlflow.sklearn.log_model(
        model,
        "model"
    )

    joblib.dump(
        model,
        "pipeline_terbaik.pkl"
    )

    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1 Score :", f1)
