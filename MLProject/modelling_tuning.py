import os
import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, log_loss, roc_auc_score
)

def run_tuning(data_dir):
    # Setup Path Dinamis
    df_train = pd.read_csv(os.path.join(data_dir, 'train_cleaned_Agung.csv'))
    df_val = pd.read_csv(os.path.join(data_dir, 'val_cleaned_Agung.csv'))
    df_test = pd.read_csv(os.path.join(data_dir, 'test_cleaned_Agung.csv'))

    X_train = pd.concat([df_train['cleaned_text'], df_val['cleaned_text']]).fillna('')
    y_train = pd.concat([df_train['label'], df_val['label']])
    X_test = df_test['cleaned_text'].fillna('')
    y_test = df_test['label']

    mlflow.set_experiment("Sentiment_Analysis_RF")
    mlflow.sklearn.autolog(disable=True) # Tetap matikan agar log manual Anda tidak tertumpuk

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 3), max_features=4500)),
        ('rf', RandomForestClassifier(random_state=42))
    ])
    
    parameters = {
        'rf__n_estimators': [100], # Dikurangi untuk mempercepat alur CI
        'rf__max_depth': [None]
    }

    with mlflow.start_run(run_name="RF_Tuning_CI"):
        grid_search = GridSearchCV(pipeline, parameters, cv=5, scoring='f1_macro', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        # --- METRIK TETAP DIPERTAHANKAN SESUAI PERMINTAAN ---
        y_train_pred = best_model.predict(X_train)
        y_train_proba = best_model.predict_proba(X_train)
        y_test_pred = best_model.predict(X_test)
        y_test_proba = best_model.predict_proba(X_test)

        # Log Metrics (Manual)
        mlflow.log_metric("train_accuracy", accuracy_score(y_train, y_train_pred))
        mlflow.log_metric("train_f1", f1_score(y_train, y_train_pred, average='weighted'))
        mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_test_pred))
        mlflow.log_metric("test_f1", f1_score(y_test, y_test_pred, average='weighted'))
        mlflow.log_metric("test_log_loss", log_loss(y_test, y_test_proba))
        mlflow.log_metric("test_roc_auc", roc_auc_score(y_test, y_test_proba, multi_class='ovr', average='weighted'))

        # Metrik Tambahan
        mlflow.log_metric("best_cv_score", grid_search.best_score_)
        
        # Simpan Artefak
        mlflow.sklearn.log_model(best_model, "random_forest_model")
        print("Tuning dan Logging Metrik Manual selesai di CI.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    args = parser.parse_args()
    run_tuning(args.data_dir)