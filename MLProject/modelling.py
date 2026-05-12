import os
import argparse
import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from modelling_tuning import run_hyperparameter_tuning

def main(data_dir):
    # 1. Persiapan Data: Load CSV files dynamically based on the input directory parameter
    df_train = pd.read_csv(os.path.join(data_dir, 'train_cleaned_Agung.csv'))
    df_val = pd.read_csv(os.path.join(data_dir, 'val_cleaned_Agung.csv'))
    df_test = pd.read_csv(os.path.join(data_dir, 'test_cleaned_Agung.csv'))

    X_train = pd.concat([df_train['cleaned_text'], df_val['cleaned_text']]).fillna('')
    y_train = pd.concat([df_train['label'], df_val['label']])
    X_test = df_test['cleaned_text'].fillna('')
    y_test = df_test['label']

    # 2. Definisi Pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 3),
            max_df=0.75,
            min_df=2,
            max_features=4500,
        )),
        ('rf', RandomForestClassifier(random_state=42))
    ])

    # 3. Ruang Parameter Tuning
    parameters = {
        'rf__n_estimators': [100, 200, 300],
        'rf__max_depth': [10, 20, None],
        'rf__min_samples_leaf': [5, 10, 20],
        'rf__min_samples_split': [10, 20],
        'rf__max_features': ['sqrt', 'log2'],
        'rf__class_weight': ['balanced', 'balanced_subsample']
    }

    # Eksekusi tuning model
    best_rf_model = run_hyperparameter_tuning(
        pipeline, 
        parameters, 
        X_train, 
        y_train, 
        X_test, 
        y_test
    )
    print("Proses pemodelan selesai dengan MLflow Tracking.")

if __name__ == "__main__":
    # Setup argument parser to capture the parameter forwarded by 'mlflow run'
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path direktori data preprocessing")
    args = parser.parse_args()

    main(args.data_dir)