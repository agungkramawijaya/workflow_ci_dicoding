import os
import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

def main(data_dir):
    # Penyesuaian Path agar fleksibel (CI/CD friendly)
    train_path = os.path.join(data_dir, 'train_cleaned_Agung.csv')
    val_path = os.path.join(data_dir, 'val_cleaned_Agung.csv')

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    X_train = pd.concat([df_train['cleaned_text'], df_val['cleaned_text']]).fillna('')
    y_train = pd.concat([df_train['label'], df_val['label']])

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('rf', RandomForestClassifier(random_state=42))
    ])

    # Tracking URI akan otomatis diambil dari environment variable di GitHub Actions
    mlflow.set_experiment("Sentiment_Analysis_Baseline")
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="Baseline_Model_CI"):
        pipeline.fit(X_train, y_train)
        print("Model baseline berhasil dilatih di lingkungan CI.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    args = parser.parse_args()
    main(args.data_dir)