import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, log_loss, roc_auc_score
)

def run_hyperparameter_tuning(pipeline, parameters, X_train, y_train, X_test, y_test, experiment_name="Sentiment_Analysis_RF"):
    # Deteksi apakah skrip berjalan di bawah 'mlflow run' (Parent Run)
    active_run = mlflow.active_run()
    
    if not active_run:
        mlflow.set_experiment(experiment_name)
    
    # Menggunakan nested=True agar kompatibel dengan workflow CI/CD nanti
    with mlflow.start_run(run_name="RF_Final_Evaluation", nested=True if active_run else False):
        # 1. Inisialisasi & Fit GridSearchCV
        grid_search = GridSearchCV(
            pipeline, 
            parameters, 
            cv=10, 
            scoring='f1_macro', 
            n_jobs=1, # Diatur ke 1 agar stabil di GitHub Runner
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        # 2. Prediksi pada data TRAINING
        y_train_pred = best_model.predict(X_train)
        y_train_proba = best_model.predict_proba(X_train)
        
        # 3. Prediksi pada data TEST (Untuk deteksi overfitting)
        y_test_pred = best_model.predict(X_test)
        y_test_proba = best_model.predict_proba(X_test)
        
        # --- LOGGING METRIK TRAINING ---
        mlflow.log_metric("best_cv_score", grid_search.best_score_)
        mlflow.log_metric("train_accuracy", accuracy_score(y_train, y_train_pred))
        mlflow.log_metric("train_f1", f1_score(y_train, y_train_pred, average='weighted'))
        mlflow.log_metric("train_precision", precision_score(y_train, y_train_pred, average='weighted'))
        mlflow.log_metric("train_recall", recall_score(y_train, y_train_pred, average='weighted'))
        mlflow.log_metric("train_log_loss", log_loss(y_train, y_train_proba))
        mlflow.log_metric("train_roc_auc", roc_auc_score(y_train, y_train_proba, multi_class='ovr', average='weighted'))

        # --- LOGGING METRIK TEST (Kunci Deteksi Overfitting) ---
        mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_test_pred))
        mlflow.log_metric("test_f1", f1_score(y_test, y_test_pred, average='weighted'))
        mlflow.log_metric("test_precision", precision_score(y_test, y_test_pred, average='weighted'))
        mlflow.log_metric("test_recall", recall_score(y_test, y_test_pred, average='weighted'))
        mlflow.log_metric("test_log_loss", log_loss(y_test, y_test_proba))
        mlflow.log_metric("test_roc_auc", roc_auc_score(y_test, y_test_proba, multi_class='ovr', average='weighted'))

        # --- LOG PARAMETERS ---
        for param, value in grid_search.best_params_.items():
            mlflow.log_param(param, value)
            
        # --- LOG MODEL ---
        mlflow.sklearn.log_model(best_model, "random_forest_model")
        
        print("Evaluasi selesai. Bandingkan 'train_f1' dan 'test_f1' di MLflow untuk cek overfitting.")
        return best_model