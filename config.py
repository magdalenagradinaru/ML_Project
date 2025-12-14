#config.py - Configurare centralizata pentru proiect

# Cale directoare si fisiere
DATA_DIR = r"C:\Users\user\Proiect_Sem\DateProjSem"
OUTPUT_DIR = r"C:\Users\user\Proiect_Sem"

FISIER_DATE_LUNARE = f"{OUTPUT_DIR}/date_lunare.xlsx"
FISIER_DATE_AGREGAT = f"{OUTPUT_DIR}/date_agregat.xlsx"
FISIER_PREPROCESATE = f"{OUTPUT_DIR}/date_preprocesate_ml.xlsx"
FISIER_ENCODERS = f"{OUTPUT_DIR}/label_encoders.pkl"
FISIER_RESULTS = f"{OUTPUT_DIR}/model_comparison_results.csv"
FISIER_PREDICTIONS = f"{OUTPUT_DIR}/future_predictions.csv"
FISIER_CONFIDENCE = f"{OUTPUT_DIR}/future_predictions_confidence.csv"



#Parametri modele ML--------------------------------------------------------
# Random Forest
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 5,
    'random_state': 42,
    'n_jobs': -1
}

# XGBoost
XGBOOST_PARAMS = {
    'n_estimators': 500,
    'max_depth': 5,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'verbosity': 0
}

# LightGBM
LIGHTGBM_PARAMS = {
    'n_estimators': 500,
    'max_depth': 5,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'verbose': -1
}

# SARIMA
SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (1, 1, 1, 12)





# Parametri generali ML--------------------------------------------------------
TEST_MONTHS = 12
CV_SPLITS = 5
MONTHS_AHEAD = 6
MIN_TRAIN_SAMPLES = 24