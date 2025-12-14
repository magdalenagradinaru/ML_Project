
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import warnings
import sys
import codecs

from config import *


warnings.filterwarnings('ignore')

"""
if sys.version_info[0] == 3:
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

"""


#Încarca date preprocesate-------------------------------------------------
def load_data(filepath):
    df = pd.read_excel(filepath)
    df['Data'] = pd.to_datetime(df['Data'])
    df = df.sort_values('Data')
    return df



# Încarca encodere---------------------------------------------------------
def load_encoders(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)




# Calculeaza metrici de evaluare-------------------------------------------------
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # MAPE corect (exclude valori zero)
    mask = y_true > 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}



# Afiseaza metrici-------------------------------------------------
def print_metrics(metrics, model_name):
    print(f"\n{model_name}:")
    print(f"  MAE:  {metrics['MAE']:>8.2f}")
    print(f"  RMSE: {metrics['RMSE']:>8.2f}")
    print(f"  R2:   {metrics['R2']:>8.4f}")
    if not np.isnan(metrics['MAPE']):
        print(f"  MAPE: {metrics['MAPE']:>8.2f}%")





#Creeaza modele----------------------------------------------------------------------------------------------------------------------------
def create_models():
    """Factory pentru toate modelele"""
    return {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(**RF_PARAMS),
        'XGBoost': xgb.XGBRegressor(**XGBOOST_PARAMS),
        'LightGBM': lgb.LGBMRegressor(**LIGHTGBM_PARAMS)
    }

def train_all_models(data, test_months=12, verbose=True):
    """
    Antrenează toate modelele și returnează rezultate complete
    Această funcție ÎNLOCUIEȘTE tot codul duplicat din app.py
    """
    
    # Features
    feature_cols = [
        'An', 'Luna', 'Scop_Encoded',
        'Luna_Sin', 'Luna_Cos', 'Trimestru', 'Sezon', 'Este_Ianuarie',
        'Months_Since_Start', 'Year_Month',
        'Lag_1', 'Lag_2', 'Lag_3', 'Lag_12',
        'Rolling_Mean_3', 'Rolling_Std_3'
    ]
    
    # Split train/test
    split_date = data['Data'].max() - pd.DateOffset(months=test_months)
    train = data[data['Data'] < split_date]
    test = data[data['Data'] >= split_date]
    
    if verbose:
        print(f"\nSplit date: {split_date.strftime('%Y-%m')}")
        print(f"Train: {len(train):,} | Test: {len(test):,}")
    
    X_train = train[feature_cols]
    y_train = train['Numar']
    X_test = test[feature_cols]
    y_test = test['Numar']
    
    # Creare și antrenare modele
    models = create_models()
    results = {}
    
    for name, model in models.items():
        if verbose:
            print(f"Antrenare: {name}")
        
        # Antrenare
        if 'XGBoost' in name:
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        elif 'LightGBM' in name:
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], 
                      callbacks=[lgb.log_evaluation(0)])
        else:
            model.fit(X_train, y_train)
        
        # Predicții
        y_pred = model.predict(X_test)
        y_pred = np.maximum(y_pred, 0)
        
        # Metrici
        metrics = calculate_metrics(y_test, y_pred)
        
        # Feature importance
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = {
                'features': feature_cols,
                'importances': model.feature_importances_.tolist()
            }
        elif hasattr(model, 'coef_'):
            feature_importance = {
                'features': feature_cols,
                'coefficients': model.coef_.tolist()
            }
        
        results[name] = {
            'model': model,
            'metrics': metrics,
            'feature_importance': feature_importance
        }
    
    # Identifică cel mai bun model
    best_model_name = min(results.keys(), key=lambda k: results[k]['metrics']['RMSE'])
    
    return {
        'results': results,
        'best_model_name': best_model_name,
        'split_info': {
            'split_date': split_date.strftime('%Y-%m-%d'),
            'train_size': len(train),
            'test_size': len(test)
        },
        'feature_cols': feature_cols
    }


import json
from pathlib import Path

def save_training_results(training_results, output_dir=OUTPUT_DIR):
    """Salvează rezultatele antrenamentului"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. CSV pentru comparație
    metrics_data = {name: res['metrics'] for name, res in training_results['results'].items()}
    df_metrics = pd.DataFrame(metrics_data).T.sort_values('RMSE')
    df_metrics.to_csv(f'{output_dir}/model_comparison_results.csv')
    
    # 2. Salvează cel mai bun model (pickle)
    best_name = training_results['best_model_name']
    best_model = training_results['results'][best_name]['model']
    
    model_package = {
        'model': best_model,
        'model_name': best_name,
        'metrics': training_results['results'][best_name]['metrics'],
        'feature_cols': training_results['feature_cols'],
        'split_info': training_results['split_info']
    }
    
    with open(f'{output_dir}/best_model.pkl', 'wb') as f:
        pickle.dump(model_package, f)
    
    # 3. JSON pentru frontend
    json_results = {
        'best_model_name': best_name,
        'split_info': training_results['split_info'],
        'feature_cols': training_results['feature_cols'],
        'models': {
            name: {
                'metrics': res['metrics'],
                'feature_importance': res['feature_importance']
            }
            for name, res in training_results['results'].items()
        }
    }
    
    with open(f'{output_dir}/training_results.json', 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    return df_metrics


def load_best_model(filepath=f'{OUTPUT_DIR}/best_model.pkl'):
    """Încarcă modelul salvat"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def load_training_results(filepath=f'{OUTPUT_DIR}/training_results.json'):
    """Încarcă rezultatele JSON"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
    


# Antreneaza si evalueaza model-------------------------------------------------
def train_and_evaluate_model(model, model_name, X_train, y_train, X_test, y_test, feature_cols):
    print(f"Antrenare: {model_name}")
    
    # Antrenare
    if 'XGBoost' in model_name:
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    elif 'LightGBM' in model_name:
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], 
                  callbacks=[lgb.log_evaluation(0)])
    else:
        model.fit(X_train, y_train)
    
    # Predictii
    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)  # Nu permite valori negative
    
    # Evaluare
    metrics = calculate_metrics(y_test, y_pred)
    print_metrics(metrics, model_name)
    
    # Feature importance pentru tree-based models
    if hasattr(model, 'feature_importances_'):
        print("\n  Top 5 Features:")
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1][:5]
        for i, idx in enumerate(indices, 1):
            print(f"    {i}. {feature_cols[idx]}: {importance[idx]:.4f}")
    
    # Pentru Linear Regression, arata coeficienti
    if hasattr(model, 'coef_'):
        coef = model.coef_
        indices = np.argsort(np.abs(coef))[::-1][:5]
        print("\n  Top 5 Features (coeficienti):")
        for i, idx in enumerate(indices, 1):
            print(f"    {i}. {feature_cols[idx]}: {coef[idx]:.4f}")
    
    return model, metrics



# Comparare modele-----------------------------------------------------------------
def compare_models(data, test_months=12):
    """Wrapper pentru compatibilitate cu codul vechi"""
    training_results = train_all_models(data, test_months, verbose=True)
    
    trained_models = {name: res['model'] for name, res in training_results['results'].items()}
    all_metrics = {name: res['metrics'] for name, res in training_results['results'].items()}
    feature_cols = training_results['feature_cols']
    
    return trained_models, all_metrics, feature_cols




# Afiseaza comparatia finala-------------------------------------------------
def print_final_comparison(metrics_dict):
    print("Comparatie finala a modelelor:")
    
    df_results = pd.DataFrame(metrics_dict).T
    df_results = df_results.sort_values('RMSE')
    
    print(df_results.to_string())
    
    return df_results




# Predictii pentru lunile viitoare-------------------------------------------------
def predict_future(model, data, feature_cols, months_ahead=6):
 
    print(f"Predictii pentru {months_ahead} luni")
    
    predictions = []
    last_date = data['Data'].max()
    
    for scop in data['Scop'].unique():
        data_scop = data[data['Scop'] == scop].copy().sort_values('Data')
        
        # Istoricul recent (ultimele 12 valori REALE)
        recent_values = data_scop['Numar'].tail(12).tolist()
        
        # Info pentru features
        scop_encoded = data_scop.iloc[-1]['Scop_Encoded']
        months_since_start_base = data_scop.iloc[-1]['Months_Since_Start']
        
        print(f"\n[{scop}] Ultimele 3 valori: {recent_values[-3:]}")
        
        # Genereaza predictii luna cu luna
        for i in range(1, months_ahead + 1):
            future_date = last_date + pd.DateOffset(months=i)
            luna = future_date.month
            an = future_date.year
            
            # === FEATURES TEMPORALE ===
            luna_sin = np.sin(2 * np.pi * luna / 12)
            luna_cos = np.cos(2 * np.pi * luna / 12)
            trimestru = (luna - 1) // 3 + 1
            sezon = (luna % 12 + 3) // 3
            if sezon == 0:
                sezon = 4
            este_ianuarie = 1 if luna == 1 else 0
            
            # === TREND FEATURES ===
            months_since_start = months_since_start_base + i
            year_month = an + luna / 12
            
            # === LAG FEATURES (CRITIC: Foloseste recent_values) ===
            lag_1 = recent_values[-1] if len(recent_values) >= 1 else 0
            lag_2 = recent_values[-2] if len(recent_values) >= 2 else lag_1
            lag_3 = recent_values[-3] if len(recent_values) >= 3 else lag_1
            lag_12 = recent_values[-12] if len(recent_values) >= 12 else lag_1
            
            # === ROLLING FEATURES ===
            rolling_mean = np.mean(recent_values[-3:]) if len(recent_values) >= 3 else lag_1
            rolling_std = np.std(recent_values[-3:]) if len(recent_values) >= 3 else 0
            
            # Construieste input pentru model
            X_future = pd.DataFrame([{
                'An': an, 'Luna': luna, 'Scop_Encoded': scop_encoded,
                'Luna_Sin': luna_sin, 'Luna_Cos': luna_cos,
                'Trimestru': trimestru, 'Sezon': sezon, 'Este_Ianuarie': este_ianuarie,
                'Months_Since_Start': months_since_start, 'Year_Month': year_month,
                'Lag_1': lag_1, 'Lag_2': lag_2, 'Lag_3': lag_3, 'Lag_12': lag_12,
                'Rolling_Mean_3': rolling_mean, 'Rolling_Std_3': rolling_std
            }])
            
            # Predictie
            pred = model.predict(X_future)[0]
            pred = max(0, round(pred))
            
            # CRUCIAL: Actualizeaza recent_values cu PREDICTIA
            recent_values.append(pred)
            if len(recent_values) > 12:
                recent_values.pop(0)
            
            predictions.append({
                'Data': future_date,
                'An': an,
                'Luna': luna,
                'Scop': scop,
                'Predictie': pred
            })
            
            print(f"  Luna {i}: {future_date.strftime('%Y-%m')} -> {pred:,}")
    
    df_pred = pd.DataFrame(predictions)
    
    print("SUMAR PREDICTII:")
    print(df_pred.groupby('Scop')['Predictie'].agg(['mean', 'min', 'max']).round(0))
    
    return df_pred


# Predictii cu intervale de incredere (doar pentru Random Forest)-------------------------------------------------
def predict_with_confidence(model, data, feature_cols, months_ahead=6):
    """
    Predictii cu intervale de incredere (doar pentru Random Forest)
    """
    if not hasattr(model, 'estimators_'):
        print("\nATENTIE: Intervale de incredere doar pentru Random Forest!")
        return predict_future(model, data, feature_cols, months_ahead)
    
    print(f"\n{'='*70}")
    print(f"PREDICTII CU INTERVALE DE INCREDERE (95%)")
    print(f"{'='*70}")
    
    predictions = []
    last_date = data['Data'].max()
    
    for scop in data['Scop'].unique():
        data_scop = data[data['Scop'] == scop].copy().sort_values('Data')
        recent_values = data_scop['Numar'].tail(12).tolist()
        scop_encoded = data_scop.iloc[-1]['Scop_Encoded']
        months_since_start_base = data_scop.iloc[-1]['Months_Since_Start']
        
        for i in range(1, months_ahead + 1):
            future_date = last_date + pd.DateOffset(months=i)
            luna = future_date.month
            an = future_date.year
            
            # Features (acelasi cod ca in predict_future)
            luna_sin = np.sin(2 * np.pi * luna / 12)
            luna_cos = np.cos(2 * np.pi * luna / 12)
            trimestru = (luna - 1) // 3 + 1
            sezon = (luna % 12 + 3) // 3
            if sezon == 0:
                sezon = 4
            este_ianuarie = 1 if luna == 1 else 0
            months_since_start = months_since_start_base + i
            year_month = an + luna / 12
            
            lag_1 = recent_values[-1] if len(recent_values) >= 1 else 0
            lag_2 = recent_values[-2] if len(recent_values) >= 2 else lag_1
            lag_3 = recent_values[-3] if len(recent_values) >= 3 else lag_1
            lag_12 = recent_values[-12] if len(recent_values) >= 12 else lag_1
            rolling_mean = np.mean(recent_values[-3:]) if len(recent_values) >= 3 else lag_1
            rolling_std = np.std(recent_values[-3:]) if len(recent_values) >= 3 else 0
            
            X_future = pd.DataFrame([{
                'An': an, 'Luna': luna, 'Scop_Encoded': scop_encoded,
                'Luna_Sin': luna_sin, 'Luna_Cos': luna_cos,
                'Trimestru': trimestru, 'Sezon': sezon, 'Este_Ianuarie': este_ianuarie,
                'Months_Since_Start': months_since_start, 'Year_Month': year_month,
                'Lag_1': lag_1, 'Lag_2': lag_2, 'Lag_3': lag_3, 'Lag_12': lag_12,
                'Rolling_Mean_3': rolling_mean, 'Rolling_Std_3': rolling_std
            }])
            
            # Predictii de la fiecare arbore
            tree_predictions = [max(0, tree.predict(X_future)[0]) for tree in model.estimators_]
            
            pred_mean = np.mean(tree_predictions)
            pred_std = np.std(tree_predictions)
            lower_95 = max(0, pred_mean - 1.96 * pred_std)
            upper_95 = pred_mean + 1.96 * pred_std
            
            # Actualizeaza istoric
            recent_values.append(pred_mean)
            if len(recent_values) > 12:
                recent_values.pop(0)
            
            predictions.append({
                'Data': future_date,
                'An': an,
                'Luna': luna,
                'Scop': scop,
                'Predictie': round(pred_mean),
                'Lower_95': round(lower_95),
                'Upper_95': round(upper_95)
            })
    
    df_pred = pd.DataFrame(predictions)
    print("\n" + df_pred.to_string(index=False))
    
    return df_pred



# Main=========================================================================
if __name__ == "__main__":
    
    # Incarcare date
    print("\n1. Incarcare date...")
    data = load_data(FISIER_PREPROCESATE)
    encoders = load_encoders(FISIER_ENCODERS)
    print(f" {len(data):,} inregistrari")
    
    # Antrenament si comparatie
    print("\n2. Antrenament modele...")
    trained_models, all_metrics, feature_cols = compare_models(data, TEST_MONTHS)
    
    results_df = print_final_comparison(all_metrics)
    results_df.to_csv(f'{OUTPUT_DIR}/model_comparison_results.csv')
    print(f"\n Rezultate salvate: model_comparison_results.csv")
    
    # Selectare cel mai bun model
    best_model_name = results_df.index[0]
    best_model = trained_models[best_model_name]
    
    # Predictii viitoare
    print("\n3. Generare predictii...")
    future_pred = predict_future(best_model, data, feature_cols, MONTHS_AHEAD)
    future_pred.to_csv(f'{OUTPUT_DIR}/future_predictions.csv', index=False)
    print(f"\n Salvat: future_predictions.csv")
    
    # Intervale de incredere (daca e Random Forest)
    if best_model_name == 'Random Forest':
        print("\n4. Generare predictii cu intervale de incredere...")
        future_conf = predict_with_confidence(best_model, data, feature_cols, MONTHS_AHEAD)
        future_conf.to_csv(f'{OUTPUT_DIR}/future_predictions_confidence.csv', index=False)
        print(f"\n Salvat: future_predictions_confidence.csv")
    else:
        print("\n4. Skip intervale de incredere (doar pentru Random Forest)")
    