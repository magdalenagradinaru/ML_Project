# Script pentru preprocesarea datelor înainte de aplicarea algoritmilor de Machine Learning


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import sys
import codecs

from config import *


if sys.version_info[0] == 3:
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')



# Adauga features temporale pentru sezonalitate-------------------------------------------------
def add_time_features(df):
    df = df.copy()
    
    # Transformare ciclică a lunii folosind funcții trigonometrice
    # Acest lucru captează natura circulară a anului (luna 12 este aproape de luna 1)

    df['Luna_Sin'] = np.sin(2 * np.pi * df['Luna'] / 12)
    df['Luna_Cos'] = np.cos(2 * np.pi * df['Luna'] / 12)

    # Gruparea lunilor în trimestre (1-4)
    df['Trimestru'] = ((df['Luna'] - 1) // 3 + 1).astype(int)
    
    # Sezon: 1-Iarna, 2-Primavara, 3-Vara, 4-Toamna
    df['Sezon'] = pd.cut(df['Luna'], 
                        bins=[0, 2, 5, 8, 11, 12],
                        labels=[1, 2, 3, 4, 1])    
    # Indicator binar pentru luna ianuarie (poate fi important pentru trenduri post-festive)
    df['Este_Ianuarie'] = (df['Luna'] == 1).astype(int)
    
    return df




# Adauga caracteristici de întârziere - valori din lunile anterioare-------------------------------------------------
def add_lag_features(df):
    df = df.sort_values(['Scop', 'An', 'Luna']).copy()
    
    # Lag_1 = valoarea de acum 1 lună, Lag_12 = valoarea de acum 1 an 
    for lag in [1, 2, 3, 12]:
        df[f'Lag_{lag}'] = df.groupby(['Scop'])['Numar'].shift(lag)
    
    # Media mobilă pe 3 luni
    df['Rolling_Mean_3'] = (
        df.groupby(['Scop'])['Numar']
        .shift(1) # pentru a nu include luna curentă
        .rolling(window=3, min_periods=1)
        .mean()
    )
    
    # Deviatia standard mobila pe 3 luni
    df['Rolling_Std_3'] = (
        df.groupby(['Scop'])['Numar']
        .shift(1)
        .rolling(window=3, min_periods=1)
        .std()
        .fillna(0)
    )
    
    return df




# Adauga caracteristici pentru capturarea tendintelor-------------------------------------------------
def add_trend_features(df):
    df = df.copy()
    
    # Creează coloana Data pentru manipulări temporale mai ușoare
    df['Data'] = pd.to_datetime(df[['An', 'Luna']].rename(columns={'An': 'year', 'Luna': 'month'}).assign(day=1))
    df['Months_Since_Start'] = df.groupby(['Scop']).cumcount()
    df['Year_Month'] = df['An'] + df['Luna'] / 12
    
    return df




# Codeaza variabilele categoriale-------------------------------------------------
def encode_categorical(df):
    df = df.copy()
    
    # Pentru date agregate, encodam doar Scop (nu mai avem Tara)
    le_scop = LabelEncoder()
    df['Scop_Encoded'] = le_scop.fit_transform(df['Scop'])
    
    return df, le_scop



# Curata datele incomplete-------------------------------------------------
def clean_incomplete_data(df):
   
    initial_count = len(df)
    
    # Elimina randurile unde lipsesc lag features critice
    critical_lags = ['Lag_1', 'Lag_12', 'Rolling_Mean_3']
    df_clean = df.dropna(subset=critical_lags).copy()
    
    removed_count = initial_count - len(df_clean)
    
    print(f"  Inainte: {initial_count:,} randuri")
    print(f"  Dupa:   {len(df_clean):,} randuri")
    print(f"  Eliminate: {removed_count:,} randuri cu date incomplete")
    
    return df_clean.reset_index(drop=True)



# Pregateste datasetul final pentru ML-------------------------------------------------
def prepare_ml_dataset(df):
    feature_columns = [
        'Data',
        'An', 'Luna', 'Scop',
        'Scop_Encoded',
        'Numar',
        'Luna_Sin', 'Luna_Cos', 'Trimestru', 'Sezon', 'Este_Ianuarie',
        'Months_Since_Start', 'Year_Month',
        'Lag_1', 'Lag_2', 'Lag_3', 'Lag_12',
        'Rolling_Mean_3', 'Rolling_Std_3'
    ]
    
    return df[feature_columns].copy()



# Salveaza encoderele-------------------------------------------------
import pickle
def save_encoders(le_scop, output_dir):
    encoders = {'scop': le_scop}

    filepath = f'{output_dir}/label_encoders.pkl'
    with open(filepath, 'wb') as f:
        pickle.dump(encoders, f)
    
    return filepath





# Main ==================================================================================
if __name__ == "__main__":
    
    print("\n1. Incarcare date agregate...")
    try:
        df = pd.read_excel(FISIER_DATE_AGREGAT)
        print(f"   {len(df):,} inregistrari incarcate")
    except FileNotFoundError:
        print(f"   EROARE: {FISIER_DATE_AGREGAT} nu exista!")
        print("  Rulati mai intai process_files.py")
        sys.exit(1)
    
    print("\n2. Adaugare features temporale...")
    df = add_time_features(df)

    print("\n3. Adaugare trend features...")
    df = add_trend_features(df)
    
    print("\n4. Adaugare lag features...")
    df = add_lag_features(df)
    
    print("\n5.Encoding categorial...")
    df, le_scop = encode_categorical(df)
    print(f"Scop -> Scop_Encoded ({len(le_scop.classes_)} categorii)")
    
    print("\n6.Pregatire si curatare dataset...")
    df_ml = prepare_ml_dataset(df)
    df_ml = clean_incomplete_data(df_ml)
    
    # Salvare
    print("Rezultate:")
    print(f"  Inregistrari: {len(df_ml):,}")
    print(f"  Features: {len(df_ml.columns)}")
    print(f"  Perioada: {df_ml['Data'].min().strftime('%Y-%m')} -> {df_ml['Data'].max().strftime('%Y-%m')}")
    
    df_ml.to_excel(FISIER_PREPROCESATE, index=False)
    print(f"\nSalvat: {FISIER_PREPROCESATE}")
    
    encoder_path = save_encoders(le_scop, OUTPUT_DIR)
    print(f"Salvat: {encoder_path}")
    
  