"""
process_files.py - Procesare Primara Date Brute

Functionalitate:
- Citeste fisiere Excel din director
- Converteste date cumulative in date lunare
- Creeaza dataset complet (cu tari)
- Creeaza dataset agregat (fara tari)
"""

import pandas as pd
from pathlib import Path
import re
import unicodedata
import sys
import codecs

if sys.version_info[0] == 3:
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')



# Elimina diacritice si spatii extra--------------------------------------------------
def normalize_text(text):
    if pd.isna(text):    
        return text
    text = str(text).strip() 
    # Elimina diacritice (ex: ä -> a, ș -> s)   
    text = ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')
    # Inlocuieste spatiile multiple cu unul singur
    text = ' '.join(text.split())
    return text


# Extrage anul si luna din numele fisierului-------------------------------------------------
def extract_month_from_filename(filename):
    # Inlocuieste extensiile si caractere speciale
    name = filename.replace('.xlsx', '').replace('.xls', '').replace('--', '-')
    matches = re.findall(r'(\d{2})\.(\d{2})\.(\d{4})', name)
    if len(matches) >= 2:
        day, month, year = matches[1] # Luam a doua aparitie
        return int(year), int(month)
    return None, None


# Verifica daca fisierul incepe cu 01.01-------------------------------------------------
def is_valid_file(filename):
    return filename.startswith('01.01')


# Citeste fisierul Excel si returneaza DataFrame in format long-------------------------------------------------
def read_excel_file(file_path, year, month):
    df = pd.read_excel(file_path, header=0)
    
    country_col = df.columns[0]
    scope_cols = [col for col in df.columns[1:] 
                  if 'total' not in str(col).lower() 
                  and not str(col).startswith('Unnamed')]
    
    # Filtreaza randurile fara tara valida
    df = df[df[country_col].notna()]
    # Elimina randurile cu valori nedorite (total, gol, etc.)
    df = df[~df[country_col].astype(str).str.lower().str.strip().isin(
        ['', 'total', 'nu este definit']
    )]
    
    # Converteste in format long
    df_long = df.melt(
        id_vars=[country_col],
        value_vars=scope_cols,  # Devin valori
        var_name='Scop',
        value_name='Numar'
    )
    
    df_long.rename(columns={country_col: 'Tara'}, inplace=True)
    df_long['An'] = year
    df_long['Luna'] = month
    df_long['Numar'] = pd.to_numeric(df_long['Numar'], errors='coerce').fillna(0).astype(int)
    df_long['Tara'] = df_long['Tara'].apply(normalize_text)
    df_long['Scop'] = df_long['Scop'].apply(normalize_text)
    
    return df_long[['An', 'Luna', 'Tara', 'Scop', 'Numar']]



# Combina toate fisierele Excel valide dintr-un director-------------------------------------------------
def combine_files(data_directory):
    data_dir = Path(data_directory)
    all_data = []
    files = sorted(data_dir.glob("*.xlsx"))
    
    print(f"Gasite {len(files)} fisiere")
    
    # Parcurge fiecare fisier si citeste datele valide
    for file_path in files:
        filename = file_path.name
        if not is_valid_file(filename):
            continue
        
        year, month = extract_month_from_filename(filename)
        if not year or not month:
            continue
        
        try:
            df = read_excel_file(file_path, year, month)
            all_data.append(df)
        except:
            continue
    
    if not all_data:
        return None
    
    # Combina toate DataFrame-urile
    combined = pd.concat(all_data, ignore_index=True)

    combined.sort_values(['An', 'Luna', 'Tara', 'Scop'], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    
    return combined


# Converteste datele cumulative in date lunare-------------------------------------------------
def convert_cumulative_to_monthly(df):
    # Sorteaza dupa Tara, Scop, An, Luna
    df_sorted = df.sort_values(['Tara', 'Scop', 'An', 'Luna']).copy()
    
    # Calculeaza diferenta pentru a obtine valorile lunare
    df_sorted['Numar_Lunar'] = df_sorted.groupby(['Tara', 'Scop', 'An'])['Numar'].diff()
    
    # Pentru prima luna a fiecarui grup, pastreaza valoarea originala
    first_month_mask = df_sorted.groupby(['Tara', 'Scop', 'An']).cumcount() == 0
    df_sorted.loc[first_month_mask, 'Numar_Lunar'] = df_sorted.loc[first_month_mask, 'Numar']
    
    # Inlocuieste valorile negative cu 0 
    df_sorted.loc[df_sorted['Numar_Lunar'] < 0, 'Numar_Lunar'] = 0
    
    df_sorted['Numar'] = df_sorted['Numar_Lunar'].astype(int)
    df_sorted = df_sorted.drop('Numar_Lunar', axis=1)
    
    return df_sorted.reset_index(drop=True)



# Creeaza dataset agregat fara tara-------------------------------------------------
# Grupeaza dupa An, Luna, Scop si insumeaza valorile
def create_aggregated(df):
    return df.groupby(['An', 'Luna', 'Scop'])['Numar'].sum().reset_index()








# Functia principala pentru testare=======================================================
if __name__ == "__main__":
    DATA_DIR = r"C:\Users\user\Proiect_Sem\DateProjSem"
    
    print("Incarcare si combinare fisiere...")
    data = combine_files(DATA_DIR)
    
    if data is None:
        print("Nu s-au gasit date!")
        exit()


    
    print("Conversie cumulative -> monthly...")
    data = convert_cumulative_to_monthly(data)

    
    print(f"Total inregistrari: {len(data):,}")

    
    data.to_excel("date_lunare.xlsx", index=False)
    print("Salvat: date_lunare.xlsx")

    
    agg = create_aggregated(data)
    agg.to_excel("date_agregat.xlsx", index=False)
    print("Salvat: date_agregat.xlsx")