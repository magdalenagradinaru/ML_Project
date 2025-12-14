import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

from config import *

# IMPORT FUNCȚII BACKEND 
from model_ml import (
    # Data loading
    load_data,
    load_encoders,
    
    # Training (ADĂUGATE!)
    train_all_models,
    save_training_results,
    
    # Persistence
    load_best_model,
    load_training_results,
    
    # Predictions
    predict_future,
    predict_with_confidence
)


st.set_page_config(
    page_title="Analiza Imigranti Moldova",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)





#Încarcă toate datele necesare pentru frontend (cached)
@st.cache_data
def load_all_data():
    try:
        # 1. Date lunare (cu țări)
        date_lunare = pd.read_excel(FISIER_DATE_LUNARE)
        date_lunare['Data'] = pd.to_datetime(
            date_lunare[['An', 'Luna']]
            .rename(columns={'An': 'year', 'Luna': 'month'})
            .assign(day=1)
        )

        # 2. Date agregate (fără țări)
        date_agregat = pd.read_excel(FISIER_DATE_AGREGAT)
        date_agregat['Data'] = pd.to_datetime(
            date_agregat[['An', 'Luna']]
            .rename(columns={'An': 'year', 'Luna': 'month'})
            .assign(day=1)
        )

        # 3. Date pentru ML - FOLOSIND FUNCȚIA DIN BACKEND
        date_ml = load_data(FISIER_PREPROCESATE)
        
        # 4. Encoders - FOLOSIND FUNCȚIA DIN BACKEND
        encoders = load_encoders(FISIER_ENCODERS)

        return date_lunare, date_agregat, date_ml, encoders

    except Exception as e:
        st.error(f"Eroare la încărcarea datelor: {e}")
        st.info("Asigurați-vă că ați rulat process_files.py și procesare.py")
        return None, None, None, None


#Încarcă rezultate antrenament (cached)
@st.cache_data
def load_trained_results():
    results_path = Path(f'{OUTPUT_DIR}/training_results.json')
    
    if results_path.exists():
        return load_training_results(str(results_path))
    return None


#Încarcă model salvat (cached)
@st.cache_data
def load_saved_model():
    model_path = Path(f'{OUTPUT_DIR}/best_model.pkl')
    
    if model_path.exists():
        return load_best_model(str(model_path))
    return None


# SIDEBAR NAVIGATION==============================================================================

def sidebar_navigation():
    st.sidebar.title("Navigare")
    st.sidebar.markdown("---")
    
    sectiune = st.sidebar.radio(
        "Selectează secțiunea:",
        ["Acasă", "Vizualizări Date", "Antrenament Modele", "Predicții"],
        index=0
    )
    st.sidebar.markdown("---")
    
    # Info despre date
    date_lunare, _, _, _ = load_all_data()
    if date_lunare is not None:
        st.sidebar.info(f"""
        **Date disponibile:**
        - {len(date_lunare):,} înregistrări
        - {date_lunare['Tara'].nunique()} țări
        - {date_lunare['Scop'].nunique()} scopuri
        """)
    
    # Status model
    trained_results = load_trained_results()
    if trained_results:
        st.sidebar.success(f"""
        ** Model antrenat:**
        {trained_results['best_model_name']}
        """)
    else:
        st.sidebar.warning(" Niciun model antrenat")
    
    return sectiune


# PAGINA ACASĂ==============================================================================

def pagina_acasa():
    st.title("Analiza Imigranti Republica Moldova")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    # CORECTARE: load_all_data() în loc de load_data()
    date_lunare, date_agregat, _, _ = load_all_data()
    
    if date_lunare is not None:
        with col1:
            st.metric(
                "Total Inregistrari",
                f"{len(date_lunare):,}",
                "Date lunare cu tari"
            )
        
        with col2:
            st.metric(
                "Tari Unice",
                f"{date_lunare['Tara'].nunique()}",
                "Diversitate surse"
            )
        
        with col3:
            st.metric(
                "Scopuri Unice",
                f"{date_lunare['Scop'].nunique()}",
                "Categorii imigrare"
            )
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Perioada Acoperita")
            st.write(f"**De la:** {date_lunare['Data'].min().strftime('%B %Y')}")
            st.write(f"**Pana la:** {date_lunare['Data'].max().strftime('%B %Y')}")
            st.write(f"**Durata:** {(date_lunare['Data'].max() - date_lunare['Data'].min()).days // 30} luni")
        
        with col2:
            st.subheader("Statistici Rapide")
            st.write(f"**Total imigranti:** {date_lunare['Numar'].sum():,}")
            st.write(f"**Media lunara:** {date_lunare['Numar'].mean():.0f}")
            st.write(f"**Maxim lunar:** {date_lunare['Numar'].max():,}")
        
        # Status model
        st.markdown("---")
        trained_results = load_trained_results()
        
        if trained_results:
            st.success("**Model ML antrenat și disponibil pentru predicții**")
            
            col1, col2, col3 = st.columns(3)
            best_metrics = trained_results['models'][trained_results['best_model_name']]['metrics']
            
            with col1:
                st.metric("Model", trained_results['best_model_name'])
            with col2:
                st.metric("R² Score", f"{best_metrics['R2']:.4f}")
            with col3:
                st.metric("RMSE", f"{best_metrics['RMSE']:.2f}")
        else:
            st.warning("**Niciun model antrenat** - Accesează secțiunea 'Antrenament Modele'")





# PAGINA VIZUALIZĂRI DATE=============================================================================
# PAGINA VIZUALIZĂRI EXTINSĂ - ÎNLOCUIEȘTE pagina_vizualizari() din app.py
# ====================================================================================================

def pagina_vizualizari():
    st.title("Vizualizări Date - Comparație Înainte/După Procesare")
    st.markdown("---")
    
    # Încarcă TOATE fișierele
    try:
        # Date originale
        df_original = pd.read_excel("date_agregat.xlsx")
        df_original['Data'] = pd.to_datetime(
            df_original['An'].astype(str) + '-' + df_original['Luna'].astype(str) + '-01'
        )
        
        # Date completate
        df_completat = pd.read_excel("date_agregat_completate.xlsx")
        df_completat['Data'] = pd.to_datetime(
            df_completat['An'].astype(str) + '-' + df_completat['Luna'].astype(str) + '-01'
        )
        
        # Date ML (cu features)
        df_ml = pd.read_excel("date_preprocesate_ml.xlsx")
        df_ml['Data'] = pd.to_datetime(df_ml['Data'])
        
        # Date complete cu țări (pentru analiză suplimentară)
        df_tari = pd.read_excel("date_lunare_completate.xlsx")
        df_tari['Data'] = pd.to_datetime(
            df_tari['An'].astype(str) + '-' + df_tari['Luna'].astype(str) + '-01'
        )
        
    except FileNotFoundError as e:
        st.error(f"Eroare: {e}")
        st.info("Rulați notebook-ul de procesare pentru a genera toate fișierele necesare.")
        return
    
    # TABS PRINCIPALE
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Comparație Date",
        "Date Adăugate",
        "Features ML",
        "Analiză Scopuri",
        "Analiză Țări"    ])
    
    # TAB 1: COMPARAȚIE ÎNAINTE/DUPĂ
    with tab1:
        st.header("Comparație Date Originale vs Procesate")
        
        
        # Grafic comparativ - Evoluție temporală        
        scop_selectat = st.selectbox(
            "Selectează scopul pentru comparație:",
            sorted(df_original['Scop'].unique()),
            key="comp_scop"
        )
        
        # Pregătește date pentru grafic
        data_original = df_original[df_original['Scop'] == scop_selectat].groupby('Data')['Numar'].sum().reset_index()
        data_completat = df_completat[df_completat['Scop'] == scop_selectat].groupby('Data')['Numar'].sum().reset_index()
        data_ml = df_ml[df_ml['Scop'] == scop_selectat].groupby('Data')['Numar'].sum().reset_index()
        
        fig = go.Figure()
        
        # Original
        fig.add_trace(go.Scatter(
            x=data_original['Data'],
            y=data_original['Numar'],
            mode='lines+markers',
            name='Original',
            line=dict(color='lightgray', width=2),
            marker=dict(size=6)
        ))
        
        # Completat
        fig.add_trace(go.Scatter(
            x=data_completat['Data'],
            y=data_completat['Numar'],
            mode='lines+markers',
            name='Completat',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))
        
        # ML (cu features)
        fig.add_trace(go.Scatter(
            x=data_ml['Data'],
            y=data_ml['Numar'],
            mode='lines+markers',
            name='ML Ready',
            line=dict(color='green', width=2, dash='dot'),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title=f"Comparație Date: {scop_selectat}",
            xaxis_title="Data",
            yaxis_title="Număr Imigranti",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)



        
        # Tabel comparativ statistici
        st.markdown("---")
        st.subheader("Statistici Comparative")
        
        stats_comp = pd.DataFrame({
            'Metrică': ['Total Înregistrări', 'Luni Unice', 'Scopuri', 'Total Imigranti', 'Media/Lună'],
            'Original': [
                len(df_original),
                df_original[['An', 'Luna']].drop_duplicates().shape[0],
                df_original['Scop'].nunique(),
                df_original['Numar'].sum(),
                df_original['Numar'].mean()
            ],
            'Completat': [
                len(df_completat),
                df_completat[['An', 'Luna']].drop_duplicates().shape[0],
                df_completat['Scop'].nunique(),
                df_completat['Numar'].sum(),
                df_completat['Numar'].mean()
            ],
            'ML Ready': [
                len(df_ml),
                df_ml[['An', 'Luna']].drop_duplicates().shape[0],
                df_ml['Scop'].nunique(),
                df_ml['Numar'].sum(),
                df_ml['Numar'].mean()
            ]
        })
        
        st.dataframe(
            stats_comp.style.format({
                'Original': '{:,.0f}',
                'Completat': '{:,.0f}',
                'ML Ready': '{:,.0f}'
            }),
            use_container_width=True
        )
    
    # TAB 2: DATE ADĂUGATE
    with tab2:
        st.header("Date Adăugate prin Completare")
        
        # Identifică datele noi
        original_keys = set(zip(df_original['An'], df_original['Luna'], df_original['Scop']))
        completat_keys = set(zip(df_completat['An'], df_completat['Luna'], df_completat['Scop']))
        date_noi = completat_keys - original_keys
        
        if date_noi:
            st.success(f"Au fost adăugate {len(date_noi)} înregistrări noi")
            
            # Conversie la DataFrame
            df_noi = df_completat[
                df_completat.apply(
                    lambda row: (row['An'], row['Luna'], row['Scop']) in date_noi,
                    axis=1
                )
            ].sort_values(['An', 'Luna', 'Scop'])
            
            # Grupează pe luni
            luni_noi = df_noi[['An', 'Luna']].drop_duplicates().sort_values(['An', 'Luna'])
            
            st.subheader("Luni Completate")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.info(f"**{len(luni_noi)} luni completate:**")
                for _, row in luni_noi.iterrows():
                    total = df_noi[(df_noi['An'] == row['An']) & (df_noi['Luna'] == row['Luna'])]['Numar'].sum()
                    st.write(f"• {row['An']}-{row['Luna']:02d}: {total:,} imigranti")
            
            with col2:
                # Grafic distributie pe scopuri pentru lunile noi
                scop_dist = df_noi.groupby('Scop')['Numar'].sum().sort_values(ascending=False)
                
                fig = px.bar(
                    x=scop_dist.values,
                    y=scop_dist.index,
                    orientation='h',
                    title="Distribuție Date Noi pe Scopuri",
                    labels={'x': 'Număr Estimat', 'y': 'Scop'},
                    color=scop_dist.values,
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.subheader("Detalii Date Adăugate")
            
            # Tabel cu date noi
            df_noi_display = df_noi[['An', 'Luna', 'Scop', 'Numar']].copy()
            df_noi_display['Luna_Nume'] = df_noi_display['Luna'].map({
                1: 'Ian', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'Mai', 6: 'Iun',
                7: 'Iul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Noi', 12: 'Dec'
            })
            
            st.dataframe(
                df_noi_display[['An', 'Luna_Nume', 'Scop', 'Numar']],
                use_container_width=True,
                height=400
            )
            
            
     
    # TAB 3: FEATURES ML
    
    with tab3:
        
    # Lista features
    
        st.subheader("Lista Features")
        
        feature_categories = {
            'Temporale': ['An', 'Luna', 'Data', 'Luna_Sin', 'Luna_Cos', 'Trimestru', 'Sezon', 'Este_Ianuarie'],
            'Trend': ['Months_Since_Start', 'Year_Month'],
            'Lag': ['Lag_1', 'Lag_2', 'Lag_3', 'Lag_12'],
            'Rolling': ['Rolling_Mean_3', 'Rolling_Std_3'],
            'Target & Categorial': ['Scop', 'Scop_Encoded', 'Numar']
        }
        
        for category, features in feature_categories.items():
            with st.expander(f"{category} ({len(features)})"):
                for feat in features:
                    if feat in df_ml.columns:
                        st.write(f"• {feat}")



        # Vizualizare LAG features
        st.subheader("Vizualizare Lag Features")
        
        scop_lag = st.selectbox(
            "Selectează scopul:",
            sorted(df_ml['Scop'].unique()),
            key="lag_viz"
        )
        
        df_scop = df_ml[df_ml['Scop'] == scop_lag].sort_values('Data').tail(24)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_scop['Data'],
            y=df_scop['Numar'],
            mode='lines+markers',
            name='Numar',
            line=dict(color='blue', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=df_scop['Data'],
            y=df_scop['Lag_1'],
            mode='lines+markers',
            name='Lag_1',
            line=dict(color='orange', dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=df_scop['Data'],
            y=df_scop['Lag_12'],
            mode='lines+markers',
            name='Lag_12',
            line=dict(color='green', dash='dot')
        ))
        
        fig.add_trace(go.Scatter(
            x=df_scop['Data'],
            y=df_scop['Rolling_Mean_3'],
            mode='lines',
            name='Rolling Mean 3',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title=f"Lag Features pentru: {scop_lag} (ultimele 24 luni)",
            xaxis_title="Data",
            yaxis_title="Valoare",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
      
    
    # ====================================================================================================
    # TAB 4: ANALIZĂ SCOPURI
    # ====================================================================================================
    with tab4:
        st.header("Analiză Detaliată pe Scopuri")
        
        # Evoluție multi-scop
        st.subheader("Evoluție Comparativă")
        
        scopuri_compare = st.multiselect(
            "Selectează scopuri pentru comparație (max 5):",
            sorted(df_completat['Scop'].unique()),
            default=sorted(df_completat['Scop'].unique())[:3],
            key="multi_scop"
        )
        
        if scopuri_compare:
            fig = go.Figure()
            
            for scop in scopuri_compare[:5]:
                data_scop = df_completat[df_completat['Scop'] == scop].groupby('Data')['Numar'].sum().reset_index()
                
                fig.add_trace(go.Scatter(
                    x=data_scop['Data'],
                    y=data_scop['Numar'],
                    mode='lines+markers',
                    name=scop,
                    line=dict(width=2)
                ))
            
            fig.update_layout(
                xaxis_title="Data",
                yaxis_title="Număr Imigranti",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    


    
    # TAB 5: ANALIZĂ ȚĂRI
    with tab5:
        st.header("Analiză pe Țări")
        
        
        st.subheader("Top Țări")
        top_tari = df_tari.groupby('Tara')['Numar'].sum().sort_values(ascending=False).head(15)
        
        fig = px.bar(
            x=top_tari.values,
            y=top_tari.index,
            orientation='h',
            title="Top 15 Țări",
            labels={'x': 'Total Imigranti', 'y': 'Țară'},
            color=top_tari.values,
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
        

        
        st.markdown("---")
        
        # Analiză per țară
        st.subheader("Analiză Detaliată per Țară")
        
        tara_selectata = st.selectbox(
            "Selectează țara:",
            sorted(df_tari['Tara'].unique()),
            key="tara_detail"
        )
        
        df_tara = df_tari[df_tari['Tara'] == tara_selectata]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Imigranti", f"{df_tara['Numar'].sum():,}")
        
        with col2:
            st.metric("Scopuri Active", df_tara['Scop'].nunique())
        
        with col3:
            st.metric("Media/Lună", f"{df_tara['Numar'].mean():.0f}")
        
        # Evoluție temporală
        data_tara_time = df_tara.groupby('Data')['Numar'].sum().reset_index()
        
        fig = px.line(
            data_tara_time,
            x='Data',
            y='Numar',
            title=f"Evoluție: {tara_selectata}",
            markers=True
        )
        fig.update_layout(hovermode='x unified', height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribuție pe scopuri pentru țara selectată
        scop_tara = df_tara.groupby('Scop')['Numar'].sum().sort_values(ascending=False)
        
        fig = px.bar(
            x=scop_tara.values,
            y=scop_tara.index,
            orientation='h',
            title=f"Scopuri pentru {tara_selectata}",
            labels={'x': 'Număr', 'y': 'Scop'},
            color=scop_tara.values,
            color_continuous_scale='Greens'
        )
        st.plotly_chart(fig, use_container_width=True)
    





# PAGINA ANTRENAMENT=============================================================================
def pagina_antrenament():
    st.title("Antrenament Modele Machine Learning")
    st.markdown("---")
    
    _, _, date_ml, _ = load_all_data()
    
    if date_ml is None:
        return
    
    # Verifică dacă există model antrenat
    trained_results = load_trained_results()
    
    if trained_results:
        st.info(f"""
         **Model deja antrenat:** {trained_results['best_model_name']}
        
        Poți antrena din nou sau vizualiza rezultatele existente.
        """)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info("Se vor antrena 4 modele: Linear Regression, Random Forest, XGBoost, LightGBM")
    
    with col2:
        retrain = st.button(" (Re)Antrenează", type="primary", use_container_width=True)
    
    
    
    
    # ANTRENAMENT - APEL BACKEND
    if retrain:
        with st.spinner("Antrenare în curs..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Pregătire date...")
            progress_bar.progress(10)
            
            status_text.text("Antrenare modele...")
            progress_bar.progress(30)
            
            # APEL BACKEND - SINGURA LINIE
            training_results = train_all_models(date_ml, TEST_MONTHS, verbose=False)
            
            progress_bar.progress(80)
            status_text.text("Salvare rezultate...")
            
            save_training_results(training_results)
            
            progress_bar.progress(100)
            status_text.text("Antrenare completă!")
            
            st.cache_data.clear()
            st.success("Modele antrenate cu succes!")
            st.rerun()
    




    # Rezultate antrenament==============================================================================
    if trained_results:
        st.markdown("---")
        st.subheader("Rezultate Antrenament")
        
        # Pregătire date pentru afișare
        models_data = trained_results['models']
        df_results = pd.DataFrame({
            name: data['metrics'] 
            for name, data in models_data.items()
        }).T.sort_values('RMSE')
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(
                df_results.style.format({
                    'MAE': '{:.2f}',
                    'RMSE': '{:.2f}',
                    'R2': '{:.4f}',
                    'MAPE': '{:.2f}'
                }).background_gradient(cmap='RdYlGn_r', subset=['RMSE', 'MAE'])
                .background_gradient(cmap='RdYlGn', subset=['R2']),
                use_container_width=True
            )
        
        with col2:
            best_name = trained_results['best_model_name']
            best_metrics = models_data[best_name]['metrics']
            
            st.metric("Cel Mai Bun", best_name)
            st.metric("R² Score", f"{best_metrics['R2']:.4f}")
            st.metric("RMSE", f"{best_metrics['RMSE']:.2f}")
            
            if not np.isnan(best_metrics['MAPE']):
                st.metric("MAPE", f"{best_metrics['MAPE']:.2f}%")
            
            if best_metrics['R2'] > 0.85:
                st.success("Calitate: EXCELENTĂ")
            elif best_metrics['R2'] > 0.70:
                st.info("Calitate: BUNĂ")
            else:
                st.warning("Calitate: ACCEPTABILĂ")
        


        # Grafic comparativ
        st.markdown("---")
        st.subheader("Comparație Vizuală")
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='R2 Score', x=df_results.index, y=df_results['R2'],
            text=df_results['R2'].round(4), textposition='auto',
            marker_color='lightblue'
        ))
        fig.add_trace(go.Bar(
            name='RMSE', x=df_results.index, y=df_results['RMSE'],
            text=df_results['RMSE'].round(2), textposition='auto',
            marker_color='lightcoral', yaxis='y2'
        ))
        
        fig.update_layout(
            title="Comparație Performanță: R2 vs RMSE",
            xaxis_title="Model", yaxis_title="R2 Score",
            yaxis2=dict(title="RMSE", overlaying='y', side='right'),
            barmode='group', height=500, hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature Importance
        st.markdown("---")
        st.subheader("Feature Importance")
        
        tabs = st.tabs([name for name in models_data.keys()])
        
        for tab, (name, data) in zip(tabs, models_data.items()):
            with tab:
                if data['feature_importance']:
                    if 'importances' in data['feature_importance']:
                        imp_df = pd.DataFrame({
                            'Feature': data['feature_importance']['features'],
                            'Importance': data['feature_importance']['importances']
                        }).sort_values('Importance', ascending=False).head(10)
                        
                        fig = px.bar(
                            imp_df, x='Importance', y='Feature', orientation='h',
                            title=f'Top 10 Features - {name}',
                            color='Importance', color_continuous_scale='Viridis'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif 'coefficients' in data['feature_importance']:
                        coef_df = pd.DataFrame({
                            'Feature': data['feature_importance']['features'],
                            'Coefficient': data['feature_importance']['coefficients']
                        }).sort_values('Coefficient', key=abs, ascending=False).head(10)
                        
                        fig = px.bar(
                            coef_df, x='Coefficient', y='Feature', orientation='h',
                            title=f'Top 10 Features - {name}',
                            color='Coefficient', color_continuous_scale='RdBu'
                        )
                        st.plotly_chart(fig, use_container_width=True)


# PAGINA PREDICȚII=============================================================================

def pagina_predictii():
    st.title("Predicții Viitoare")
    st.markdown("---")
    
    # Verifică model
    model_package = load_saved_model()
    trained_results = load_trained_results()
    
    if not model_package or not trained_results:
        st.warning("Niciun model antrenat! Accesează secțiunea 'Antrenament Modele' mai întâi.")
        return
    
    st.success(f"Model încărcat: **{model_package['model_name']}** (R² = {model_package['metrics']['R2']:.4f})")
    
    # Setări predicții
    col1, col2 = st.columns([3, 1])
    
    with col1:
        months_ahead = st.slider("Număr luni de prezis:", 1, 24, 6)
    
    with col2:
        with_confidence = st.checkbox(
            "Intervale încredere", 
            value=(model_package['model_name'] == 'Random Forest'),
            disabled=(model_package['model_name'] != 'Random Forest')
        )
    
    if st.button("Generează Predicții", type="primary"):
        _, _, date_ml, _ = load_all_data()
        
        with st.spinner("Generare predicții..."):
            model = model_package['model']
            feature_cols = model_package['feature_cols']
            
            # APEL FUNCȚII BACKEND
            if with_confidence and model_package['model_name'] == 'Random Forest':
                predictions = predict_with_confidence(model, date_ml, feature_cols, months_ahead)
            else:
                predictions = predict_future(model, date_ml, feature_cols, months_ahead)
        
        st.success("Predicții generate!")
        
        # Predicții pe scopuri
        st.markdown("---")
        st.subheader("Predicții pe Scopuri")
        
        for scop in predictions['Scop'].unique():
            pred_scop = predictions[predictions['Scop'] == scop]
            
            fig = go.Figure()
            
            if 'Lower_95' in pred_scop.columns:
                fig.add_trace(go.Scatter(
                    x=pred_scop['Data'], y=pred_scop['Upper_95'],
                    fill=None, mode='lines', line_color='lightblue',
                    name='Upper 95%', showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=pred_scop['Data'], y=pred_scop['Lower_95'],
                    fill='tonexty', mode='lines', line_color='lightblue',
                    name='Interval 95%'
                ))
            
            fig.add_trace(go.Scatter(
                x=pred_scop['Data'], y=pred_scop['Predictie'],
                mode='lines+markers', name='Predicție',
                line=dict(color='blue', width=3)
            ))
            
            fig.update_layout(
                title=f"Predicții: {scop}",
                xaxis_title="Data", yaxis_title="Număr Imigranți",
                hovermode='x unified', height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Tabel sumar
        st.markdown("---")
        st.subheader("Sumar Predicții")
        st.dataframe(predictions, use_container_width=True)
        


# MAIN APP============================================================================
def main():
    sectiune = sidebar_navigation()
    
    if sectiune == "Acasă":
        pagina_acasa()
    elif sectiune == "Vizualizări Date":
        pagina_vizualizari()
    elif sectiune == "Antrenament Modele":
        pagina_antrenament()
    elif sectiune == "Predicții":
        pagina_predictii()


if __name__ == "__main__":
    main()