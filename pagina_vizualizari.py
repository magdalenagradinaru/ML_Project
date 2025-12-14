"""       
def pagina_vizualizari():
    st.title("Vizualizari Date")
    st.markdown("---")
    
    # CORECTARE: load_all_data() în loc de load_data()
    date_lunare, date_agregat, _, _ = load_all_data()
    
    if date_lunare is None:
        return
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Analiza pe Tari",
        "Analiza pe Scopuri",
        "Evolutie Temporala",
        "Heatmap Sezonalitate"
    ])
    
    # TAB 1: Analiza pe Tari
    with tab1:
        st.subheader("Analiza Distributie pe Tari")
        
        top_n = st.slider("Numarul de tari afisate:", 5, 20, 10)
        top_tari = date_lunare.groupby('Tara')['Numar'].sum().sort_values(ascending=False).head(top_n)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                values=top_tari.values,
                names=top_tari.index,
                title=f"Top {top_n} Tari (Distributie %)",
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                x=top_tari.values,
                y=top_tari.index,
                orientation='h',
                title=f"Top {top_n} Tari (Numar Absolut)",
                labels={'x': 'Numar Total', 'y': 'Tara'},
                color=top_tari.values,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Evolutie Tari in Timp")
        
        tara_selectata = st.selectbox("Selecteaza tara:", sorted(date_lunare['Tara'].unique()))
        data_tara = date_lunare[date_lunare['Tara'] == tara_selectata].groupby('Data')['Numar'].sum().reset_index()
        
        fig = px.line(
            data_tara, x='Data', y='Numar',
            title=f"Evolutie Imigranti din {tara_selectata}",
            markers=True
        )
        fig.update_layout(hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: Analiza pe Scopuri
    with tab2:
        st.subheader("Analiza pe Scopuri Imigrare")
        
        col1, col2 = st.columns(2)
        
        with col1:
            scopuri_totale = date_agregat.groupby('Scop')['Numar'].sum().sort_values(ascending=False)
            fig = px.bar(
                x=scopuri_totale.values, y=scopuri_totale.index, orientation='h',
                title="Total Imigranti pe Scop",
                labels={'x': 'Numar Total', 'y': 'Scop'},
                color=scopuri_totale.values, color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            media_scopuri = date_agregat.groupby('Scop')['Numar'].mean().sort_values(ascending=False)
            fig = px.bar(
                x=media_scopuri.values, y=media_scopuri.index, orientation='h',
                title="Media Lunara pe Scop",
                labels={'x': 'Media Lunara', 'y': 'Scop'},
                color=media_scopuri.values, color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Evolutie Scopuri in Timp")
        
        scopuri_selectate = st.multiselect(
            "Selecteaza scopuri pentru comparatie:",
            sorted(date_agregat['Scop'].unique()),
            default=sorted(date_agregat['Scop'].unique())[:3]
        )
        
        if scopuri_selectate:
            data_scopuri = date_agregat[date_agregat['Scop'].isin(scopuri_selectate)]
            fig = px.line(
                data_scopuri, x='Data', y='Numar', color='Scop',
                title="Evolutie Comparativa Scopuri", markers=True
            )
            fig.update_layout(hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 3: Evolutie Temporala
    with tab3:
        st.subheader("Analiza Evolutie Temporala")
        
        evolutie_anuala = date_agregat.groupby('An')['Numar'].sum().reset_index()
        fig = px.bar(
            evolutie_anuala, x='An', y='Numar',
            title="Evolutie Anuala Totala",
            labels={'Numar': 'Total Imigranti', 'An': 'An'},
            color='Numar', color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            evolutie_lunara = date_agregat.groupby('Luna')['Numar'].mean().reset_index()
            evolutie_lunara['Luna_Nume'] = evolutie_lunara['Luna'].map({
                1: 'Ian', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'Mai', 6: 'Iun',
                7: 'Iul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Noi', 12: 'Dec'
            })
            fig = px.bar(
                evolutie_lunara, x='Luna_Nume', y='Numar',
                title="Sezonalitate (Media pe Luna)",
                labels={'Numar': 'Media', 'Luna_Nume': 'Luna'},
                color='Numar', color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            evolutie_trimestru = date_agregat.copy()
            evolutie_trimestru['Trimestru'] = ((evolutie_trimestru['Luna'] - 1) // 3 + 1)
            evolutie_trimestru = evolutie_trimestru.groupby('Trimestru')['Numar'].mean().reset_index()
            fig = px.bar(
                evolutie_trimestru, x='Trimestru', y='Numar',
                title="Media pe Trimestru",
                labels={'Numar': 'Media', 'Trimestru': 'Trimestru'},
                color='Numar', color_continuous_scale='Plasma'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 4: Heatmap
    with tab4:
        st.subheader("Heatmap Sezonalitate")
        
        heatmap_data = date_agregat.pivot_table(
            values='Numar', index='Scop', columns='Luna', aggfunc='mean'
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=['Ian', 'Feb', 'Mar', 'Apr', 'Mai', 'Iun', 'Iul', 'Aug', 'Sep', 'Oct', 'Noi', 'Dec'],
            y=heatmap_data.index,
            colorscale='RdYlGn',
            text=heatmap_data.values.round(0),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Media")
        ))
        
        fig.update_layout(
            title="Heatmap Sezonalitate (Scop x Luna)",
            xaxis_title="Luna", yaxis_title="Scop", height=600
        )
        st.plotly_chart(fig, use_container_width=True)

"""