import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

# Configuración de la página
st.set_page_config(page_title="Perfiles de Clientes Personalizados", layout="wide")

# Título de la aplicación
st.title("Identificación de Perfiles de Clientes Personalizados")

# Cargar archivo CSV
st.sidebar.header("Carga tus datos de clientes")
uploaded_file = st.sidebar.file_uploader("Sube un archivo CSV", type=["csv"])

if uploaded_file:
    # Leer archivo CSV
    data = pd.read_csv(uploaded_file)

    # Permitir al usuario elegir cuántas filas cargar (por defecto 100)
    num_rows = st.sidebar.number_input("Número de filas a cargar", min_value=1, max_value=len(data), value=100)
    data = data.head(num_rows)  # Cargar solo las primeras N filas

    st.subheader(f"Datos Cargados (mostrando las primeras {num_rows} filas)")
    st.write(data.head())

    # Selección de características para análisis
    st.sidebar.header("Selecciona las columnas para análisis")
    selected_features = st.sidebar.multiselect("Selecciona las columnas para el análisis de perfiles", data.columns.tolist())

    if selected_features:
        st.subheader(f"Análisis basado en las columnas seleccionadas: {', '.join(selected_features)}")

        # Mostrar visualizaciones interactivas usando Plotly
        for feature in selected_features:
            if data[feature].dtype == 'object' or len(data[feature].unique()) < 10:  # Si es categórica o tiene pocos valores únicos
                fig = px.bar(data, x=feature, title=f"Distribución de {feature}", 
                             color_discrete_sequence=px.colors.qualitative.Vivid)
                st.plotly_chart(fig)
            else:
                fig = px.histogram(data, x=feature, nbins=20, title=f"Distribución de {feature}",
                                   color_discrete_sequence=px.colors.sequential.Plasma)
                st.plotly_chart(fig)
        
        # Procesar los datos categóricos y numéricos
        categorical_features = [col for col in selected_features if data[col].dtype == 'object']
        numerical_features = [col for col in selected_features if data[col].dtype in ['int64', 'float64']]
        
        # Procesar los datos usando OneHotEncoding para categóricas y escalado para numéricas
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(), categorical_features)
            ]
        )
        
        # Aplicar K-means para agrupar en perfiles
        st.sidebar.header("Identificar Perfiles de Clientes")
        n_segments = st.sidebar.slider("Número de perfiles (segmentos)", min_value=2, max_value=10, value=3)
        
        if st.sidebar.button("Identificar Perfiles"):
            # Crear un pipeline que aplique el preprocesamiento y K-means
            kmeans = Pipeline(steps=[('preprocessor', preprocessor), ('kmeans', KMeans(n_clusters=n_segments, random_state=42))])
            kmeans.fit(data[selected_features])
            data['Perfil'] = kmeans.named_steps['kmeans'].labels_
            
            st.subheader("Resultados de la Segmentación en Perfiles")
            st.write("Los clientes han sido agrupados en los siguientes perfiles:")
            st.dataframe(data[['Perfil'] + selected_features].head())
            
            # Gráficos de Perfiles (Distribución por perfil)
            st.subheader("Distribución de Perfiles por Característica")
            for feature in selected_features:
                fig = px.bar(data, x='Perfil', y=feature, title=f"Distribución de {feature} por Perfil",
                             color='Perfil', barmode='group', 
                             color_discrete_sequence=px.colors.qualitative.Vivid)
                st.plotly_chart(fig)

            # Generar sugerencias de marketing por perfil
            st.subheader("Sugerencias de Marketing para Cada Perfil")
            perfiles = data['Perfil'].unique()
            for perfil in perfiles:
                perfil_data = data[data['Perfil'] == perfil]
                st.write(f"### Perfil {perfil}:")
                st.write(f"Este grupo tiene un total de {len(perfil_data)} clientes.")
                
                # Características más comunes en este perfil
                st.write(f"#### Características comunes en el Perfil {perfil}:")
                for feature in selected_features:
                    st.write(f"- {feature}: {perfil_data[feature].mode()[0]}")
                
                # Sugerencia de marketing
                st.write(f"#### Sugerencias de Marketing para el Perfil {perfil}:")
                st.write("- Crear campañas dirigidas a las características más comunes en este grupo.")
                st.write("- Ofrecer productos o servicios alineados con las preferencias de este perfil.")

