import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

# Configuración de la página
st.set_page_config(page_title="Recomendaciones de Marketing Personalizadas", layout="wide")

# Título de la aplicación
st.title("Recomendaciones de Marketing Personalizadas con API de Gemini")

# Ingreso de la API Key de Gemini
st.sidebar.header("Configuración de la API de Gemini")
gemini_api_key = st.sidebar.text_input("Introduce tu API Key de Gemini", type="password")

# Verificar si la API Key está ingresada
if gemini_api_key:
    st.sidebar.success("API Key ingresada correctamente.")

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

            # Procesar datos categóricos y numéricos
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

                # Generar recomendaciones de marketing usando la API de Gemini
                st.subheader("Generar Recomendaciones de Marketing")
                
                # Simulación de recomendación de marketing con la API de Gemini
                for perfil in data['Perfil'].unique():
                    perfil_data = data[data['Perfil'] == perfil]
                    common_features = {feature: perfil_data[feature].mode()[0] for feature in selected_features}
                    
                    st.write(f"### Perfil {perfil}")
                    st.write("Características comunes:", common_features)

                    # Simular llamada a la API de Gemini
                    try:
                        # Cambiar 'https://api.gemini.com/recommend' a la URL de tu API real
                        response = requests.post(
                            "https://api.gemini.com/recommend",
                            headers={"Authorization": f"Bearer {gemini_api_key}"},
                            json={"segment": perfil, "features": common_features}
                        )
                        if response.status_code == 200:
                            recommendation = response.json().get("recommendation")
                            st.write(f"**Recomendación de Marketing para el Perfil {perfil}:** {recommendation}")
                        else:
                            st.warning("No se pudo obtener una recomendación de la API de Gemini.")
                    except Exception as e:
                        st.error(f"Error en la solicitud a la API de Gemini: {e}")
else:
    st.warning("Por favor, ingresa tu API Key de Gemini para comenzar.")

