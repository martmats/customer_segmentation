import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Configuración de la página
st.set_page_config(page_title="Segmentación de Clientes Personalizada", layout="wide")

# Título de la aplicación
st.title("Segmentación de Clientes Personalizada")

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
    selected_features = st.sidebar.multiselect("Selecciona las columnas para el análisis de segmentación", data.columns.tolist())

    if selected_features:
        st.subheader(f"Análisis basado en las columnas seleccionadas: {', '.join(selected_features)}")

        # Mostrar visualizaciones interactivas usando Plotly
        for feature in selected_features:
            if data[feature].dtype == 'object' or len(data[feature].unique()) < 10:  # Si es categórica o tiene pocos valores únicos
                fig = px.bar(data, x=feature, title=f"Distribución de {feature}")
                st.plotly_chart(fig)
            else:
                fig = px.histogram(data, x=feature, nbins=20, title=f"Distribución de {feature}")
                st.plotly_chart(fig)
        
        # Filtrar las columnas categóricas y numéricas
        categorical_features = [col for col in selected_features if data[col].dtype == 'object']
        numerical_features = [col for col in selected_features if data[col].dtype in ['int64', 'float64']]
        
        # Procesar los datos usando OneHotEncoding para categóricas y escalado para numéricas
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(), categorical_features)
            ]
        )
        
        # Aplicar K-means para la segmentación
        st.sidebar.header("Segmentación")
        n_clusters = st.sidebar.slider("Selecciona el número de clusters", min_value=2, max_value=10, value=3)
        
        if st.sidebar.button("Aplicar Segmentación"):
            # Crear un pipeline que aplique el preprocesamiento y K-means
            kmeans = Pipeline(steps=[('preprocessor', preprocessor), ('kmeans', KMeans(n_clusters=n_clusters, random_state=42))])
            kmeans.fit(data[selected_features])
            data['Cluster'] = kmeans.named_steps['kmeans'].labels_
            
            st.subheader("Resultados de la Segmentación")
            st.write("Los clientes han sido agrupados en los siguientes clusters:")
            st.dataframe(data[['Cluster'] + selected_features].head())
            
            # Gráfico de clusters (si hay al menos dos características numéricas)
            if len(numerical_features) >= 2:
                fig = px.scatter(data, x=numerical_features[0], y=numerical_features[1], color='Cluster', title="Clusters de Clientes")
                st.plotly_chart(fig)
            
            # Visualización de clusters usando Pairplot
            st.subheader("Visualización detallada de los Clusters")
            if len(selected_features) > 1:
                sns.pairplot(data, hue="Cluster", vars=selected_features[:3], palette="viridis")
                st.pyplot(plt)
            
            st.write("""
            ### Interpretación de los Clusters:
            - Utiliza los grupos creados para identificar características comunes en los clientes.
            - Ejemplo: Si descubres que un cluster está dominado por mujeres que poseen un coche en una región específica, puedes crear campañas de marketing dirigidas a esas características.
            """)

