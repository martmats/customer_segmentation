import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la aplicación
st.title("Segmentación Avanzada de Clientes")
st.write("Sube un archivo CSV con datos de clientes (historial de compras, edad, ubicación, intereses, etc.)")

# Cargar el archivo
uploaded_file = st.file_uploader("Carga un archivo CSV", type=["csv"])

if uploaded_file:
    # Leer el archivo
    data = pd.read_csv(uploaded_file)
    st.write("Datos cargados:")
    st.dataframe(data.head())

    # Selección de columnas para el clustering
    features = st.multiselect("Selecciona las columnas para el análisis de clustering", options=data.columns.tolist())
    
    if features:
        # Seleccionar solo las columnas numéricas
        data_selected = data[features].select_dtypes(include=['number']).dropna()  # Selecciona columnas numéricas y elimina filas con valores nulos
        data_selected = data_selected.drop_duplicates()  # Elimina duplicados
        
        # Verificar si hay columnas numéricas seleccionadas
        if data_selected.empty:
            st.error("Selecciona al menos una columna numérica para realizar el clustering.")
        else:
            st.write("Datos seleccionados para el clustering:")
            st.dataframe(data_selected.head())
            
            # Escalado de datos
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_selected)

            # Selección del número de clusters
            st.write("Selecciona el número de clusters")
            num_clusters = st.slider("Número de clusters (K)", min_value=2, max_value=10, value=3)

            # Aplicación del modelo de clustering K-means
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            clusters = kmeans.fit_predict(data_scaled)
            data_selected["Cluster"] = clusters

            # Visualización de los resultados
            st.write("Segmentos de clientes:")
            st.dataframe(data_selected)

            # Gráfica de clusters
            st.write("Visualización de Clusters (usando las dos primeras características)")
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=data_scaled[:, 0], y=data_scaled[:, 1], hue=clusters, palette="viridis")
            plt.xlabel(features[0])
            plt.ylabel(features[1])
            plt.title("Visualización de Clusters")
            st.pyplot(plt)

            # Interpretación de resultados
            st.write("Interpretación de Resultados")
            for cluster in range(num_clusters):
                st.write(f"Características del Cluster {cluster}")
                st.write(data_selected[data_selected["Cluster"] == cluster].describe())
