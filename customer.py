import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Configuración de la página
st.set_page_config(page_title="Segmentación Predictiva de Clientes", layout="wide")

# Título de la aplicación
st.title("Segmentación Predictiva de Clientes: De Storytelling a Predicción")

# Cargar archivo CSV
st.sidebar.header("Carga tus datos de clientes")
uploaded_file = st.sidebar.file_uploader("Sube un archivo CSV", type=["csv"])

if uploaded_file:
    # Leer archivo CSV
    data = pd.read_csv(uploaded_file)
    st.subheader("Datos Cargados")
    st.write(data.head())

    # Storytelling: Segmentación de Clientes
    st.header("Storytelling: Segmentación de Clientes")
    st.write("Visualización interactiva para entender mejor los grupos de clientes")

    # Selección de columnas
    features = st.multiselect("Selecciona las columnas para la segmentación", data.columns.tolist())
    
    if len(features) > 1:
        # Limpiar y preparar los datos
        data_selected = data[features].dropna()  # Filtrar datos
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_selected)
        
        # Selección del número de clusters
        num_clusters = st.slider("Selecciona el número de clusters (K)", min_value=2, max_value=10, value=3)

        # Aplicar K-means
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(data_scaled)
        data_selected["Cluster"] = clusters

        # Visualización de los clusters
        st.subheader("Visualización de los Segmentos de Clientes")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=data_scaled[:, 0], y=data_scaled[:, 1], hue=clusters, palette="viridis")
        plt.title("Clusters de Clientes")
        st.pyplot(plt)

        # Características de los clusters
        st.subheader("Características de los Clusters")
        st.write(data_selected.groupby("Cluster").mean())

        # Storytelling sobre los clusters
        st.write("""
        ### Insights clave:
        - El análisis de clustering nos permite visualizar cómo se agrupan los clientes en función de sus comportamientos y características.
        - Con esto, puedes identificar segmentos específicos como "clientes de alto valor" o "clientes con bajo nivel de compromiso".
        - Esta información es útil para estrategias de marketing personalizadas y la mejora de la relación con los clientes.
        """)

        # Predicción del Comportamiento Futuro
        st.header("Predicción: ¿Cómo será el comportamiento futuro de los clientes?")
        st.write("""
        Ahora que hemos segmentado a los clientes, podemos entrenar un modelo predictivo que nos ayude a anticipar comportamientos como:
        - ¿Realizará este cliente una gran compra?
        - ¿Es probable que este cliente abandone?
        """)

        # Selección de la columna objetivo (variable dependiente para la predicción)
        target_column = st.selectbox("Selecciona la columna objetivo para la predicción", data.columns)

        # Entrenar el modelo predictivo
        if target_column:
            X = data_selected.drop(columns=["Cluster", target_column])
            y = data[target_column].dropna()

            # Dividir datos en entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Modelo de predicción (Random Forest)
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Mostrar precisión del modelo
            accuracy = accuracy_score(y_test, y_pred)
            st.subheader(f"Precisión del modelo: {accuracy:.2%}")

            # Visualización de la importancia de las características
            feature_importance = pd.DataFrame(model.feature_importances_, index=X.columns, columns=["Importancia"])
            feature_importance = feature_importance.sort_values(by="Importancia", ascending=False)
            st.subheader("Importancia de las características en la predicción")
            st.bar_chart(feature_importance)

            st.write("Con estos insights, puedes anticipar mejor las decisiones de tus clientes y ajustar tus estrategias de negocio.")
