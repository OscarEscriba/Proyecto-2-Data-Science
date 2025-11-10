import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import joblib
import pickle
from pathlib import Path

st.set_page_config(
    page_title="Predicción de Compradores Recurrentes",
    page_icon="cart",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
    }
    
    .stButton>button {
        background-color: #4a90e2;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #357abd;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(74, 144, 226, 0.4);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .info-box {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    h1 {
        color: #2c3e50;
        font-weight: 700;
    }
    
    h2, h3 {
        color: #4a90e2;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).resolve().parent

try:
    train_df = pd.read_parquet(BASE_DIR / "train_df.parquet")
    test_df = pd.read_parquet(BASE_DIR / "test_df.parquet")
    resultados = pd.read_csv(BASE_DIR / "resultados.csv")
    le_user = joblib.load(BASE_DIR / "le_user.pkl")
    le_merchant = joblib.load(BASE_DIR / "le_merchant.pkl")

    rf_model = joblib.load(BASE_DIR / "rf_model.pkl")
    lr_model = joblib.load(BASE_DIR / "lr_model.pkl")
    xgb_model = joblib.load(BASE_DIR / "xgb_model.pkl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from modelos import SimpleGNN, EmbeddingModel, DeepFM

    n_users = len(le_user.classes_)
    n_merchants = len(le_merchant.classes_)

    gnn_model = SimpleGNN(n_users, n_merchants, embedding_dim=64).to(device)
    gnn_model.load_state_dict(torch.load(BASE_DIR / "gnn_model.pth", map_location=device))
    gnn_model.eval()

    emb_model = EmbeddingModel(n_users, n_merchants, embedding_dim=50).to(device)
    emb_model.load_state_dict(torch.load(BASE_DIR / "emb_model.pth", map_location=device))
    emb_model.eval()

    deepfm_model = DeepFM(n_users=n_users, n_merchants=n_merchants, embedding_dim=32).to(device)
    deepfm_model.load_state_dict(torch.load(BASE_DIR / "deepfm_model.pth", map_location=device))
    deepfm_model.eval()

    with open(BASE_DIR / "validacion.pkl", "rb") as f:
        valid_data = pickle.load(f)

    y_val = valid_data["y_val"]
    gnn_predictions_proba = valid_data["gnn_predictions_proba"]
    emb_predictions_proba = valid_data["emb_predictions_proba"]
    deepfm_predictions_proba = valid_data["deepfm_predictions_proba"]
    rf_predictions_proba = valid_data["rf_predictions_proba"]
    lr_predictions_proba = valid_data["lr_predictions_proba"]
    xgb_predictions_proba = valid_data["xgb_predictions_proba"]

    st.sidebar.success("Datos cargados correctamente")

except Exception as e:
    st.error(f"Error al cargar datos o modelos: {e}")
    st.stop()

st.markdown(
    """
    <h1 style='text-align: center; color: #2c3e50;'>
        Sistema de Predicción de Compradores Recurrentes
    </h1>
    <p style='text-align: center; color: #7f8c8d; font-size: 1.2rem;'>
        Plataforma de Machine Learning con 6 Modelos de Predicción
    </p>
    <hr style='margin: 2rem 0;'>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("Navegación")
page = st.sidebar.radio(
    "Selecciona una sección:",
    ["Inicio", "Exploración de Datos", "Realizar Predicción", "Rendimiento de Modelos", "Información"],
)

if page == "Inicio":
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3>Usuarios</h3>
                <h2>{train_df['user_id'].nunique():,}</h2>
                <p>Usuarios únicos</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3>Merchants</h3>
                <h2>{train_df['merchant_id'].nunique():,}</h2>
                <p>Comercios registrados</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3>Tasa de Compra</h3>
                <h2>{train_df['label'].mean() * 100:.2f}%</h2>
                <p>Conversión promedio</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="info-box">
            <h3>Objetivo del Proyecto</h3>
            <p style='margin: 0;'>
                Este sistema utiliza 6 algoritmos diferentes de Machine Learning para predecir 
                si un usuario realizará una compra en un merchant específico.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Modelos Implementados")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        Modelos de Deep Learning:
        - GNN (Graph Neural Network)
        - Neural Embeddings
        - DeepFM (Deep Factorization Machine)
        """
        )

    with col2:
        st.markdown(
            """
        Modelos Tradicionales:
        - Random Forest
        - Logistic Regression
        - XGBoost
        """
        )

elif page == "Exploración de Datos":
    st.header("Exploración Interactiva de Datos")

    tab1, tab2, tab3, tab4 = st.tabs(["Distribuciones", "Relaciones", "Por Segmento", "Datos Crudos"])

    with tab1:
        st.subheader("Distribución de Variables")

        col1, col2 = st.columns(2)

        with col1:
            label_counts = train_df["label"].value_counts()
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=["No Compra", "Compra"],
                        y=label_counts.values,
                        marker=dict(color=["#e74c3c", "#2ecc71"]),
                        text=label_counts.values,
                        textposition="auto",
                    )
                ]
            )
            fig.update_layout(
                title="Distribución: Compradores vs No Compradores",
                xaxis_title="Clase",
                yaxis_title="Frecuencia",
                template="plotly_white",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if "age_range" in train_df.columns:
                age_counts = train_df["age_range"].value_counts().sort_index()
                fig = go.Figure(
                    data=[
                        go.Bar(
                            x=age_counts.index,
                            y=age_counts.values,
                            marker=dict(color="#3498db"),
                            text=age_counts.values,
                            textposition="auto",
                        )
                    ]
                )
                fig.update_layout(
                    title="Distribución por Rango de Edad",
                    xaxis_title="Rango de Edad",
                    yaxis_title="Frecuencia",
                    template="plotly_white",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            if "gender" in train_df.columns:
                gender_counts = train_df["gender"].value_counts()
                gender_labels_map = {0: "Femenino", 1: "Masculino", 2: "Desconocido"}
                fig = go.Figure(
                    data=[
                        go.Pie(
                            labels=[gender_labels_map.get(x, str(x)) for x in gender_counts.index],
                            values=gender_counts.values,
                            hole=0.4,
                            marker=dict(colors=["#e74c3c", "#3498db", "#95a5a6"]),
                        )
                    ]
                )
                fig.update_layout(
                    title="Distribución por Género",
                    template="plotly_white",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

        with col4:
            top_merchants = train_df["merchant_id"].value_counts().head(10)
            fig = go.Figure(
                data=[
                    go.Bar(
                        y=[f"Merchant {x}" for x in top_merchants.index],
                        x=top_merchants.values,
                        orientation="h",
                        marker=dict(color="#e67e22"),
                        text=top_merchants.values,
                        textposition="auto",
                    )
                ]
            )
            fig.update_layout(
                title="Top 10 Merchants Más Frecuentes",
                xaxis_title="Frecuencia",
                yaxis_title="Merchant",
                template="plotly_white",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Relaciones entre Variables")

        col1, col2 = st.columns(2)

        with col1:
            if "age_range" in train_df.columns:
                conversion_by_age = train_df.groupby("age_range")["label"].mean()
                fig = go.Figure(
                    data=[
                        go.Scatter(
                            x=conversion_by_age.index,
                            y=conversion_by_age.values * 100,
                            mode="lines+markers",
                            marker=dict(size=10, color="#9b59b6"),
                            line=dict(width=3),
                        )
                    ]
                )
                fig.update_layout(
                    title="Tasa de Conversión por Rango de Edad",
                    xaxis_title="Rango de Edad",
                    yaxis_title="Tasa de Conversión (%)",
                    template="plotly_white",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if "gender" in train_df.columns:
                conversion_by_gender = train_df.groupby("gender")["label"].mean()
                gender_labels_map = {0: "Femenino", 1: "Masculino", 2: "Desconocido"}
                fig = go.Figure(
                    data=[
                        go.Bar(
                            x=[gender_labels_map.get(x, str(x)) for x in conversion_by_gender.index],
                            y=conversion_by_gender.values * 100,
                            marker=dict(color=["#e74c3c", "#3498db", "#95a5a6"]),
                            text=[f"{v:.2f}%" for v in conversion_by_gender.values * 100],
                            textposition="auto",
                        )
                    ]
                )
                fig.update_layout(
                    title="Tasa de Conversión por Género",
                    xaxis_title="Género",
                    yaxis_title="Tasa de Conversión (%)",
                    template="plotly_white",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Análisis por Segmento")

        col1, col2 = st.columns(2)

        age_filter = None
        gender_filter = None

        with col1:
            if "age_range" in train_df.columns:
                age_filter = st.multiselect(
                    "Filtrar por Rango de Edad",
                    options=sorted(train_df["age_range"].unique()),
                    default=sorted(train_df["age_range"].unique()),
                )

        with col2:
            if "gender" in train_df.columns:
                gender_labels_map = {0: "Femenino", 1: "Masculino", 2: "Desconocido"}
                gender_filter = st.multiselect(
                    "Filtrar por Género",
                    options=sorted(train_df["gender"].unique()),
                    default=sorted(train_df["gender"].unique()),
                    format_func=lambda x: gender_labels_map.get(x, str(x)),
                )

        filtered_df = train_df.copy()
        if age_filter:
            filtered_df = filtered_df[filtered_df["age_range"].isin(age_filter)]
        if gender_filter:
            filtered_df = filtered_df[filtered_df["gender"].isin(gender_filter)]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Registros", f"{len(filtered_df):,}")
        with col2:
            st.metric("Usuarios Únicos", f"{filtered_df['user_id'].nunique():,}")
        with col3:
            st.metric("Tasa de Compra", f"{filtered_df['label'].mean()*100:.2f}%")
        with col4:
            st.metric("Merchants Únicos", f"{filtered_df['merchant_id'].nunique():,}")

    with tab4:
        st.subheader("Vista de Datos Crudos")
        st.dataframe(train_df.head(100), use_container_width=True, height=400)

        st.markdown("Estadísticas Descriptivas:")
        st.dataframe(train_df.describe(), use_container_width=True)

elif page == "Realizar Predicción":
    st.header("Realizar Predicción de Compra")

    st.markdown(
        """
        <div class="info-box">
            <p style='margin: 0;'>
                Ingresa los datos del usuario y merchant para obtener predicciones de 
                todos los modelos entrenados.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            user_options = sorted(train_df["user_id"].unique())[:1000]
            user_id = st.selectbox("User ID", options=user_options)

            if "age_range" in train_df.columns:
                age_range_val = st.selectbox(
                    "Rango de Edad",
                    options=sorted(train_df["age_range"].unique()),
                )
            else:
                age_range_val = 0

        with col2:
            merchant_options = sorted(train_df["merchant_id"].unique())[:1000]
            merchant_id = st.selectbox("Merchant ID", options=merchant_options)

            if "gender" in train_df.columns:
                gender_labels_map = {0: "Femenino", 1: "Masculino", 2: "Desconocido"}
                gender_val = st.selectbox(
                    "Género",
                    options=sorted(train_df["gender"].unique()),
                    format_func=lambda x: gender_labels_map.get(x, str(x)),
                )
            else:
                gender_val = 2

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            modelo_seleccionado = st.selectbox(
                "Modelo de predicción",
                ["Ensemble", "Random Forest", "Logistic Regression", "XGBoost", "GNN", "Neural Embeddings", "DeepFM"],
            )
        with col3:
            submit_button = st.form_submit_button("Predecir", use_container_width=True)

    if submit_button:
        with st.spinner("Realizando predicciones..."):
            def safe_encode(encoder, value):
                if value in encoder.classes_:
                    return encoder.transform([value])[0]
                return 0

            user_encoded = safe_encode(le_user, user_id)
            merchant_encoded = safe_encode(le_merchant, merchant_id)

            X_input = np.array([[user_encoded, merchant_encoded, age_range_val, gender_val]])

            predictions = {}
            device_pred = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            for name, model in [("Random Forest", rf_model), ("Logistic Regression", lr_model), ("XGBoost", xgb_model)]:
                pred_proba = model.predict_proba(X_input)[0, 1]
                predictions[name] = {
                    "probability": float(pred_proba),
                    "class": int(pred_proba >= 0.5),
                    "confidence": float(max(pred_proba, 1 - pred_proba)),
                }

            with torch.no_grad():
                user_tensor = torch.LongTensor([user_encoded]).to(device_pred)
                merchant_tensor = torch.LongTensor([merchant_encoded]).to(device_pred)
                age_tensor = torch.FloatTensor([age_range_val]).to(device_pred)
                gender_tensor = torch.FloatTensor([gender_val]).to(device_pred)

                for name, model in [("GNN", gnn_model), ("Neural Embeddings", emb_model), ("DeepFM", deepfm_model)]:
                    model.eval()
                    outputs = model(user_tensor, merchant_tensor, age_tensor, gender_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    pred_proba = probs[0, 1].cpu().numpy()
                    predictions[name] = {
                        "probability": float(pred_proba),
                        "class": int(pred_proba >= 0.5),
                        "confidence": float(max(pred_proba, 1 - pred_proba)),
                    }

            avg_prob = np.mean([p["probability"] for p in predictions.values()])
            predictions["Ensemble"] = {
                "probability": float(avg_prob),
                "class": int(avg_prob >= 0.5),
                "confidence": float(max(avg_prob, 1 - avg_prob)),
            }

        st.success("Predicciones completadas")

        modelo_principal = modelo_seleccionado if modelo_seleccionado in predictions else "Ensemble"
        main_result = predictions[modelo_principal]

        result_text = "COMPRARÁ" if main_result["class"] == 1 else "NO COMPRARÁ"
        result_color = "#2ecc71" if main_result["class"] == 1 else "#e74c3c"

        st.markdown(
            f"""
            <div style='background: linear-gradient(135deg, {result_color} 0%, {result_color}dd 100%); 
                 padding: 2rem; border-radius: 12px; color: white; text-align: center; margin: 1rem 0;'>
                <h2 style='margin: 0; color: white;'>{result_text}</h2>
                <p style='font-size: 1.5rem; margin: 0.5rem 0;'>
                    Probabilidad: {main_result['probability']*100:.2f}%
                </p>
                <p style='margin: 0; opacity: 0.9;'>
                    Confianza: {main_result['confidence']*100:.2f}%
                </p>
                <p style='margin-top: 0.5rem; opacity: 0.9;'>
                    Modelo seleccionado: {modelo_principal}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Resultados por Modelo")

        model_options = [k for k in predictions.keys() if k != "Ensemble"]
        selected_models = st.multiselect(
            "Selecciona modelos:",
            options=["Todos"] + model_options,
            default=["Todos"],
        )

        models_to_show = model_options if "Todos" in selected_models else selected_models

        if models_to_show:
            results_data = []
            for model_name in models_to_show:
                pred = predictions[model_name]
                results_data.append(
                    {
                        "Modelo": model_name,
                        "Predicción": "Comprará" if pred["class"] == 1 else "No Comprará",
                        "Probabilidad": pred["probability"],
                        "Confianza": pred["confidence"],
                    }
                )

            results_comparison = pd.DataFrame(results_data)
            st.dataframe(
                results_comparison.style.format(
                    {
                        "Probabilidad": "{:.2%}",
                        "Confianza": "{:.2%}",
                    }
                ).background_gradient(subset=["Probabilidad"], cmap="RdYlGn"),
                use_container_width=True,
            )

            col1, col2 = st.columns(2)

            with col1:
                colors = ["#2ecc71" if predictions[m]["class"] == 1 else "#e74c3c" for m in models_to_show]
                fig = go.Figure(
                    data=[
                        go.Bar(
                            x=models_to_show,
                            y=[predictions[m]["probability"] * 100 for m in models_to_show],
                            marker=dict(color=colors),
                            text=[f"{predictions[m]['probability']*100:.1f}%" for m in models_to_show],
                            textposition="auto",
                        )
                    ]
                )
                fig.add_hline(y=50, line_dash="dash", line_color="gray")
                fig.update_layout(
                    title="Probabilidad de Compra por Modelo",
                    yaxis_title="Probabilidad (%)",
                    template="plotly_white",
                    height=400,
                    yaxis=dict(range=[0, 100]),
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = go.Figure(
                    data=[
                        go.Bar(
                            x=models_to_show,
                            y=[predictions[m]["confidence"] * 100 for m in models_to_show],
                            marker=dict(color="#3498db"),
                            text=[f"{predictions[m]['confidence']*100:.1f}%" for m in models_to_show],
                            textposition="auto",
                        )
                    ]
                )
                fig.update_layout(
                    title="Nivel de Confianza por Modelo",
                    yaxis_title="Confianza (%)",
                    template="plotly_white",
                    height=400,
                    yaxis=dict(range=[0, 100]),
                )
                st.plotly_chart(fig, use_container_width=True)

elif page == "Rendimiento de Modelos":
    st.header("Análisis de Rendimiento de Modelos")

    show_metrics = st.checkbox("Mostrar métricas de rendimiento", value=True)

    if show_metrics:
        st.markdown(
            """
            <div class="success-box">
                <h4>Métricas de Validación</h4>
                <p style='margin: 0;'>
                    Calculadas con el conjunto de validación.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Tabla Comparativa")

        results_sorted = resultados.sort_values("ROC-AUC", ascending=False)

        styled_results = results_sorted.style.format(
            {
                "ROC-AUC": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1-Score": "{:.4f}",
            }
        ).background_gradient(subset=["ROC-AUC", "Precision", "Recall", "F1-Score"], cmap="RdYlGn")

        st.dataframe(styled_results, use_container_width=True, height=300)

        st.markdown("### Mejores Modelos")

        col1, col2, col3, col4 = st.columns(4)

        best_auc = resultados.loc[resultados["ROC-AUC"].idxmax()]
        best_precision = resultados.loc[resultados["Precision"].idxmax()]
        best_recall = resultados.loc[resultados["Recall"].idxmax()]
        best_f1 = resultados.loc[resultados["F1-Score"].idxmax()]

        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h4>Mejor ROC-AUC</h4>
                    <h3>{best_auc['Modelo']}</h3>
                    <p style='font-size: 1.5rem; margin: 0;'>{best_auc['ROC-AUC']:.4f}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h4>Mejor Precision</h4>
                    <h3>{best_precision['Modelo']}</h3>
                    <p style='font-size: 1.5rem; margin: 0;'>{best_precision['Precision']:.4f}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h4>Mejor Recall</h4>
                    <h3>{best_recall['Modelo']}</h3>
                    <p style='font-size: 1.5rem; margin: 0;'>{best_recall['Recall']:.4f}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h4>Mejor F1-Score</h4>
                    <h3>{best_f1['Modelo']}</h3>
                    <p style='font-size: 1.5rem; margin: 0;'>{best_f1['F1-Score']:.4f}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("### Visualización de Métricas")

        tab1, tab2 = st.tabs(["Comparación General", "Radar Chart"])

        with tab1:
            metrics = ["ROC-AUC", "Precision", "Recall", "F1-Score"]
            fig = go.Figure()

            for metric in metrics:
                fig.add_trace(
                    go.Bar(
                        name=metric,
                        x=resultados["Modelo"],
                        y=resultados[metric],
                        text=resultados[metric].apply(lambda x: f"{x:.3f}"),
                        textposition="auto",
                    )
                )

            fig.update_layout(
                title="Comparación de Todas las Métricas",
                xaxis_title="Modelo",
                yaxis_title="Score",
                barmode="group",
                template="plotly_white",
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            fig = go.Figure()

            for _, row in resultados.iterrows():
                fig.add_trace(
                    go.Scatterpolar(
                        r=[row["ROC-AUC"], row["Precision"], row["Recall"], row["F1-Score"]],
                        theta=["ROC-AUC", "Precision", "Recall", "F1-Score"],
                        fill="toself",
                        name=row["Modelo"],
                    )
                )

            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title="Comparación Multidimensional de Modelos",
                template="plotly_white",
                height=600,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.info("Un modelo más cercano al borde exterior indica mejor rendimiento.")

    else:
        st.info("Las métricas están ocultas. Marca la casilla arriba para visualizarlas.")

elif page == "Información":
    st.header("Información del Proyecto")

    st.markdown(
        """
    ### Objetivo
    
    Desarrollar un sistema de Machine Learning capaz de predecir si un usuario realizará una 
    compra en un merchant específico, basándose en su historial de comportamiento y características demográficas.
    
    ### Dataset
    
    El proyecto utiliza datos de un sistema de recomendación de comercio electrónico:
    
    - Train Set: Datos históricos de interacciones usuario-merchant
    - Test Set: Nuevas interacciones para predicciones
    - User Info: Información demográfica (edad, género)
    - User Log: Historial detallado de actividades
    
    ### Modelos Implementados
    
    Modelos de Deep Learning
    
    1. GNN (Graph Neural Network)
       - Arquitectura especializada en relaciones de grafo
       - Embeddings de dimensión 64
       - Tres capas de convolución
       - Dropout 0.3
    
    2. Neural Embeddings
       - Red neuronal profunda con embeddings
       - Embeddings de dimensión 50
       - Tres capas fully-connected
       - Batch normalization y dropout
    
    3. DeepFM (Deep Factorization Machine)
       - Combina Factorization Machine con Deep Learning
       - Captura interacciones de bajo y alto orden
       - Dimensión de embedding 32
    
    Modelos Tradicionales
    
    4. Random Forest
       - Doscientos árboles de decisión
       - Profundidad máxima 20
       - Balanceo de clases automático
    
    5. Logistic Regression
       - Modelo lineal simple y eficiente
       - Regularización L2
       - Balanceo de clases
    
    6. XGBoost
       - Trescientos árboles con gradient boosting
       - Learning rate 0.05
       - Manejo de desbalanceo con scale_pos_weight
    
    ### Teoría del Color Aplicada
    
    La aplicación utiliza una paleta de colores basada en psicología del color:
    
    - Azul (#4a90e2): Confianza, profesionalismo, botones principales
    - Verde (#2ecc71): Éxito, predicciones positivas
    - Rojo (#e74c3c): Alerta, predicciones negativas
    - Naranja (#e67e22): Energía, información secundaria
    - Púrpura (#9b59b6): Sofisticación, gráficas especiales
    - Gris (#95a5a6): Neutralidad, datos desconocidos
    
    ### Métricas de Evaluación
    
    - ROC-AUC: Área bajo la curva ROC, capacidad de discriminación
    - Precision: Proporción de predicciones positivas correctas
    - Recall: Proporción de casos positivos identificados
    - F1-Score: Media armónica entre Precision y Recall
    
    ### Tecnologías Utilizadas
    
    - Python 3.8 o superior
    - PyTorch para modelos de deep learning
    - Scikit-learn para modelos tradicionales y métricas
    - XGBoost para gradient boosting
    - Streamlit para la interfaz web interactiva
    - Plotly para visualizaciones interactivas
    - Pandas y NumPy para manipulación de datos
    
    ### Equipo de Desarrollo
    
    Proyecto desarrollado para el curso CC3066 - Data Science  
    Universidad del Valle de Guatemala - 2025
    
    ### Funcionalidades
    
    - Exploración de datos con dashboard y gráficas enlazadas  
    - Predicciones en tiempo real con selección de modelos  
    - Visualización de rendimiento con opción de mostrar u ocultar  
    - Diseño intuitivo con teoría del color aplicada  
    - Interactividad completa con Plotly  
    """,
    )

    st.markdown(
        """
        <br><br>
        <div style='text-align: center; padding: 2rem; background-color: #ecf0f1; border-radius: 12px;'>
            <p style='color: #7f8c8d; margin: 0;'>
                © 2025 - Sistema de Predicción de Compradores Recurrentes
            </p>
            <p style='color: #95a5a6; margin: 0.5rem 0 0 0; font-size: 0.9rem;'>
                Desarrollado con Streamlit y PyTorch
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
