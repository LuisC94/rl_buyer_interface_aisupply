import os
import sys
import tempfile
import io
import zipfile
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import torch
import joblib

# Adicionar o diretório atual ao path para garantir importações
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from training_runner import (
        train_single_core_generator,
        train_multi_core_generator,
        run_testing_simulation,
        train_mlp_forecaster_generator,
        train_autoformer_forecaster_generator,
        run_forecast_inference,
        populate_prediction_column
    )
except ImportError:
    from rl_buyer_interface.training_runner import (
        train_single_core_generator,
        train_multi_core_generator,
        run_testing_simulation,
        train_mlp_forecaster_generator,
        train_autoformer_forecaster_generator,
        run_forecast_inference,
        populate_prediction_column
    )

# --- CONFIGURAÇÃO DA PÁGINA STREAMLIT ---
st.set_page_config(
    page_title="RL Buyer Agent - Eureka Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- IDIOMA / LANGUAGE SELECTOR ---
lang = st.sidebar.selectbox("🌐 Language / Idioma", ["English", "Português"], index=0)

def T(pt, en):
    return en if lang == "English" else pt

# --- CUSTOM CSS: PREMIUM LIGHT ORANGE THEME ---
st.markdown("""
<style>
    @font-face {
        font-family: 'Neo Tech Std Medium';
        font-style: normal;
        font-weight: 500;
        src: local('Neo Tech Std Medium'), local('NeoTechStd-Medium'), local('Neo Tech Std');
    }
    
    html, body, [class*="css"], .stApp {
        font-family: 'Neo Tech Std Medium', 'Inter', -apple-system, sans-serif !important;
    }
    
    .stApp {
        background-color: #ffffff;
        color: #1e293b;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Cartões Modernos com Transições Suaves */
    .custom-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 22px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .custom-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.03);
        border-color: #cbd5e1;
    }
    
    /* Botões Streamlit Premium */
    .stButton>button {
        border-radius: 8px !important;
        background: linear-gradient(135deg, #f97316 0%, #ea580c 100%) !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 10px 24px !important;
        box-shadow: 0 4px 6px -1px rgba(234, 88, 12, 0.2) !important;
        transition: all 0.2s ease !important;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #ea580c 0%, #c2410c 100%) !important;
        box-shadow: 0 10px 15px -3px rgba(234, 88, 12, 0.3) !important;
        transform: translateY(-1px) !important;
    }
    .stButton>button:active {
        transform: translateY(1px) !important;
    }
    
    .main-title {
        background: linear-gradient(135deg, #f97316 0%, #ea580c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.8rem;
        margin-bottom: 5px;
    }
    .subtitle {
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }
    
    .console-header {
        background-color: #0f172a;
        border-radius: 8px 8px 0 0;
        border: 1px solid #1e293b;
        border-bottom: none;
        padding: 10px 18px;
        font-size: 0.85rem;
        color: #f97316;
        font-family: monospace;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .console-box {
        background-color: #020617;
        border: 1px solid #1e293b;
        border-radius: 0 0 8px 8px;
        padding: 15px;
        font-family: 'Courier New', Courier, monospace;
        font-size: 0.9rem;
        color: #38bdf8;
        height: 250px;
        overflow-y: auto;
        white-space: pre-wrap;
        box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.5);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 1px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 52px;
        font-size: 1.05rem;
        font-weight: 600;
        color: #64748b;
        border-bottom-width: 2px;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #ea580c;
    }
    .stTabs [aria-selected="true"] {
        color: #ea580c !important;
        border-bottom-color: #ea580c !important;
    }
    
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ea580c;
        margin: 5px 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# --- DIRECTÓRIO DE MODELOS ---
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# --- UTILS FOR FILE ZIP & DOWNLOADS ---
def create_model_zip_bytes(model_base_path):
    """ Lê os arquivos gerados do modelo e empacota-os num buffer ZIP em memória """
    zip_buffer = io.BytesIO()
    suffixes = ['_actor.pth', '_critic.pth', '_scaler.pth', '_econ_stat.pth']
    found_any = False
    
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for suffix in suffixes:
            file_path = model_base_path + suffix
            if os.path.exists(file_path):
                zip_file.write(file_path, os.path.basename(file_path))
                found_any = True
                
    if not found_any:
        return None
    return zip_buffer.getvalue()

# --- APP LAYOUT ---
col_logo, col_desc = st.columns([1, 12])
with col_desc:
    st.markdown('<h1 class="main-title">RL Buyer Agent - Eureka</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="subtitle">{T("Interface Interativa de Inteligência Artificial para Gestão Automática de Inventário Fruta & Logística", "Interactive Artificial Intelligence Interface for Automatic Fruit Inventory & Logistics Management")}</p>', unsafe_allow_html=True)

# ----------------- SIDEBAR: GLOBAL CONFIGS & LOT SELECTOR -----------------
with st.sidebar:
    lote_id = "uploaded_dataset"
    lote_label = T("Dataset Carregado", "Uploaded Dataset")
    
    st.markdown(f"### ⚙️ {T('Configuração Global', 'Global Configuration')}")
    device_opt = st.selectbox(T("Hardware de Execução (PyTorch):", "Execution Hardware (PyTorch):"), ["CPU", "GPU"], index=0)
    device = "cuda" if device_opt == "GPU" and torch.cuda.is_available() else "cpu"
    if device_opt == "GPU" and not torch.cuda.is_available():
        st.warning(T("Aceleração GPU (CUDA) indisponível. Usando CPU.", "GPU acceleration (CUDA) unavailable. Using CPU."))
        
    st.markdown("---")
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "exemplo_treino_previsoes.xlsx")
    if os.path.exists(template_path):
        try:
            with open(template_path, "rb") as f:
                template_bytes = f.read()
            st.download_button(
                label=T("📥 Descarregar Excel Exemplo", "📥 Download Example Excel"),
                data=template_bytes,
                file_name="exemplo_treino_inventario.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        except Exception as e_dl:
            st.error(f"Erro ao carregar template: {e_dl}")

# ----------------- SESSION STATE INITS -----------------
if 'forecast_df' not in st.session_state:
    st.session_state.forecast_df = {}
if 'train_log' not in st.session_state:
    st.session_state.train_log = ""
if 'test_log' not in st.session_state:
    st.session_state.test_log = ""
if 'test_completed' not in st.session_state:
    st.session_state.test_completed = False
if 'test_results' not in st.session_state:
    st.session_state.test_results = None
if 'trained_forecasters' not in st.session_state:
    st.session_state.trained_forecasters = set()
if 'trained_buyer' not in st.session_state:
    st.session_state.trained_buyer = False
if 'last_uploaded_file_name' not in st.session_state:
    st.session_state.last_uploaded_file_name = None

# ----------------- TABS PRINCIPAIS -----------------
tab_forecast_train, tab_buyer_train, tab_sim, tab_compare = st.tabs([
    T("📈 1. Treinar Previsões", "📈 1. Train Forecasting"),
    T("🏋️ 2. Treinar Buyer Agent", "🏋️ 2. Train Buyer Agent"),
    T("🧪 3. Simulação & Inferência", "🧪 3. Simulation & Inference"),
    T("📊 4. Comparar Modelos", "📊 4. Compare Models")
])

# Helpers para check de existência dos modelos
def get_forecaster_status(lote):
    mlp_exists = os.path.exists(os.path.join(MODELS_DIR, f"sales_mlp_{lote}.joblib"))
    auto_exists = os.path.exists(os.path.join(MODELS_DIR, f"sales_autoformer_{lote}.pt"))
    return mlp_exists, auto_exists

def get_buyer_status(lote):
    actor_exists = os.path.exists(os.path.join(MODELS_DIR, f"buyer_agent_{lote}_actor.pth"))
    return actor_exists

def create_diagnostics_chart(plot_days, orders, sales, spoilage, missed_sales, stock_levels):
    fig = go.Figure()
    
    # 1. Line: Orders (Encomendas)
    fig.add_trace(go.Scatter(
        x=plot_days, y=orders, mode='lines',
        name=T('Encomendas PPO (Ord)', 'PPO Orders (Ord)'),
        line=dict(color='#8b5cf6', width=2.5, shape='spline'),
        hovertemplate='%{y:.0f} un'
    ))
    
    # 2. Line: Sales (Vendas)
    fig.add_trace(go.Scatter(
        x=plot_days, y=sales, mode='lines',
        name=T('Vendas Efetuadas', 'Sales Made'),
        line=dict(color='#10b981', width=2.5, shape='spline'),
        hovertemplate='%{y:.0f} un'
    ))
    
    # 3. Markers: Spoilage (Expirados)
    spoil_x = []
    spoil_y = []
    for i, val in enumerate(spoilage):
        if val > 0:
            spoil_x.append(plot_days[i])
            spoil_y.append(sales[i])
            
    if spoil_x:
        fig.add_trace(go.Scatter(
            x=spoil_x, y=spoil_y, mode='markers',
            name=T('Expirados (Lixo)', 'Expired (Waste)'),
            marker=dict(symbol='square', size=9, color='#ef4444', line=dict(width=1.5, color='#ffffff')),
            hovertemplate=T('Dia %{x}: Produto Expirado!', 'Day %{x}: Product Expired!')
        ))
        
    # 4. Markers: Stockout / Missed Sales
    stockout_x = []
    stockout_y = []
    for i, val in enumerate(missed_sales):
        if val > 0 or stock_levels[i] <= 0:
            stockout_x.append(plot_days[i])
            stockout_y.append(sales[i])
            
    if stockout_x:
        fig.add_trace(go.Scatter(
            x=stockout_x, y=stockout_y, mode='markers',
            name=T('Stockout / Stock Zero', 'Stockout / Zero Stock'),
            marker=dict(symbol='triangle-up', size=10, color='#eab308', line=dict(width=1.5, color='#ffffff')),
            hovertemplate=T('Dia %{x}: Rotura de Stock!', 'Day %{x}: Out of Stock!')
        ))
        
    fig.update_layout(
        title=dict(
            text=T("Gráfico de Diagnóstico: Vendas vs Encomendas", "Diagnostic Chart: Sales vs Orders"),
            font=dict(size=16, weight='bold')
        ),
        xaxis=dict(title=T("Dias", "Days"), showgrid=True, gridcolor='#e2e8f0', linecolor='#cbd5e1'),
        yaxis=dict(title=T("Quantidade (un)", "Quantity (units)"), showgrid=True, gridcolor='#e2e8f0', linecolor='#cbd5e1'),
        hovermode="x unified",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        font=dict(family='"Segoe UI", "Roboto", sans-serif', color="#1e293b"),
        legend=dict(
            orientation="h", y=1.1, x=0.5, xanchor="center",
            bgcolor="rgba(255, 255, 255, 0.8)", bordercolor="#cbd5e1", borderwidth=1
        ),
        margin=dict(t=60, b=45, l=50, r=20)
    )
    return fig

# =====================================================================
# TAB 1: TREINAR MODELO DE PREVISÃO
# =====================================================================
with tab_forecast_train:
    st.markdown(f"### {T('Treinar Modelo de Previsão de Vendas', 'Train Sales Forecasting Model')}")
    
    col_fc1, col_fc2 = st.columns([1, 1])
    with col_fc1:
        st.markdown(f'<div class="custom-card">', unsafe_allow_html=True)
        st.markdown(f"##### {T('1. Configuração do Modelo', '1. Model Configuration')}")
        fc_model_type = st.radio(
            T("Tipo de Modelo de Previsão:", "Forecasting Model Type:"),
            ["MLP (Multi-Layer Perceptron)", "Autoformer (Deep Learning)"],
            key="fc_model_type_radio"
        )
        fc_type_id = "mlp" if "MLP" in fc_model_type else "autoformer"
        
        uploaded_fc_file = st.file_uploader(
            T("Histórico de Vendas (Excel ou CSV):", "Sales History (Excel or CSV):"),
            type=["xlsx", "csv"],
            key="fc_file_uploader",
            help=T("Colunas necessárias: date/Data, sales_quantity_kg/Vendas_Kg. Opcional: price_per_kg/Preco_Kg.",
                   "Required columns: date/Data, sales_quantity_kg/Vendas_Kg. Optional: price_per_kg/Preco_Kg.")
        )
        
        # Detecção de alteração de arquivo para fazer reset completo e carregar cache
        if uploaded_fc_file is not None:
            if st.session_state.get('last_uploaded_file_name') != uploaded_fc_file.name:
                try:
                    if uploaded_fc_file.name.endswith('.xlsx'):
                        df_temp = pd.read_excel(uploaded_fc_file)
                    else:
                        df_temp = pd.read_csv(uploaded_fc_file)
                    
                    # Renomear colunas
                    col_mapping = {
                        'Data': 'date', 'date': 'date',
                        'Vendas_Kg': 'sales_quantity_kg', 'sales_quantity_kg': 'sales_quantity_kg',
                        'Preco_Kg': 'price_per_kg', 'price_per_kg': 'price_per_kg',
                        'Validade/Prazo': 'validade', 'validade': 'validade', 'prazo': 'validade', 'shelf_life': 'validade', 'Validade': 'validade'
                    }
                    df_temp = df_temp.rename(columns=col_mapping)
                    df_temp['date'] = pd.to_datetime(df_temp['date']).dt.date
                    if 'price_per_kg' not in df_temp.columns:
                        df_temp['price_per_kg'] = 2.0
                        
                    keep_cols = ['date', 'sales_quantity_kg', 'price_per_kg']
                    if 'validade' in df_temp.columns:
                        keep_cols.append('validade')
                        df_temp['validade'] = pd.to_numeric(df_temp['validade'], errors='coerce')
                        
                    df_temp = df_temp[keep_cols].dropna(subset=['date', 'sales_quantity_kg'])
                    
                    st.session_state.forecast_df[lote_id] = df_temp
                    st.session_state.last_uploaded_file_name = uploaded_fc_file.name
                    st.session_state.trained_forecasters = set()
                    st.session_state.trained_buyer = False
                    st.session_state.test_completed = False
                    st.session_state.test_results = None
                    st.session_state.train_log = ""
                    st.session_state.test_log = ""
                except Exception as parse_ex:
                    st.error(f"Erro ao ler ficheiro de previsões: {parse_ex}")
        
        btn_train_fc = st.button(T("⚙️ Treinar Modelo de Previsão", "⚙️ Train Forecasting Model"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_fc2:
        st.markdown(f'<div class="custom-card">', unsafe_allow_html=True)
        st.markdown(f"##### {T('2. Inferência Rápida', '2. Quick Inference')}")
        st.markdown(f"##### {T('💡 Gerar Previsões de Vendas', '💡 Generate Sales Forecasts')}")
        
        active_trained_models = []
        if "mlp" in st.session_state.trained_forecasters:
            active_trained_models.append("MLP")
        if "autoformer" in st.session_state.trained_forecasters:
            active_trained_models.append("Autoformer")
        
        if len(active_trained_models) == 0:
            st.info(T("Treine um modelo primeiro para poder inferir previsões.", "Train a model first to run forecast inferences."))
            selected_infer_model = None
        else:
            selected_infer_model = st.selectbox(T("Modelo para Inferência:", "Model for Inference:"), active_trained_models)
            
        horizon_days = st.selectbox(T("Horizonte Temporal (Dias):", "Time Horizon (Days):"), [15, 30], index=1)
        btn_infer_fc = st.button(T("🔮 Gerar Previsão", "🔮 Generate Forecast"), use_container_width=True, disabled=(selected_infer_model is None))
        st.markdown('</div>', unsafe_allow_html=True)
        
    # Execução do treino preditivo
    if btn_train_fc:
        df_fc = st.session_state.forecast_df.get(lote_id)
        if df_fc is None:
            st.error(T("Por favor, faça upload de um ficheiro com o histórico de vendas na Aba 1.", "Please upload a file with sales history in Tab 1."))
        else:
            with st.spinner(T("A treinar modelo preditivo de vendas... Isto pode demorar alguns segundos.", "Training sales forecast model... This may take a few seconds.")):
                try:
                    # Executar treino
                    progress_bar = st.progress(0.0)
                    log_container = st.empty()
                    log_lines = []
                    
                    if fc_type_id == "mlp":
                        gen = train_mlp_forecaster_generator(lote_id, df_fc, MODELS_DIR)
                    else:
                        gen = train_autoformer_forecaster_generator(lote_id, df_fc, MODELS_DIR)
                        
                    for pct, log_msg in gen:
                        progress_bar.progress(pct / 100.0)
                        log_lines.append(log_msg)
                        # Mostrar as últimas 5 linhas do log para manter limpo
                        log_container.code("\n".join(log_lines[-5:]))
                        
                    if fc_type_id == "mlp":
                        st.session_state.trained_forecasters.add("mlp")
                    else:
                        st.session_state.trained_forecasters.add("autoformer")
                    st.success(T(f"🎉 Modelo {fc_model_type} treinado com sucesso!",
                                 f"🎉 Model {fc_model_type} trained successfully!"))
                    st.rerun()
                except Exception as ex:
                    st.error(f"{T('Erro durante o treino do forecaster:', 'Error during forecaster training:')} {ex}")

    # Execução da inferência preditiva
    if btn_infer_fc and selected_infer_model:
        history_df = st.session_state.forecast_df.get(lote_id)
        if history_df is None:
            # Tentar carregar a partir do ficheiro carregado se ainda estiver no uploader
            if uploaded_fc_file is not None:
                try:
                    if uploaded_fc_file.name.endswith('.xlsx'):
                        history_df = pd.read_excel(uploaded_fc_file)
                    else:
                        history_df = pd.read_csv(uploaded_fc_file)
                    col_mapping = {
                        'Data': 'date', 'date': 'date',
                        'Vendas_Kg': 'sales_quantity_kg', 'sales_quantity_kg': 'sales_quantity_kg',
                        'Preco_Kg': 'price_per_kg', 'price_per_kg': 'price_per_kg'
                    }
                    history_df = history_df.rename(columns=col_mapping)
                    history_df['date'] = pd.to_datetime(history_df['date']).dt.date
                    if 'price_per_kg' not in history_df.columns:
                        history_df['price_per_kg'] = 2.0
                    history_df = history_df[['date', 'sales_quantity_kg', 'price_per_kg']].dropna(subset=['date', 'sales_quantity_kg'])
                    st.session_state.forecast_df[lote_id] = history_df
                except:
                    pass
                    
        if history_df is None:
            st.error(T("Não foi encontrado histórico em memória. Por favor, forneça o ficheiro excel no uploader do lado esquerdo.",
                       "No sales history found in memory. Please upload the excel file on the left side."))
        else:
            with st.spinner(T("A gerar previsões futuras...", "Generating future forecasts...")):
                try:
                    infer_type_id = "mlp" if selected_infer_model == "MLP" else "autoformer"
                    preds = run_forecast_inference(lote_id, infer_type_id, history_df, horizon_days, MODELS_DIR)
                    
                    df_preds = pd.DataFrame(preds, columns=['Data', 'Vendas Previstas (Kg)'])
                    
                    st.markdown("##### 📊 " + T("Resultados das Vendas Previstas", "Forecasting Sales Results"))
                    
                    col_res1, col_res2 = st.columns([7, 5])
                    with col_res1:
                        # Gráfico Plotly das previsões
                        fig_fc = go.Figure()
                        # Últimos 15 dias históricos
                        df_hist_slice = history_df.tail(15)
                        fig_fc.add_trace(go.Scatter(x=df_hist_slice['date'], y=df_hist_slice['sales_quantity_kg'], mode='lines+markers', name=T('Histórico Real', 'Real History'), line=dict(color='#64748b')))
                        fig_fc.add_trace(go.Scatter(x=df_preds['Data'], y=df_preds['Vendas Previstas (Kg)'], mode='lines+markers', name=T('Previsão Futura', 'Future Forecast'), line=dict(color='#ff8c00', width=2.5)))
                        
                        fig_fc.update_layout(
                            title=f"{T('Previsão de Vendas', 'Sales Forecast')} ({selected_infer_model})",
                            xaxis_title=T("Data", "Date"),
                            yaxis_title=T("Quantidade (Kg)", "Quantity (Kg)"),
                            paper_bgcolor="#ffffff",
                            plot_bgcolor="#f8fafc",
                            font=dict(color="#1e293b"),
                            legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center")
                        )
                        st.plotly_chart(fig_fc, use_container_width=True)
                        
                    with col_res2:
                        # Tabela formatada
                        st.dataframe(df_preds, height=280)
                except Exception as ex:
                    st.error(f"{T('Erro ao gerar inferência preditiva:', 'Error generating predictive inference:')} {ex}")

# =====================================================================
# TAB 2: TREINO DO BUYER AGENT
# =====================================================================
with tab_buyer_train:
    st.markdown(f"### {T('Treino do Buyer Agent (Reinforcement Learning - PPO)', 'Train Buyer Agent (Reinforcement Learning - PPO)')}")
    mlp_trained = "mlp" in st.session_state.trained_forecasters
    auto_trained = "autoformer" in st.session_state.trained_forecasters
    
    if not mlp_trained and not auto_trained:
        st.warning(T("⚠️ É necessário treinar um modelo de previsão de vendas primeiro na Aba 1 antes de treinar o Buyer Agent.",
                     "⚠️ You must train a sales forecasting model first in Tab 1 before training the Buyer Agent."))
    else:
        st.success(T("✅ Modelo de previsão de vendas detetado. Podes avançar para o treino do Buyer Agent.",
                     "✅ Sales forecasting model detected. You can proceed with Buyer Agent training."))
        
        col_by1, col_by2 = st.columns([1, 1])
        with col_by1:
            st.markdown(f'<div class="custom-card">', unsafe_allow_html=True)
            st.markdown(f"##### {T('1. Dados & Modelo Preditor Associado', '1. Data & Associated Predictor Model')}")
            
            available_forecasts = []
            if mlp_trained: available_forecasts.append("MLP")
            if auto_trained: available_forecasts.append("Autoformer")
            
            selected_forecast_type = st.selectbox(
                T("Modelo preditor a associar ao Buyer Agent:", "Forecaster model to associate with Buyer Agent:"),
                available_forecasts
            )
            forecast_type_id = "mlp" if selected_forecast_type == "MLP" else "autoformer"
            
            train_split = st.slider(T("Percentagem (%) de Dados para Treino:", "Percentage (%) of Data for Training:"), min_value=30, max_value=90, value=60, step=5, key="buyer_train_split")
            max_capacity = st.number_input(T("Capacidade Máxima do Armazém (unidades):", "Maximum Warehouse Capacity (units):"), min_value=100, max_value=5000, value=500, step=100, key="buyer_max_cap")
            max_shelf_life = st.number_input(
                T("Validade do Produto (Dias):", "Product Shelf Life (Days):"),
                min_value=2,
                max_value=120,
                value=15,
                step=1,
                key="buyer_max_shelf_life"
            )
            max_episodes = st.number_input(
                T("Número Máximo de Simulações:", "Maximum Number of Training Simulations:  Min-64, Recommended-20000"),
                min_value=64,
                max_value=20000,
                value=64, 
                step=64, 
                key="buyer_max_episodes"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_by2:
            st.markdown(f'<div class="custom-card">', unsafe_allow_html=True)
            st.markdown(f"##### {T('2. Penalizações & Custos do Armazém', '2. Warehouse Penalties & Costs')}")
            
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                holding_cost = st.number_input(T("Custo Armazenamento (€/m³/dia):", "Storage Cost (€/m³/day):"), min_value=0.0, max_value=10.0, value=0.70, step=0.05, format="%.2f", key="by_hold")
                transport_cost = st.number_input(T("Custo Transporte (€/m³):", "Transport Cost (€/m³):"), min_value=0.0, max_value=100.0, value=10.00, step=0.50, format="%.2f", key="by_trans")
                fixed_transport_cost = st.number_input(T("Taxa Fixa Camião (€):", "Fixed Truck Fee (€):"), min_value=0.0, max_value=500.0, value=10.00, step=1.00, format="%.2f", key="by_fixed_trans")
                product_volume_val = st.number_input(T("Volume do Produto (m³):", "Product Volume (m³):"), min_value=0.0001, max_value=10.0, value=0.002, step=0.0005, format="%.4f", key="by_prod_volume")
            with col_b2:
                stockout_penalty_val = st.number_input(T("Penalização Vendas Perdidas (% preço):", "Lost Sales Penalty (% price):"), min_value=0.0, max_value=200.0, value=25.0, step=5.0, key="by_stockout_pct") / 100.0
                waste_penalty_val = st.number_input(T("Penalização Desperdício (% preço):", "Waste Penalty (% price):"), min_value=0.0, max_value=500.0, value=100.0, step=10.0, key="by_waste_pct") / 100.0
                zero_stock_penalty_val = st.number_input(T("Penalização Stockout / Stock Zero (% preço):", "Stockout / Zero Stock Penalty (% price):"), min_value=0.0, max_value=1000.0, value=500.0, step=50.0, key="by_zero_pct") / 100.0
            
            core_mode = st.radio(T("Modo de Processamento:", "Processing Mode:"), [T("Single-Core (Estável em Cloud)", "Single-Core (Stable in Cloud)"), T("Multi-Core (Mais Rápido Localmente)", "Multi-Core (Faster Locally)")], index=0, key="by_core_mode")
            
            workers = 4
            if "Multi-Core" in core_mode or "Faster Locally" in core_mode:
                workers = st.slider(T("Número de Cores Físicos (Workers):", "Number of Physical Cores (Workers):"), min_value=2, max_value=16, value=4, step=2, key="by_workers")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        btn_train_buyer = st.button(T("🚀 Iniciar Treino do Buyer Agent", "🚀 Start Buyer Agent Training"), use_container_width=True)
        
        # Console de Logs
        st.markdown(f'<div class="console-header">📁 {T("Terminal do Buyer Agent - Logs de Treino", "Buyer Agent Terminal - Training Logs")}</div>', unsafe_allow_html=True)
        console_placeholder = st.empty()
        console_placeholder.markdown(f'<div class="console-box">{T("Agente à espera de comando...", "Agent waiting for command...")}</div>', unsafe_allow_html=True)
        
        if btn_train_buyer:
            st.session_state.train_log = ""
            
            # Buscar dataset apropriado
            df_buyer_train = st.session_state.forecast_df.get(lote_id)
                
            if df_buyer_train is None:
                st.error(T("Histórico de vendas não encontrado! Por favor, carregue o dataset na Aba 1.",
                           "Sales history not found! Please upload the dataset in Tab 1."))
            else:
                with st.spinner(T("A preparar dados e a povoar coluna de previsões...", "Preparing data and populating predictions column...")):
                    try:
                        # Limpar colunas e renomear
                        col_mapping = {
                            'Data': 'date', 'date': 'date',
                            'Vendas_Kg': 'sales_quantity_kg', 'sales_quantity_kg': 'sales_quantity_kg',
                            'Preco_Kg': 'price_per_kg', 'price_per_kg': 'price_per_kg',
                            'Validade/Prazo': 'validade', 'validade': 'validade', 'prazo': 'validade', 'shelf_life': 'validade', 'Validade': 'validade'
                        }
                        df_buyer_train = df_buyer_train.rename(columns=col_mapping)
                        df_buyer_train['date'] = pd.to_datetime(df_buyer_train['date']).dt.date
                        if 'price_per_kg' not in df_buyer_train.columns:
                            df_buyer_train['price_per_kg'] = 2.0
                            
                        keep_cols = ['date', 'sales_quantity_kg', 'price_per_kg']
                        if 'validade' in df_buyer_train.columns:
                            keep_cols.append('validade')
                            df_buyer_train['validade'] = pd.to_numeric(df_buyer_train['validade'], errors='coerce')
                            
                        df_buyer_train = df_buyer_train[keep_cols].dropna(subset=['date', 'sales_quantity_kg'])
                        
                        # Alimentar o uploader de previsões a partir do modelo pré-treinado
                        df_buyer_with_pred = populate_prediction_column(lote_id, forecast_type_id, df_buyer_train, MODELS_DIR)
                        
                        # Renomear para o formato do ambiente
                        rename_dict = {
                            'sales_quantity_kg': 'real_value',
                            'price_per_kg': 'price',
                            'date': 'day'
                        }
                        df_env_format = df_buyer_with_pred.rename(columns=rename_dict)
                        if 'volume' not in df_env_format.columns:
                            df_env_format['volume'] = product_volume_val
                        if 'validade' not in df_env_format.columns:
                            df_env_format['validade'] = max_shelf_life
                            
                        # Escrever arquivo temporário no disco
                        temp_buyer_excel_path = os.path.join(tempfile.gettempdir(), f"buyer_train_temp_{lote_id}.xlsx")
                        df_env_format.to_excel(temp_buyer_excel_path, index=False)
                        
                        # Configurar variáveis de treino
                        lr_act = 0.0003
                        lr_crit = 0.001
                        gamma = 0.8
                        eps_clip = 0.2
                        k_epochs = 30
                        batch_size = 2048
                        max_episodes = int(st.session_state.get("buyer_max_episodes", 640))
                        seed = 1337
                        
                        temp_model_dir = MODELS_DIR
                        model_base_name = f"buyer_agent_{lote_id}"
                        
                        # Selecionar modo de cores/cores do CPU
                        if "Single-Core" in core_mode:
                            gen = train_single_core_generator(
                                seed=seed,
                                excel_path=temp_buyer_excel_path,
                                train_split=(train_split / 100.0),
                                max_capacity=max_capacity,
                                lr_actor=lr_act,
                                lr_critic=lr_crit,
                                gamma=gamma,
                                k_epochs=k_epochs,
                                eps_clip=eps_clip,
                                batch_size=batch_size,
                                max_episodes_total=max_episodes,
                                num_envs=64,
                                save_dir=temp_model_dir,
                                holding_cost=holding_cost,
                                transport_cost=transport_cost,
                                fixed_transport_cost=fixed_transport_cost,
                                stockout_penalty=stockout_penalty_val,
                                waste_penalty=waste_penalty_val,
                                zero_stock_penalty=zero_stock_penalty_val,
                                max_shelf_life=max_shelf_life
                            )
                        else:
                            gen = train_multi_core_generator(
                                seed=seed,
                                excel_path=temp_buyer_excel_path,
                                train_split=(train_split / 100.0),
                                max_capacity=max_capacity,
                                lr_actor=lr_act,
                                lr_critic=lr_crit,
                                gamma=gamma,
                                k_epochs=k_epochs,
                                eps_clip=eps_clip,
                                batch_size=batch_size,
                                max_episodes_total=max_episodes,
                                num_envs=64,
                                num_workers=workers,
                                save_dir=temp_model_dir,
                                holding_cost=holding_cost,
                                transport_cost=transport_cost,
                                fixed_transport_cost=fixed_transport_cost,
                                stockout_penalty=stockout_penalty_val,
                                waste_penalty=waste_penalty_val,
                                zero_stock_penalty=zero_stock_penalty_val,
                                max_shelf_life=max_shelf_life
                            )
                            
                        # Renomear os ficheiros finais gerados para conter o lote_id específico
                        # para podermos guardar múltiplos lotes ao mesmo tempo
                        progress_bar = st.progress(0.0)
                        
                        for log_line in gen:
                            st.session_state.train_log += log_line + "\n"
                            console_placeholder.markdown(
                                f'<div class="console-box">{st.session_state.train_log}</div>',
                                unsafe_allow_html=True
                            )
                            
                            if "Episodes:" in log_line:
                                try:
                                    parts = log_line.split("Episodes:")[1].split("/")[0].strip()
                                    current_ep = int(parts)
                                    pct = min(1.0, current_ep / max_episodes)
                                    progress_bar.progress(pct)
                                except:
                                    pass
                                    
                        progress_bar.progress(1.0)
                        
                        # Renomear checkpoints genéricos para o base name correto do lote
                        generic_base = os.path.join(temp_model_dir, "ppo_constrained_final")
                        target_base = os.path.join(temp_model_dir, model_base_name)
                        
                        suffixes = ['_actor.pth', '_critic.pth', '_scaler.pth', '_econ_stat.pth']
                        for suffix in suffixes:
                            if os.path.exists(generic_base + suffix):
                                if os.path.exists(target_base + suffix):
                                    os.remove(target_base + suffix)
                                os.rename(generic_base + suffix, target_base + suffix)
                                
                        st.session_state.trained_buyer = True
                        st.success(T("🎉 Buyer Agent treinado e salvo com sucesso!",
                                     "🎉 Buyer Agent trained and saved successfully!"))
                        st.rerun()
                        
                    except Exception as run_ex:
                        st.error(f"{T('Erro ao rodar treino do Buyer Agent:', 'Error running Buyer Agent training:')} {run_ex}")

# =====================================================================
# TAB 3: SIMULAÇÃO E INFERÊNCIA DO BUYER AGENT
# =====================================================================
with tab_sim:
    st.markdown(f"### {T('Simulação em Mercado Real e Inferência de Compras', 'Real Market Simulation & Purchase Inference')}")
    
    forecaster_trained = len(st.session_state.trained_forecasters) > 0
    buyer_trained = st.session_state.trained_buyer
    
    if not forecaster_trained or not buyer_trained:
        st.warning(T("⚠️ Deves treinar ambos o Modelo de Previsão (Aba 1) e o Buyer Agent (Aba 2) antes de correres a simulação.",
                     "⚠️ You must train both the Forecasting Model (Tab 1) and the Buyer Agent (Tab 2) before running the simulation."))
    else:
        st.success(T("✅ Modelos prontos! Configura os parâmetros e inicia a simulação.", "✅ Models ready! Configure parameters and start the simulation."))
        
        # Recuperar parâmetros do Buyer Agent definidos no Treino (Aba 2)
        max_shelf_life = st.session_state.get("buyer_max_shelf_life", 15)
        product_volume_val = st.session_state.get("by_prod_volume", 0.002)
        
        # Escolher qual o modelo preditivo a utilizar
        available_forecasts = []
        if "mlp" in st.session_state.trained_forecasters: available_forecasts.append("MLP")
        if "autoformer" in st.session_state.trained_forecasters: available_forecasts.append("Autoformer")
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown(f'<div class="custom-card">', unsafe_allow_html=True)
            st.markdown(f"##### {T('1. Origem dos Dados de Teste', '1. Test Data Origin')}")
            
            selected_forecast_type = st.selectbox(
                T("Usar previsões geradas pelo modelo:", "Use forecasts generated by model:"),
                available_forecasts,
                key="sim_forecast_select"
            )
            forecast_type_id = "mlp" if selected_forecast_type == "MLP" else "autoformer"
            
            test_split_option = st.radio(
                T("Origem do dataset de teste:", "Test dataset origin:"),
                [T("Usar os restantes (100 - X)% do dataset carregado na Aba 1", "Use the remaining (100 - X)% of the dataset loaded in Tab 1"),
                 T("Carregar novo ficheiro excel de teste (com dias futuros)", "Upload new test excel file (with future days)")]
            )
            
            test_split_val = 0.6
            df_test_data = None
            
            if "Usar os restantes" in test_split_option or "Use the remaining" in test_split_option:
                test_split_val = st.session_state.get("buyer_train_split", 60) / 100.0
                df_test_data = st.session_state.forecast_df.get(lote_id)
                
                if df_test_data is None:
                    st.info(T("Por favor, carregue o histórico de vendas na Aba 1.",
                              "Please upload sales history in Tab 1."))
            else:
                uploaded_test_file = st.file_uploader(
                    T("Dataset de Teste / Futuro:", "Test / Future Dataset:"),
                    type=["xlsx", "csv"],
                    key="sim_test_uploader"
                )
                if uploaded_test_file is not None:
                    if uploaded_test_file.name.endswith('.xlsx'):
                        df_test_data = pd.read_excel(uploaded_test_file)
                    else:
                        df_test_data = pd.read_csv(uploaded_test_file)
                        
                    # Renomear colunas
                    col_mapping = {
                        'Data': 'date', 'date': 'date',
                        'Vendas_Kg': 'sales_quantity_kg', 'sales_quantity_kg': 'sales_quantity_kg',
                        'Preco_Kg': 'price_per_kg', 'price_per_kg': 'price_per_kg',
                        'Validade/Prazo': 'validade', 'validade': 'validade', 'prazo': 'validade', 'shelf_life': 'validade', 'Validade': 'validade'
                    }
                    df_test_data = df_test_data.rename(columns=col_mapping)
                    df_test_data['date'] = pd.to_datetime(df_test_data['date']).dt.date
                    if 'price_per_kg' not in df_test_data.columns:
                        df_test_data['price_per_kg'] = 2.0
                        
                    keep_cols = ['date', 'sales_quantity_kg', 'price_per_kg']
                    if 'validade' in df_test_data.columns:
                        keep_cols.append('validade')
                        df_test_data['validade'] = pd.to_numeric(df_test_data['validade'], errors='coerce')
                        
                    df_test_data = df_test_data[keep_cols].dropna(subset=['date', 'sales_quantity_kg'])
                    test_split_val = 0.0 # Usar todo o ficheiro novo
                    
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_s2:
            st.markdown(f'<div class="custom-card" style="height: 100%;">', unsafe_allow_html=True)
            st.markdown(f"##### {T('2. Parâmetros da Simulação & Baselines', '2. Simulation Parameters & Baselines')}")
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                s_min = st.number_input(T("Valor Mínimo Baseline (s):", "Minimum Baseline Value (s):"), min_value=1, max_value=500, value=24, help=T("Ponto de reposição.", "Reorder point."))
                S_max = st.number_input(T("Valor Máximo Baseline (S):", "Maximum Baseline Value (S):"), min_value=50, max_value=2000, value=60, help=T("Stock máximo target.", "Max stock target."))
            with col_t2:
                update_interval = st.number_input(T("Intervalo de Fine-Tuning (Dias):", "Fine-Tuning Interval (Days):"), min_value=1, max_value=90, value=15, help=T("Frequência de atualização contínua.", "Frequency of continuous updates."))
                max_capacity = st.number_input(T("Capacidade Máxima Armazém:", "Max Warehouse Capacity:"), min_value=100, max_value=5000, value=st.session_state.get("buyer_max_cap", 500), step=100, key="sim_max_cap")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Botão para correr simulação
        btn_test = st.button(T("🏁 Iniciar Simulação e Teste Contínuo", "🏁 Start Simulation & Continuous Test"), use_container_width=True)
        
        col_chart, col_side_logs = st.columns([7, 5])
        with col_chart:
            chart_placeholder = st.empty()
            chart_placeholder_diagnostics = st.empty()
        with col_side_logs:
            st.markdown(f'<div class="console-header">🤖 {T("Logs Diários da Simulação", "Daily Simulation Logs")}</div>', unsafe_allow_html=True)
            test_console_placeholder = st.empty()
            test_console_placeholder.markdown(f'<div class="console-box">{T("Simulação parada...", "Simulation stopped...")}</div>', unsafe_allow_html=True)
            
        if btn_test:
            if df_test_data is None:
                st.error(T("Dados de teste em falta!", "Test data missing!"))
            else:
                st.session_state.test_log = ""
                st.session_state.test_completed = False
                
                with st.spinner(T("A preparar simulação e a correr inferência preditiva...", "Preparing simulation and running forecasting inference...")):
                    try:
                        # Limpar colunas e povoar previsão
                        col_mapping = {
                            'Data': 'date', 'date': 'date',
                            'Vendas_Kg': 'sales_quantity_kg', 'sales_quantity_kg': 'sales_quantity_kg',
                            'Preco_Kg': 'price_per_kg', 'price_per_kg': 'price_per_kg'
                        }
                        df_test_cleaned = df_test_data.rename(columns=col_mapping)
                        df_test_cleaned['date'] = pd.to_datetime(df_test_cleaned['date']).dt.date
                        if 'price_per_kg' not in df_test_cleaned.columns:
                            df_test_cleaned['price_per_kg'] = 2.0
                        df_test_cleaned = df_test_cleaned[['date', 'sales_quantity_kg', 'price_per_kg']].dropna(subset=['date', 'sales_quantity_kg'])
                        
                        # Populate
                        df_test_with_pred = populate_prediction_column(lote_id, forecast_type_id, df_test_cleaned, MODELS_DIR)
                        
                        rename_dict = {
                            'sales_quantity_kg': 'real_value',
                            'price_per_kg': 'price',
                            'date': 'day'
                        }
                        df_env_test_format = df_test_with_pred.rename(columns=rename_dict)
                        if 'volume' not in df_env_test_format.columns:
                            df_env_test_format['volume'] = product_volume_val
                        if 'validade' not in df_env_test_format.columns:
                            df_env_test_format['validade'] = max_shelf_life
                            
                        # Gravar ficheiro temporário para simulação
                        temp_sim_excel_path = os.path.join(tempfile.gettempdir(), f"buyer_sim_temp_{lote_id}.xlsx")
                        df_env_test_format.to_excel(temp_sim_excel_path, index=False)
                        
                        initial_model_base_path = os.path.join(MODELS_DIR, f"buyer_agent_{lote_id}")
                        
                        # Instanciar o gerador da simulação
                        sim_gen = run_testing_simulation(
                            excel_path=temp_sim_excel_path,
                            train_split=test_split_val,
                            max_capacity=max_capacity,
                            initial_model_base_path=initial_model_base_path,
                            s_min=s_min,
                            S_max=S_max,
                            update_interval_days=update_interval,
                            online_lr_actor=1e-5,
                            online_lr_critic=5e-5,
                            online_batch_size=32,
                            save_dir=MODELS_DIR,
                            holding_cost=st.session_state.get("by_hold", 0.70),
                            transport_cost=st.session_state.get("by_trans", 10.00),
                            fixed_transport_cost=st.session_state.get("by_fixed_trans", 10.00),
                            stockout_penalty=st.session_state.get("by_stockout_pct", 25.0)/100.0,
                            waste_penalty=st.session_state.get("by_waste_pct", 100.0)/100.0,
                            zero_stock_penalty=st.session_state.get("by_zero_pct", 500.0)/100.0,
                            max_shelf_life=max_shelf_life
                        )
                        
                        plot_data = {
                            "Dia": [], "RL Agent": [], "Min-Max": [], "Oracle": [], "Real Demand": [], "Stock Level": [],
                            "Agent Action": [], "Agent Sales": [], "Spoilage": [], "Missed Sales": []
                        }
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='RL Agent (Profit Acumulado)', line=dict(color='#ff8c00', width=2.5)))
                        fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Min-Max Baseline', line=dict(color='#64748b', width=1.5, dash='dot')))
                        fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Oráculo (God Mode)', line=dict(color='#0ea5e9', width=2.0, dash='dash')))
                        fig.update_layout(
                            title=T("Evolução Comparativa do Lucro Acumulado em Tempo Real", "Real-Time Cumulative Profit Comparison"),
                            xaxis_title=T("Dias", "Days"),
                            yaxis_title=T("Lucro Acumulado (€)", "Cumulative Profit (€)"),
                            paper_bgcolor="#ffffff",
                            plot_bgcolor="#f8fafc",
                            font=dict(color="#1e293b"),
                            legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center")
                        )
                        chart_placeholder.plotly_chart(fig, use_container_width=True)
                        
                        for sim_step in sim_gen:
                            status = sim_step.get("status")
                            msg = sim_step.get("msg")
                            
                            if status in ["init", "start", "error", "warning"]:
                                st.session_state.test_log += msg + "\n"
                                test_console_placeholder.markdown(
                                    f'<div class="console-box">{st.session_state.test_log}</div>',
                                    unsafe_allow_html=True
                                )
                                if status == "error":
                                    st.error(T("Erro crítico na simulação.", "Critical simulation error."))
                                    break
                                    
                            elif status == "running":
                                st.session_state.test_log += msg + "\n"
                                lines = st.session_state.test_log.split("\n")
                                visible_logs = "\n".join(lines[-15:])
                                test_console_placeholder.markdown(
                                    f'<div class="console-box">{visible_logs}</div>',
                                    unsafe_allow_html=True
                                )
                                
                                plot_data["Dia"].append(sim_step["day"])
                                plot_data["RL Agent"].append(sim_step["agent_profit_cum"])
                                plot_data["Min-Max"].append(sim_step["minmax_profit_cum"])
                                plot_data["Oracle"].append(sim_step["oracle_profit_cum"])
                                plot_data["Real Demand"].append(sim_step["real_demand"])
                                plot_data["Stock Level"].append(sim_step["stock_level"])
                                plot_data["Agent Action"].append(sim_step["agent_action"])
                                plot_data["Agent Sales"].append(sim_step["agent_sales"])
                                plot_data["Spoilage"].append(sim_step["spoilage"])
                                plot_data["Missed Sales"].append(sim_step["missed_sales"])
                                
                                if sim_step["day"] % 3 == 0 or sim_step["update_triggered"]:
                                    fig_real = go.Figure()
                                    fig_real.add_trace(go.Scatter(
                                        x=plot_data["Dia"], y=plot_data["RL Agent"], mode='lines',
                                        name=f'RL Agent ({plot_data["RL Agent"][-1]:.0f}€)',
                                        line=dict(color='#ff6b00', width=3.0, shape='spline'),
                                        hovertemplate='%{y:,.2f}€'
                                    ))
                                    fig_real.add_trace(go.Scatter(
                                        x=plot_data["Dia"], y=plot_data["Min-Max"], mode='lines',
                                        name=f'Min-Max ({plot_data["Min-Max"][-1]:.0f}€)',
                                        line=dict(color='#94a3b8', width=2.0, dash='dot', shape='spline'),
                                        hovertemplate='%{y:,.2f}€'
                                    ))
                                    fig_real.add_trace(go.Scatter(
                                        x=plot_data["Dia"], y=plot_data["Oracle"], mode='lines',
                                        name=f'Oráculo ({plot_data["Oracle"][-1]:.0f}€)',
                                        line=dict(color='#3b82f6', width=2.0, dash='dash', shape='spline'),
                                        hovertemplate='%{y:,.2f}€'
                                    ))
                                    fig_real.update_layout(
                                        title=dict(
                                            text=T("Evolução Comparativa do Lucro Acumulado em Tempo Real", "Real-Time Cumulative Profit Comparison"),
                                            font=dict(size=16, weight='bold')
                                        ),
                                        xaxis=dict(title=T("Dias", "Days"), showgrid=True, gridcolor='#e2e8f0', linecolor='#cbd5e1'),
                                        yaxis=dict(title=T("Lucro Acumulado (€)", "Cumulative Profit (€)"), showgrid=True, gridcolor='#e2e8f0', linecolor='#cbd5e1'),
                                        paper_bgcolor="#ffffff",
                                        plot_bgcolor="#f8fafc",
                                        hovermode="x unified",
                                        font=dict(family='"Segoe UI", "Roboto", sans-serif', color="#1e293b"),
                                        legend=dict(
                                            orientation="h", y=1.1, x=0.5, xanchor="center",
                                            bgcolor="rgba(255, 255, 255, 0.8)", bordercolor="#cbd5e1", borderwidth=1
                                        ),
                                        margin=dict(t=60, b=40, l=50, r=20)
                                    )
                                    chart_placeholder.plotly_chart(fig_real, use_container_width=True)
                                    
                                    fig_diag = create_diagnostics_chart(
                                        plot_data["Dia"],
                                        plot_data["Agent Action"],
                                        plot_data["Agent Sales"],
                                        plot_data["Spoilage"],
                                        plot_data["Missed Sales"],
                                        plot_data["Stock Level"]
                                    )
                                    chart_placeholder_diagnostics.plotly_chart(fig_diag, use_container_width=True)
                                    
                            elif status == "complete":
                                st.session_state.test_log += msg + "\n"
                                test_console_placeholder.markdown(
                                    f'<div class="console-box">{st.session_state.test_log}</div>',
                                    unsafe_allow_html=True
                                )
                                st.session_state.test_completed = True
                                st.session_state.test_results = sim_step
                                
                                fig_final = go.Figure()
                                fig_final.add_trace(go.Scatter(
                                    x=plot_data["Dia"], y=plot_data["RL Agent"], mode='lines',
                                    name=f'RL Agent ({sim_step["cum_profit_agent"]:.1f}€)',
                                    line=dict(color='#ff6b00', width=3.0, shape='spline'),
                                    hovertemplate='%{y:,.2f}€'
                                ))
                                fig_final.add_trace(go.Scatter(
                                    x=plot_data["Dia"], y=plot_data["Min-Max"], mode='lines',
                                    name=f'Min-Max ({sim_step["cum_profit_minmax"]:.1f}€)',
                                    line=dict(color='#94a3b8', width=2.0, dash='dot', shape='spline'),
                                    hovertemplate='%{y:,.2f}€'
                                ))
                                fig_final.add_trace(go.Scatter(
                                    x=plot_data["Dia"], y=plot_data["Oracle"], mode='lines',
                                    name=f'Oráculo ({sim_step["cum_profit_oracle"]:.1f}€)',
                                    line=dict(color='#3b82f6', width=2.0, dash='dash', shape='spline'),
                                    hovertemplate='%{y:,.2f}€'
                                ))
                                
                                for ud in sim_step["update_days"]:
                                    fig_final.add_vline(x=ud, line_width=1.2, line_dash="dash", line_color="#ef4444", annotation_text="Fine-Tuning", annotation_position="top left", annotation_font=dict(color="#b91c1c", size=10))
                                    
                                fig_final.update_layout(
                                    title=dict(
                                        text=T("Evolução Comparativa do Lucro Acumulado Final", "Final Cumulative Profit Comparison"),
                                        font=dict(size=16, weight='bold')
                                    ),
                                    xaxis=dict(title=T("Dias", "Days"), showgrid=True, gridcolor='#e2e8f0', linecolor='#cbd5e1'),
                                    yaxis=dict(title=T("Lucro Acumulado (€)", "Cumulative Profit (€)"), showgrid=True, gridcolor='#e2e8f0', linecolor='#cbd5e1'),
                                    paper_bgcolor="#ffffff",
                                    plot_bgcolor="#f8fafc",
                                    hovermode="x unified",
                                    font=dict(family='"Segoe UI", "Roboto", sans-serif', color="#1e293b"),
                                    legend=dict(
                                        orientation="h", y=1.1, x=0.5, xanchor="center",
                                        bgcolor="rgba(255, 255, 255, 0.8)", bordercolor="#cbd5e1", borderwidth=1
                                    ),
                                    margin=dict(t=60, b=45, l=50, r=20)
                                )
                                chart_placeholder.plotly_chart(fig_final, use_container_width=True)
                                st.rerun()
                                
                    except Exception as sim_ex:
                        st.error(f"{T('Erro ao rodar simulação do Buyer Agent:', 'Error running Buyer Agent simulation:')} {sim_ex}")

        # Exibir Scorecard e downloaders ao completar a simulação
        if st.session_state.test_completed and st.session_state.test_results is not None:
            res = st.session_state.test_results
            
            st.markdown("---")
            st.markdown(f"### 📊 {T('Scorecard de Resultados', 'Results Scorecard')}")
            
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-label">{T("Lucro Agente", "Agent Profit")}</div>'
                    f'<div class="metric-val">{res["cum_profit_agent"]:,.2f}€</div>'
                    f'</div>', unsafe_allow_html=True
                )
            with c2:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-label">{T("Lucro Min-Max", "Min-Max Profit")}</div>'
                    f'<div class="metric-val" style="color: #64748b;">{res["cum_profit_minmax"]:,.2f}€</div>'
                    f'</div>', unsafe_allow_html=True
                )
            with c3:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-label">{T("Produtos Expirados", "Expired Products")}</div>'
                    f'<div class="metric-val" style="color: #ef4444;">{res["spoilage_total"]:.0f} un</div>'
                    f'</div>', unsafe_allow_html=True
                )
            with c4:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-label">{T("Vendas Perdidas", "Lost Sales")}</div>'
                    f'<div class="metric-val" style="color: #eab308;">{res["lost_sales_total"]:.0f} un</div>'
                    f'</div>', unsafe_allow_html=True
                )
            with c5:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-label">{T("Dias Stock Zero", "Zero Stock Days")}</div>'
                    f'<div class="metric-val" style="color: #ef4444;">{res["stockout_days"]} dias</div>'
                    f'</div>', unsafe_allow_html=True
                )
                
            profit_diff = res["cum_profit_agent"] - res["cum_profit_minmax"]
            pct_improvement = (profit_diff / max(1.0, abs(res["cum_profit_minmax"]))) * 100.0
            
            st.markdown(f"#### {T('Resumo Executivo', 'Executive Summary')}")
            if profit_diff > 0:
                st.success(T(f"📈 O Agente PPO superou o baseline Min-Max tradicional em **{profit_diff:.2f}€** (+{pct_improvement:.2f}%).",
                             f"📈 The PPO Agent outperformed the traditional Min-Max baseline by **{profit_diff:.2f}€** (+{pct_improvement:.2f}%)."))
            else:
                st.warning(T(f"📉 O Agente PPO obteve um lucro inferior ao baseline Min-Max tradicional em **{abs(profit_diff):.2f}€** ({pct_improvement:.2f}%).",
                             f"📉 The PPO Agent underperformed the traditional Min-Max baseline by **{abs(profit_diff):.2f}€** ({pct_improvement:.2f}%)."))
                
            # Gráfico de evolução comparativa do lucro final
            if "log_dias" in res:
                st.markdown(f"#### {T('Gráfico de Desempenho Comparativo', 'Comparative Performance Chart')}")
                fig_final = go.Figure()
                fig_final.add_trace(go.Scatter(
                    x=res["log_dias"], y=res["log_lucro_acumulado_agente"], mode='lines',
                    name=f'RL Agent ({res["cum_profit_agent"]:.1f}€)',
                    line=dict(color='#ff6b00', width=3.0, shape='spline'),
                    hovertemplate='%{y:,.2f}€'
                ))
                fig_final.add_trace(go.Scatter(
                    x=res["log_dias"], y=res["log_lucro_acumulado_minmax"], mode='lines',
                    name=f'Min-Max ({res["cum_profit_minmax"]:.1f}€)',
                    line=dict(color='#94a3b8', width=2.0, dash='dot', shape='spline'),
                    hovertemplate='%{y:,.2f}€'
                ))
                fig_final.add_trace(go.Scatter(
                    x=res["log_dias"], y=res["log_lucro_acumulado_oracle"], mode='lines',
                    name=f'Oráculo ({res["cum_profit_oracle"]:.1f}€)',
                    line=dict(color='#3b82f6', width=2.0, dash='dash', shape='spline'),
                    hovertemplate='%{y:,.2f}€'
                ))
                
                for ud in res.get("update_days", []):
                    fig_final.add_vline(x=ud, line_width=1.2, line_dash="dash", line_color="#ef4444", annotation_text="Fine-Tuning", annotation_position="top left", annotation_font=dict(color="#b91c1c", size=10))
                    
                fig_final.update_layout(
                    title=dict(
                        text=T("Evolução Comparativa do Lucro Acumulado Final", "Final Cumulative Profit Comparison"),
                        font=dict(size=16, weight='bold')
                    ),
                    xaxis=dict(title=T("Dias", "Days"), showgrid=True, gridcolor='#e2e8f0', linecolor='#cbd5e1'),
                    yaxis=dict(title=T("Lucro Acumulado (€)", "Cumulative Profit (€)"), showgrid=True, gridcolor='#e2e8f0', linecolor='#cbd5e1'),
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#f8fafc",
                    hovermode="x unified",
                    font=dict(family='"Segoe UI", "Roboto", sans-serif', color="#1e293b"),
                    legend=dict(
                        orientation="h", y=1.1, x=0.5, xanchor="center",
                        bgcolor="rgba(255, 255, 255, 0.8)", bordercolor="#cbd5e1", borderwidth=1
                    ),
                    margin=dict(t=60, b=45, l=50, r=20)
                )
                st.plotly_chart(fig_final, use_container_width=True)
                
                # Gráfico de Diagnóstico: Vendas vs Encomendas
                st.markdown(f"#### {T('Gráfico de Diagnóstico: Vendas vs Encomendas', 'Diagnostic Chart: Sales vs Orders')}")
                fig_diag_final = create_diagnostics_chart(
                    res["log_dias"],
                    res.get("log_acoes_agente", []),
                    res.get("log_vendas_agente", []),
                    res.get("log_apodrecimento_agente", []),
                    res.get("log_vendas_perdidas_agente", []),
                    res.get("log_stock_final_agente", [])
                )
                st.plotly_chart(fig_diag_final, use_container_width=True)
                
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.markdown(f"#### 📥 {T('Exportar Resultados e Modelo Otimizado', 'Export Results & Optimized Model')}")
            col_down1, col_down2 = st.columns(2)
            
            with col_down1:
                if os.path.exists(res["excel_report_path"]):
                    with open(res["excel_report_path"], "rb") as f:
                        st.download_button(
                            label=T("📊 Descarregar Relatório Excel (.xlsx)", "📊 Download Excel Report (.xlsx)"),
                            data=f.read(),
                            file_name="relatorio_simulacao_buyer_agent.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                        
            with col_down2:
                zip_final_bytes = create_model_zip_bytes(res["final_model_path"])
                if zip_final_bytes is not None:
                    st.download_button(
                        label=T("📥 Descarregar Modelo Otimizado / Fine-Tuned (.ZIP)", "📥 Download Optimized / Fine-Tuned Model (.ZIP)"),
                        data=zip_final_bytes,
                        file_name="modelo_ppo_buyer_agent_fine_tuned.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
            st.markdown('</div>', unsafe_allow_html=True)

# =====================================================================
# TAB 4: INFERÊNCIAS E COMPARAÇÃO DE PREVISÕES
# =====================================================================
with tab_compare:
    st.markdown(f"### {T('Comparação Visual e Métricas de Previsão', 'Visual Comparison & Forecasting Metrics')}")
    
    mlp_trained = "mlp" in st.session_state.trained_forecasters
    auto_trained = "autoformer" in st.session_state.trained_forecasters
    
    if not mlp_trained and not auto_trained:
        st.warning(T("⚠️ Nenhum modelo de previsão foi treinado ainda. Por favor, vá à Aba 1 para treinar.",
                     "⚠️ No forecasting models trained yet. Please go to Tab 1 to train."))
    else:
        st.markdown(f'<div class="custom-card">', unsafe_allow_html=True)
        st.markdown(f"##### {T('1. Seleção de Modelos e Parâmetros', '1. Model Selection & Parameters')}")
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            models_to_compare = []
            if mlp_trained:
                compare_mlp = st.checkbox(T("Comparar Modelo MLP", "Compare MLP Model"), value=True)
                if compare_mlp: models_to_compare.append("MLP")
            if auto_trained:
                compare_auto = st.checkbox(T("Comparar Modelo Autoformer", "Compare Autoformer Model"), value=True)
                if compare_auto: models_to_compare.append("Autoformer")
        with col_c2:
            compare_horizon = st.selectbox(
                T("Horizonte de Previsão para Comparação (Dias):", "Forecasting Horizon for Comparison (Days):"),
                [15, 30],
                index=1,
                key="compare_horizon"
            )
            
        btn_compare = st.button(T("📊 Executar Comparação", "📊 Run Comparison"), use_container_width=True, disabled=len(models_to_compare) == 0)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if btn_compare:
            history_df = st.session_state.forecast_df.get(lote_id)
            if history_df is None:
                # Carregar o uploader da Aba 1 se possível
                fc_file = st.session_state.get("fc_file_uploader")
                if fc_file is not None:
                    try:
                        if fc_file.name.endswith('.xlsx'):
                            history_df = pd.read_excel(fc_file)
                        else:
                            history_df = pd.read_csv(fc_file)
                        col_mapping = {
                            'Data': 'date', 'date': 'date',
                            'Vendas_Kg': 'sales_quantity_kg', 'sales_quantity_kg': 'sales_quantity_kg',
                            'Preco_Kg': 'price_per_kg', 'price_per_kg': 'price_per_kg'
                        }
                        history_df = history_df.rename(columns=col_mapping)
                        history_df['date'] = pd.to_datetime(history_df['date']).dt.date
                        if 'price_per_kg' not in history_df.columns:
                            history_df['price_per_kg'] = 2.0
                        history_df = history_df[['date', 'sales_quantity_kg', 'price_per_kg']].dropna(subset=['date', 'sales_quantity_kg'])
                        st.session_state.forecast_df[lote_id] = history_df
                    except:
                        pass
                        
            if history_df is None:
                st.error(T("Erro: Carregue o histórico de vendas na Aba 1 para podermos cruzar os dados de lags para a previsão.",
                           "Error: Upload sales history in Tab 1 so we can compute the lag features for prediction."))
            else:
                with st.spinner(T("A rodar inferências comparativas...", "Running comparative inferences...")):
                    try:
                        fig_comp = go.Figure()
                        
                        # Plotar últimos 20 dias de histórico real
                        df_hist_slice = history_df.tail(20)
                        fig_comp.add_trace(go.Scatter(
                            x=df_hist_slice['date'], y=df_hist_slice['sales_quantity_kg'],
                            mode='lines+markers', name=T('Histórico Real', 'Real History'),
                            line=dict(color='#64748b', width=2.5, shape='spline'),
                            hovertemplate='%{y:,.2f} Kg'
                        ))
                        
                        colors_map = {"MLP": "#8b5cf6", "Autoformer": "#06b6d4"} # MLP purple, Autoformer cyan/teal
                        
                        df_table_compare = pd.DataFrame()
                        
                        for m_type in models_to_compare:
                            m_id = "mlp" if m_type == "MLP" else "autoformer"
                            preds = run_forecast_inference(lote_id, m_id, history_df, compare_horizon, MODELS_DIR)
                            
                            df_m_preds = pd.DataFrame(preds, columns=['Data', f'{m_type} (Kg)'])
                            if df_table_compare.empty:
                                df_table_compare['Data'] = df_m_preds['Data']
                            df_table_compare[f'{m_type} (Kg)'] = df_m_preds[f'{m_type} (Kg)']
                            
                            fig_comp.add_trace(go.Scatter(
                                x=df_m_preds['Data'],
                                y=df_m_preds[f'{m_type} (Kg)'],
                                mode='lines+markers',
                                name=f'Previsão {m_type}',
                                line=dict(color=colors_map[m_type], width=3.0, shape='spline'),
                                hovertemplate='%{y:,.2f} Kg'
                            ))
                            
                        fig_comp.update_layout(
                            title=dict(
                                text=T("Comparação de Modelos de Previsão", "Forecasting Models Comparison"),
                                font=dict(size=16, weight='bold')
                            ),
                            xaxis=dict(title=T("Data", "Date"), showgrid=True, gridcolor='#e2e8f0', linecolor='#cbd5e1'),
                            yaxis=dict(title=T("Quantidade Prevista (Kg)", "Predicted Quantity (Kg)"), showgrid=True, gridcolor='#e2e8f0', linecolor='#cbd5e1'),
                            paper_bgcolor="#ffffff",
                            plot_bgcolor="#f8fafc",
                            hovermode="x unified",
                            font=dict(family='"Segoe UI", "Roboto", sans-serif', color="#1e293b"),
                            legend=dict(
                                orientation="h", y=1.1, x=0.5, xanchor="center",
                                bgcolor="rgba(255, 255, 255, 0.8)", bordercolor="#cbd5e1", borderwidth=1
                            ),
                            margin=dict(t=60, b=45, l=50, r=20)
                        )
                        
                        col_comp_plot, col_comp_tbl = st.columns([7, 5])
                        with col_comp_plot:
                            st.plotly_chart(fig_comp, use_container_width=True)
                        with col_comp_tbl:
                            st.markdown(f"**{T('Tabela de Dados Comparativos', 'Comparative Data Table')}**")
                            # Format numbers
                            for col in df_table_compare.columns:
                                if col != 'Data':
                                    df_table_compare[col] = df_table_compare[col].map('{:,.2f}'.format)
                            st.dataframe(df_table_compare, height=280)
                            
                    except Exception as comp_ex:
                        st.error(f"{T('Erro na comparação dos modelos:', 'Error in model comparison:')} {comp_ex}")
