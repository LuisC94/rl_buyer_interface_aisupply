import os
import sys
import json
import random
import time
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =====================================================================
# --- BUYER AGENT: DSAC LIMITADO + LIFECYCLE LC v0.2.0 EMBUTIDO ---
# =====================================================================
# Motor copiado do buyer_agent_service (Retail-Eureka / Web Services / buyer_agent_service / engine):
#   - dsac_engine/reference/: DSAC original (C:\buyerAgent\DSAC) e ambiente original (30 variáveis de estado)
#   - dsac_engine/environment_lc.py: ambiente original com o motor biológico substituído pelo LC v0.2.0
# Modo LIMITADO (= DSAC/train_dsac.py sem --deslimitado): encomenda máxima diária = maior venda do treino.

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_DSAC_REF_DIR = os.path.join(_APP_DIR, "dsac_engine", "reference")
if _DSAC_REF_DIR not in sys.path:
    sys.path.insert(0, _DSAC_REF_DIR)

try:
    from dsac_engine.environment_lc import StockEnvironmentLC
    from dsac_engine.lc_kernel import LC_PRESETS, resolve_lc_fruit_key
except ImportError:
    from rl_buyer_interface.dsac_engine.environment_lc import StockEnvironmentLC
    from rl_buyer_interface.dsac_engine.lc_kernel import LC_PRESETS, resolve_lc_fruit_key

from dsac_agent import DSACAgent  # noqa: E402  (ficheiro original do DSAC)
from replay_buffer import ReplayBuffer  # noqa: E402  (ficheiro original do DSAC)

# Hiperparâmetros originais (C:\buyerAgent\DSAC\train_dsac.py)
STATE_DIM = 30
ACTION_DIM = 1
BATCH_SIZE = 256
WARMUP_STEPS = 500
DEFAULT_EPISODES = 350
EVAL_INTERVAL = 10
BUFFER_SIZE = 300_000
NUM_QUANTILES = 32
RISK_FACTOR = 0.85
SEED = 1337

# Clima por defeito quando o dataset não tem colunas de clima (mesmos defaults do ambiente)
CLIMATE_DEFAULTS = {"temperature": 22.0, "humidity": 70.0, "ethylene": 0.8}
OPTIONAL_LC_COLUMNS = ["fruit_type", "temperature", "humidity", "ethylene"]
LC_COLUMN_ALIASES = {
    'Fruta': 'fruit_type', 'fruta': 'fruit_type', 'fruit': 'fruit_type', 'Fruit': 'fruit_type',
    'Temperatura': 'temperature', 'temperatura': 'temperature', 'Temperature': 'temperature',
    'Humidade': 'humidity', 'humidade': 'humidity', 'Humidity': 'humidity',
    'Etileno': 'ethylene', 'etileno': 'ethylene', 'Ethylene': 'ethylene',
}

MODEL_FILE_SUFFIXES = ['_actor.pth', '_critic.pth', '_alpha.pth', '_meta.json']


def lc_fruit_options():
    """ Lista (chave, label) das frutas do Lifecycle LC v0.2.0, ordenada pelo label """
    return sorted([(k, p.get("label", k)) for k, p in LC_PRESETS.items()], key=lambda x: x[1])


def prepare_lc_dataframe(df, fruit_key, temperature=None, humidity=None, ethylene=None):
    """
    Garante as colunas que o ambiente DSAC + LC precisa: fruit_type, temperature, humidity, ethylene.
    Colunas já presentes no dataset têm prioridade; os valores em falta são preenchidos com os da UI.
    """
    df = df.copy()
    if 'fruit_type' not in df.columns or df['fruit_type'].isna().all():
        df['fruit_type'] = fruit_key
    else:
        df['fruit_type'] = df['fruit_type'].ffill().bfill()
    ui_values = {"temperature": temperature, "humidity": humidity, "ethylene": ethylene}
    for col, ui_val in ui_values.items():
        fallback = float(ui_val) if ui_val is not None else CLIMATE_DEFAULTS[col]
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(fallback)
        else:
            df[col] = fallback
    return df


def _read_frame(excel_path):
    if isinstance(excel_path, pd.DataFrame):
        return excel_path.copy()
    return pd.read_excel(excel_path)


def _set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_agent(device="cpu", risk_factor=RISK_FACTOR):
    return DSACAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM, gamma=0.99, tau=0.005, lr_actor=3e-4,
                     lr_critic=3e-4, lr_alpha=3e-4, num_quantiles=NUM_QUANTILES, risk_factor=risk_factor,
                     device=device)


def _action_to_order(action_norm, max_order_limit):
    """ Ação do ator em [-1, 1] -> percentagem [0, 1] -> unidades (igual a train_dsac.py) """
    action_pct = 0.5 * (float(action_norm[0]) + 1.0)
    return round(action_pct * max_order_limit), action_pct


def load_model_meta(model_base_path):
    path = model_base_path + '_meta.json'
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _save_meta(model_base_path, meta):
    with open(model_base_path + '_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def evaluate_policy(agent, eval_env):
    """ Avaliação determinística no conjunto de teste (igual a train_dsac.py) """
    state = eval_env.reset()
    done = False
    total_profit = total_sales = total_lost = total_waste = 0.0
    while not done:
        action_norm = agent.select_action(state, evaluate=True)
        action_physical, _ = _action_to_order(action_norm, eval_env.max_order_limit)
        next_state, _, done, info = eval_env.step(action_physical, update_stats=False)
        total_profit += info['profit']
        total_sales += info['sales']
        total_lost += info['missed_sales']
        total_waste += info['overflow_waste']
        state = next_state
    return {"profit": total_profit, "sales": total_sales, "missed_sales": total_lost, "waste": total_waste}


# --- TREINO DSAC LIMITADO ---
def train_dsac_generator(seed, excel_path, train_split, max_capacity, max_episodes_total, save_dir, model_name,
                         fruit_key, temperature=None, humidity=None, ethylene=None, device="cpu", num_threads=1,
                         risk_factor=RISK_FACTOR):
    """
    Treina o DSAC Limitado com o mesmo ciclo de C:\\buyerAgent\\DSAC\\train_dsac.py, no ambiente com o
    Lifecycle LC v0.2.0 embutido. Faz yield de linhas de log ("Episodes: X/Y | ..." para a barra de progresso).
    Guarda {save_dir}/{model_name}_actor/_critic/_alpha.pth (melhor avaliação), *_latest_* e {model_name}_meta.json.
    """
    t0 = time.time()
    _set_seed(seed)
    torch.set_num_threads(max(1, int(num_threads)))
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    yield f"[INFO] A iniciar treino DSAC Limitado (Distributional SAC) em {device.upper()} com {torch.get_num_threads()} thread(s)..."

    df = prepare_lc_dataframe(_read_frame(excel_path), fruit_key, temperature, humidity, ethylene)
    common = dict(train_split=train_split, max_capacity=max_capacity, completely_unconstrained=False)
    env_train = StockEnvironmentLC(df, is_training=True, **common)
    env_eval = StockEnvironmentLC(df, is_training=False, shared_stats={'econ': env_train.stat_profit},
                                  lc_model=env_train.lc_model, **common)
    if len(env_train.data) < 2 or len(env_eval.data) < 2:
        raise ValueError("Dados insuficientes: são precisos pelo menos 2 dias no treino e 2 dias na avaliação.")

    max_order_limit = env_train.max_order_limit
    fruit_label = LC_PRESETS[env_train.lc_fruit_key].get("label", env_train.lc_fruit_key)
    lc_desc = "Modelo 1 (com etileno)" if env_train.lc_model == 1 else "Modelo 2 (sem etileno)"
    yield f"[INFO] Lifecycle LC v0.2.0: {fruit_label} ({env_train.lc_fruit_key}) | {lc_desc}"
    yield f"[INFO] Max Daily Order Limit (MAX_ORDER_LIMIT) = {max_order_limit} units (Warehouse Capacity = {max_capacity} units)"
    yield f"[INFO] Dias de treino = {len(env_train.data)} | Dias de avaliação = {len(env_eval.data)} | State dimension = {STATE_DIM}, Action dimension = {ACTION_DIM}"
    yield f"[INFO] Fator de Risco Quântico (CVaR) = {risk_factor:.2f} | Warmup = {WARMUP_STEPS} passos | Batch = {BATCH_SIZE}"

    agent = _make_agent(device, risk_factor)
    replay_buffer = ReplayBuffer(state_dim=STATE_DIM, action_dim=ACTION_DIM, max_size=BUFFER_SIZE, device=device)
    os.makedirs(save_dir, exist_ok=True)
    model_base = os.path.join(save_dir, model_name)

    train_profits, eval_profits, critic_losses, actor_losses, alpha_history = [], [], [], [], []
    best_eval_profit = -float("inf")
    best_metrics = None
    total_steps = 0
    max_episodes_total = int(max_episodes_total)

    yield "[START] Training started successfully."

    for episode in range(1, max_episodes_total + 1):
        state = env_train.reset()
        done = False
        episode_profit = 0.0
        ep_critic_loss, ep_actor_loss = [], []

        while not done:
            total_steps += 1
            if total_steps < WARMUP_STEPS:
                action_norm = np.random.uniform(-1.0, 1.0, size=(ACTION_DIM,)).astype(np.float32)
            else:
                action_norm = agent.select_action(state, evaluate=False)
            physical_action, _ = _action_to_order(action_norm, max_order_limit)

            next_state, _, done, info = env_train.step(physical_action)
            episode_profit += info.get('commercial_profit', info['profit'])
            # Recompensa nativa do ambiente (já inclui o waste shaping de treino)
            reward_stationary = float(np.clip(info['profit'] / 100.0, -10.0, 10.0))
            replay_buffer.push(state, action_norm, reward_stationary, next_state, done)
            state = next_state

            if total_steps >= WARMUP_STEPS:
                metrics = agent.update(replay_buffer, batch_size=BATCH_SIZE)
                if metrics:
                    ep_critic_loss.append(metrics["loss_critic"])
                    ep_actor_loss.append(metrics["loss_actor"])
                    alpha_history.append(metrics["alpha"])

        train_profits.append(episode_profit)
        avg_c_loss = float(np.mean(ep_critic_loss)) if ep_critic_loss else 0.0
        avg_a_loss = float(np.mean(ep_actor_loss)) if ep_actor_loss else 0.0
        critic_losses.append(avg_c_loss)
        actor_losses.append(avg_a_loss)

        yield (f"Episodes: {episode}/{max_episodes_total} | Lucro Treino: {episode_profit:.2f}€ | "
               f"Critic Loss: {avg_c_loss:.4f} | Actor Loss: {avg_a_loss:.4f} | Alpha: {float(agent.alpha.item()):.4f}")

        if episode % EVAL_INTERVAL == 0:
            m = evaluate_policy(agent, env_eval)
            eval_profits.append((episode, m["profit"]))
            line = (f">>> [AVALIAÇÃO EP {episode}] Lucro Teste: {m['profit']:.2f}€ | Vendas: {m['sales']:.0f} | "
                    f"Rutura: {m['missed_sales']:.0f} | Desperdício: {m['waste']:.0f}")
            if m["profit"] > best_eval_profit:
                best_eval_profit = float(m["profit"])
                best_metrics = m
                agent.save(save_dir, filename_prefix=model_name)
                line += " | ★ Novo melhor modelo guardado"
            agent.save(save_dir, filename_prefix=f"{model_name}_latest")
            yield line

    # Garantir que existe sempre um modelo "best" (se episódios < EVAL_INTERVAL)
    if best_metrics is None:
        best_metrics = evaluate_policy(agent, env_eval)
        best_eval_profit = float(best_metrics["profit"])
        agent.save(save_dir, filename_prefix=model_name)
        agent.save(save_dir, filename_prefix=f"{model_name}_latest")

    # Dados de treino guardados para o fine-tuning online (experiências antigas, mistura 80/20)
    train_data_path = model_base + '_train_data.xlsx'
    env_train.data.to_excel(train_data_path, index=False)

    meta = {
        "algorithm": "DSAC_Limitado",
        "state_dim": STATE_DIM,
        "fruit_key": env_train.lc_fruit_key,
        "fruit_label": fruit_label,
        "lifecycle_engine": f"LC v0.2.0 (modelo {env_train.lc_model})",
        "lc_model": int(env_train.lc_model),
        "owner_type": env_train.owner_type,
        "packaging": env_train.packaging,
        "climate_defaults": {
            "temperature": float(temperature if temperature is not None else CLIMATE_DEFAULTS["temperature"]),
            "humidity": float(humidity if humidity is not None else CLIMATE_DEFAULTS["humidity"]),
            "ethylene": float(ethylene if ethylene is not None else CLIMATE_DEFAULTS["ethylene"]),
        },
        "max_capacity": float(max_capacity),
        "max_order_limit": float(max_order_limit),
        "risk_factor": float(risk_factor),
        "episodes": max_episodes_total,
        "train_split": float(train_split),
        "days_train": int(len(env_train.data)),
        "days_eval": int(len(env_eval.data)),
        "best_eval_profit": round(best_eval_profit, 2),
        "best_eval_metrics": {k: round(float(v), 2) for k, v in best_metrics.items()},
        "train_data_file": os.path.basename(train_data_path),
        "training_time_seconds": round(time.time() - t0, 1),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save_meta(model_base, meta)

    # Gráfico de evolução do treino
    try:
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        axs[0, 0].plot(train_profits, color="#1f77b4")
        axs[0, 0].set_title("Evolução do Lucro no Treino (DSAC Limitado)")
        axs[0, 0].set_xlabel("Episódio"); axs[0, 0].set_ylabel("Lucro (€)"); axs[0, 0].grid(True, alpha=0.3)
        if eval_profits:
            ex, ey = zip(*eval_profits)
            axs[0, 1].plot(ex, ey, marker="o", color="#2ca02c")
        axs[0, 1].set_title(f"Lucro na Avaliação (Melhor: {best_eval_profit:.2f} €)")
        axs[0, 1].set_xlabel("Episódio"); axs[0, 1].set_ylabel("Lucro (€)"); axs[0, 1].grid(True, alpha=0.3)
        axs[1, 0].plot(critic_losses, color="#d62728")
        axs[1, 0].set_title("Perda do Crítico Quântico (Z1 + Z2)")
        axs[1, 0].set_xlabel("Episódio"); axs[1, 0].grid(True, alpha=0.3)
        axs[1, 1].plot(alpha_history, color="#9467bd")
        axs[1, 1].set_title("Temperatura da Entropia (Alpha)")
        axs[1, 1].set_xlabel("Passos de Gradiente"); axs[1, 1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "losses_plot.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
        yield f"[WARNING] Could not create loss chart: {e}"

    yield f"[INFO] Melhor lucro na avaliação: {best_eval_profit:.2f}€ | Tempo de treino: {meta['training_time_seconds']:.0f}s"
    yield "[SUCCESS] DSAC Limitado training completed successfully!"


# --- FINE-TUNING ONLINE (OFF-POLICY) ---
def continual_training_step(agent, new_experiences, env_train, train_state, max_action_val,
                            lr_actor=1e-5, lr_critic=5e-5, batch_size=32, device="cpu"):
    """
    1 ciclo de atualização online do DSAC com buffer misto (80% experiências antigas do ambiente de treino,
    20% dias novos da simulação). Devolve o estado atual do ambiente de treino (continua no ciclo seguinte).
    """
    for opt, lr in ((agent.actor_optimizer, lr_actor), (agent.critic_optimizer, lr_critic),
                    (agent.alpha_optimizer, lr_actor)):
        for param_group in opt.param_groups:
            param_group['lr'] = lr

    num_new_days = len(new_experiences)
    num_old_needed = num_new_days * 4  # proporção 80/20
    buffer = ReplayBuffer(state_dim=STATE_DIM, action_dim=ACTION_DIM, max_size=num_new_days + num_old_needed + 1,
                          device=device)

    # 1. Experiências antigas geradas no ambiente de treino com a política atual (estocástica)
    state = train_state if train_state is not None else env_train.reset()
    for _ in range(num_old_needed):
        action_norm = agent.select_action(state, evaluate=False)
        physical_action, _ = _action_to_order(action_norm, max_action_val)
        next_state, _, done, info = env_train.step(physical_action)
        buffer.push(state, action_norm, float(np.clip(info['profit'] / 100.0, -10.0, 10.0)), next_state, done)
        state = env_train.reset() if done else next_state

    # 2. Injetar os dias novos
    for exp in new_experiences:
        buffer.push(exp['state'], exp['action'], exp['reward'], exp['next_state'], exp['is_terminal'])

    # 3. Atualizar pesos (1 passo de gradiente por transição no buffer)
    if len(buffer) >= batch_size:
        for _ in range(len(buffer)):
            agent.update(buffer, batch_size=batch_size)
    return state


# --- DISRUPÇÕES ADAPTADAS AO LIFECYCLE ---
def _force_stock(env, value):
    """ Substitui o stock por um lote fresco de 'value' unidades (ou esvazia se 0) """
    env.active_batches = [] if value <= 0 else [env.make_batch(value, age_days=0)]
    env._update_stock_profile_from_batches()


def _degrade_new_batches(env, lot_ids_before, target_rsl_days):
    """
    Qualidade inferior: os lotes que deram entrada hoje chegam já envelhecidos, até a vida útil
    restante projetada pelo Lifecycle ser <= target_rsl_days.
    """
    changed = False
    for i, b in enumerate(env.active_batches):
        if b.get('lot_id') in lot_ids_before or b['quantity'] <= 0:
            continue
        for _ in range(120):
            if env.project_batch_rsl(b) <= target_rsl_days or b.get('quality', 100.0) < 30.0:
                break
            b = env.advance_batch_one_day(b, *env._today_climate())
        env.active_batches[i] = b
        changed = True
    if changed:
        env._update_stock_profile_from_batches()


# --- SIMULATION RUNNER ---
def run_testing_simulation(excel_path, train_split, max_capacity, initial_model_base_path, s_min, S_max,
                           update_interval_days=15, online_lr_actor=1e-5, online_lr_critic=5e-5, online_batch_size=32,
                           save_dir="models", temperature=None, humidity=None, ethylene=None, disruptions=None,
                           device="cpu"):
    """
    Runs evaluation simulation, comparing the DSAC Limitado agent with Min-Max, Time Supply and Floating Point.
    Todos os ambientes usam o Lifecycle LC v0.2.0 embutido. Yields data dictionary daily for Streamlit live charting.
    """
    yield {"status": "init", "msg": "[INIT] Preparando ambientes de teste (DSAC Limitado + Lifecycle LC v0.2.0)..."}
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    meta = load_model_meta(initial_model_base_path)
    if meta is None or meta.get("state_dim") != STATE_DIM:
        yield {"status": "error", "msg": "[CRITICAL ERROR] Modelo DSAC não encontrado. Treine o Buyer Agent na Aba 3."}
        return

    clim = meta.get("climate_defaults", {})
    temperature = clim.get("temperature") if temperature is None else temperature
    humidity = clim.get("humidity") if humidity is None else humidity
    ethylene = clim.get("ethylene") if ethylene is None else ethylene

    df = _read_frame(excel_path)
    # O modelo foi treinado para uma fruta: a simulação usa sempre essa fruta
    df['fruit_type'] = meta["fruit_key"]
    df = prepare_lc_dataframe(df, meta["fruit_key"], temperature, humidity, ethylene)

    # 1. Initialize Environments (modo teste: lucro comercial, sem shaping de treino)
    common = dict(is_training=False, train_split=train_split, max_capacity=max_capacity,
                  completely_unconstrained=False, lc_model=meta.get("lc_model"),
                  owner_type=meta.get("owner_type", "Retailer"), packaging=meta.get("packaging", "bulk"))
    env_test = StockEnvironmentLC(df, **common)
    env_minmax = StockEnvironmentLC(df, **common)
    env_timesupply = StockEnvironmentLC(df, **common)
    env_floatingpoint = StockEnvironmentLC(df, **common)
    for env in (env_minmax, env_timesupply, env_floatingpoint):
        env.max_order_limit = float('inf')  # Sem limite de encomenda para os Baselines

    # Limite de encomenda do agente = o do treino (escala da ação do ator)
    max_order_limit = float(meta["max_order_limit"])
    env_test.max_order_limit = max_order_limit
    all_envs = [env_test, env_minmax, env_timesupply, env_floatingpoint]

    # Ambiente de treino (experiências antigas para o fine-tuning)
    env_train = None
    train_data_path = os.path.join(os.path.dirname(initial_model_base_path), meta.get("train_data_file", ""))
    if meta.get("train_data_file") and os.path.exists(train_data_path):
        df_train = prepare_lc_dataframe(pd.read_excel(train_data_path), meta["fruit_key"], temperature, humidity, ethylene)
        env_train = StockEnvironmentLC(df_train, is_training=True, train_split=1.0, max_capacity=max_capacity,
                                       completely_unconstrained=False, lc_model=meta.get("lc_model"),
                                       owner_type=meta.get("owner_type", "Retailer"), packaging=meta.get("packaging", "bulk"))
        env_train.max_order_limit = max_order_limit
    else:
        yield {"status": "warning", "msg": "[WARNING] Dados de treino do modelo não encontrados: fine-tuning online desativado."}

    lc_desc = "modelo 1, com etileno" if env_test.lc_model == 1 else "modelo 2, sem etileno"
    yield {"status": "init", "msg": f"[INIT] Lifecycle LC v0.2.0: {meta.get('fruit_label', meta['fruit_key'])} ({lc_desc})"}
    yield {"status": "init", "msg": f"[INIT] Max Daily Order Limit (Action Cap): {max_order_limit} units"}

    # 2. Instantiate and Load Agent
    agent = _make_agent(device, meta.get("risk_factor", RISK_FACTOR))
    try:
        agent.load(os.path.dirname(initial_model_base_path), filename_prefix=os.path.basename(initial_model_base_path))
        agent.actor.eval()
        yield {"status": "init", "msg": f"[OK] Modelo DSAC Limitado carregado (treinado em {meta.get('trained_at', '?')})!"}
    except Exception as e:
        yield {"status": "error", "msg": f"[CRITICAL ERROR] Failed to load base model: {e}"}
        return

    state = env_test.reset()
    for env in all_envs[1:]:
        env.reset()
    train_state = None

    done = False

    # Tracking variables
    profits_agent = []
    profits_minmax = []
    profits_timesupply = []
    profits_floatingpoint = []
    update_days = []

    cum_profit_agent = 0
    cum_profit_minmax = 0
    cum_profit_timesupply = 0
    cum_profit_floatingpoint = 0
    dias_simulados = 0

    # Log variables for Excel exporter
    log_dias = []
    log_procura_real = []
    log_preco_venda = []

    log_acoes_agente = []
    log_acoes_minmax = []
    log_acoes_timesupply = []
    log_acoes_floatingpoint = []

    log_stock_inicial_agente = []
    log_stock_final_agente = []
    log_vendas_agente = []
    log_vendas_perdidas_agente = []
    log_apodrecimento_agente = []
    log_excesso_agente = []

    log_lucro_diario_agente = []
    log_lucro_acumulado_agente = []
    log_lucro_acumulado_minmax = []
    log_lucro_acumulado_timesupply = []
    log_lucro_acumulado_floatingpoint = []

    # Min-Max detailed logs
    log_stock_inicial_minmax = []
    log_stock_final_minmax = []
    log_vendas_minmax = []
    log_vendas_perdidas_minmax = []
    log_apodrecimento_minmax = []
    log_excesso_minmax = []
    log_lucro_diario_minmax = []

    # Time Supply and Floating Point detailed logs
    log_stock_inicial_timesupply = []
    log_stock_final_timesupply = []
    log_vendas_timesupply = []
    log_lucro_diario_timesupply = []

    log_stock_inicial_floatingpoint = []
    log_stock_final_floatingpoint = []
    log_vendas_floatingpoint = []
    log_lucro_diario_floatingpoint = []

    # Lifecycle logs (agente)
    log_rsl_min_agente = []
    log_markdown_agente = []

    flag_stockout = []
    flag_clientes_perdidos = []
    flag_excesso_armazem = []
    flag_apodrecimento = []

    current_15d_buffer = []

    if disruptions is None:
        disruptions = []

    yield {"status": "start", "msg": "[PRODUCTION] Continuous market simulation started."}

    while not done:
        day = env_test.current_step

        # Guardar valores originais de atributos temporários para restauro no fim do dia
        orig_real_demands = {}
        for env in all_envs:
            orig_real_demands[env] = env.data.at[day, 'real_value']

        # Processar disrupções para o dia atual (1-indexed na UI)
        day_idx = day + 1
        day_disruptions = [d for d in disruptions if d["day"] == day_idx]
        shelf_life_target = None
        for dis in day_disruptions:
            dis_type = dis.get("type")
            dis_param = dis.get("param")
            dis_val = dis.get("value")

            # 1. FORÇAR STOCK (lote fresco)
            if dis_type == "stock_override":
                val = float(dis_val)
                for env in all_envs:
                    _force_stock(env, val)

            # 2. PERDA DE CARGA EM TRÂNSITO (Afeta a mercadoria que chega HOJE)
            elif dis_type == "in_transit_loss":
                val = float(dis_val)
                for env in all_envs:
                    if day in env.in_transit:
                        env.in_transit[day] = env.in_transit[day] * (1.0 - val)

            # 3. QUALIDADE INFERIOR (o lote que chega HOJE já vem envelhecido -> vida útil restante = valor)
            elif dis_type == "shelf_life_drop":
                shelf_life_target = int(dis_val)

            # 4. FORÇAR PROCURA REAL
            elif dis_type == "demand_override":
                val = float(dis_val)
                for env in all_envs:
                    env.data.at[day, 'real_value'] = val

            # 5. REDUÇÃO DE CAPACIDADE DO ARMAZÉM (Persistente)
            elif dis_type == "capacity_drop":
                val = float(dis_val)
                for env in all_envs:
                    env.max_capacity = val

            # 6. ALTERAÇÃO DE CUSTO DO AMBIENTE DSAC (Persistente)
            elif dis_type == "cost_change":
                val = float(dis_val)
                for env in all_envs:
                    if dis_param == "holding_cost":
                        env.CUSTO_ARMAZEM_POR_M3 = val
                    elif dis_param == "waste_penalty":
                        env.waste_penalty_factor = val

        # O agente decide com o estado já atualizado pelas disrupções do dia
        if day_disruptions:
            state = env_test._get_state()

        lots_before = {env: {b.get('lot_id') for b in env.active_batches} for env in all_envs}

        # --- MIN-MAX BASELINE ---
        stock_hoje_minmax = sum(env_minmax.stock_profile) + env_minmax.in_transit.get(env_minmax.current_step, 0)
        vendas_hoje_minmax = env_minmax.data.iloc[env_minmax.current_step]['real_value']
        stock_amanha_minmax = max(0, stock_hoje_minmax - vendas_hoje_minmax) + env_minmax.in_transit.get(env_minmax.current_step + 1, 0)

        action_minmax = 0
        if stock_amanha_minmax <= s_min:
            action_minmax = max(0, S_max - stock_amanha_minmax)
            action_minmax = min(action_minmax, max_capacity)

        stock_inicial_hoje_minmax = sum(env_minmax.stock_profile)
        _, _, _, info_minmax = env_minmax.step(action_minmax)
        cum_profit_minmax += info_minmax['profit']
        profits_minmax.append(cum_profit_minmax)

        # --- TIME SUPPLY BASELINE ---
        horizon_days = 7
        forecast_sum = 0
        count = 0
        for i in range(1, horizon_days + 1):
            idx = env_timesupply.current_step + i
            if idx <= env_timesupply.max_steps:
                forecast_sum += env_timesupply.data.iloc[idx]['prediction']
                count += 1
        if count > 0:
            avg_forecast = forecast_sum / count
        else:
            avg_forecast = env_timesupply.data.iloc[env_timesupply.current_step]['prediction']

        min_time_supply_days = 3
        max_time_supply_days = 7
        order_point_ts = avg_forecast * min_time_supply_days
        order_up_to_ts = avg_forecast * max_time_supply_days

        stock_hoje_ts = sum(env_timesupply.stock_profile) + env_timesupply.in_transit.get(env_timesupply.current_step, 0)
        vendas_hoje_ts = env_timesupply.data.iloc[env_timesupply.current_step]['real_value']
        stock_amanha_ts = max(0, stock_hoje_ts - vendas_hoje_ts) + env_timesupply.in_transit.get(env_timesupply.current_step + 1, 0)

        action_timesupply = 0
        if stock_amanha_ts <= order_point_ts:
            action_timesupply = max(0.0, order_up_to_ts - stock_amanha_ts)
            action_timesupply = min(action_timesupply, max_capacity)
        action_timesupply = int(round(action_timesupply))

        stock_inicial_hoje_ts = sum(env_timesupply.stock_profile)
        _, _, _, info_timesupply = env_timesupply.step(action_timesupply)

        # --- FLOATING POINT BASELINE ---
        split_index = int(len(env_floatingpoint.df) * train_split)
        global_idx = split_index + env_floatingpoint.current_step
        lookback = env_floatingpoint.df.iloc[max(0, global_idx - 14):global_idx]
        if len(lookback) > 0 and 'real_value' in lookback.columns:
            avg_historic_sales = lookback['real_value'].mean()
        else:
            avg_historic_sales = 15.0

        floating_min_days = 2
        floating_max_days = 5
        floating_min_stock = avg_historic_sales * floating_min_days
        floating_max_stock = avg_historic_sales * floating_max_days

        stock_hoje_fp = sum(env_floatingpoint.stock_profile) + env_floatingpoint.in_transit.get(env_floatingpoint.current_step, 0)
        vendas_hoje_fp = env_floatingpoint.data.iloc[env_floatingpoint.current_step]['real_value']
        stock_amanha_fp = max(0, stock_hoje_fp - vendas_hoje_fp) + env_floatingpoint.in_transit.get(env_floatingpoint.current_step + 1, 0)

        action_floatingpoint = 0
        if stock_amanha_fp <= floating_min_stock:
            action_floatingpoint = max(0.0, floating_max_stock - stock_amanha_fp)
            action_floatingpoint = min(action_floatingpoint, max_capacity)
        action_floatingpoint = int(round(action_floatingpoint))

        stock_inicial_hoje_fp = sum(env_floatingpoint.stock_profile)
        _, _, _, info_floatingpoint = env_floatingpoint.step(action_floatingpoint)

        # --- RL AGENT (DSAC Limitado, ação determinística) ---
        action_norm = agent.select_action(state, evaluate=True)
        physical_action, _ = _action_to_order(action_norm, max_order_limit)

        stock_inicial_hoje = sum(env_test.stock_profile)
        next_state, _, done_env, info = env_test.step(physical_action)
        done = done_env

        # Qualidade inferior: aplicada aos lotes que deram entrada hoje (depois do step, em todos os ambientes)
        if shelf_life_target is not None:
            for env in all_envs:
                _degrade_new_batches(env, lots_before[env], shelf_life_target)
            if not done:
                next_state = env_test._get_state()

        # Valores finais do dia (depois de eventuais disrupções pós-step)
        stock_final_hoje_minmax = sum(env_minmax.stock_profile)
        sales_mm = info_minmax['sales']
        spoilage_mm = info_minmax['spoilage']
        lost_sales_mm = max(0, info_minmax['real_demand'] - sales_mm)
        overcapacity_waste_mm = max(0, info_minmax['overflow_waste'] - spoilage_mm)

        log_stock_inicial_minmax.append(stock_inicial_hoje_minmax)
        log_stock_final_minmax.append(stock_final_hoje_minmax)
        log_vendas_minmax.append(sales_mm)
        log_vendas_perdidas_minmax.append(lost_sales_mm)
        log_apodrecimento_minmax.append(spoilage_mm)
        log_excesso_minmax.append(overcapacity_waste_mm)
        log_lucro_diario_minmax.append(info_minmax['profit'])

        cum_profit_timesupply += info_timesupply['profit']
        profits_timesupply.append(cum_profit_timesupply)
        log_stock_inicial_timesupply.append(stock_inicial_hoje_ts)
        log_stock_final_timesupply.append(sum(env_timesupply.stock_profile))
        log_vendas_timesupply.append(info_timesupply['sales'])
        log_lucro_diario_timesupply.append(info_timesupply['profit'])
        log_acoes_timesupply.append(action_timesupply)
        log_lucro_acumulado_timesupply.append(cum_profit_timesupply)

        cum_profit_floatingpoint += info_floatingpoint['profit']
        profits_floatingpoint.append(cum_profit_floatingpoint)
        log_stock_inicial_floatingpoint.append(stock_inicial_hoje_fp)
        log_stock_final_floatingpoint.append(sum(env_floatingpoint.stock_profile))
        log_vendas_floatingpoint.append(info_floatingpoint['sales'])
        log_lucro_diario_floatingpoint.append(info_floatingpoint['profit'])
        log_acoes_floatingpoint.append(action_floatingpoint)
        log_lucro_acumulado_floatingpoint.append(cum_profit_floatingpoint)

        log_msg = f"[Dia {day+1:03d}] Encomendas -> Agente: {info['order_placed']:03d} | MinMax: {info_minmax['order_placed']:03d} | TimeSupply: {action_timesupply:03d} | FloatingPoint: {action_floatingpoint:03d} || Lucro Agente: {info['profit']:.1f}€ | MinMax: {info_minmax['profit']:.1f}€"

        # Store experience (recompensa com o mesmo shaping de desperdício usado no treino)
        if env_train is not None:
            train_like_profit = (info['commercial_profit']
                                 - info['base_waste_cost'] * (env_train.waste_penalty_factor - 1.0)
                                 - float(env_test.stock_profile[3]) * env_train.latent_risk_factor)
            current_15d_buffer.append({
                'state': np.asarray(state, dtype=np.float32),
                'action': np.asarray(action_norm, dtype=np.float32),
                'reward': float(np.clip(train_like_profit / 100.0, -10.0, 10.0)),
                'next_state': np.asarray(next_state, dtype=np.float32),
                'is_terminal': done
            })

        # Global stats tracking
        cum_profit_agent += info['profit']
        profits_agent.append(cum_profit_agent)

        # Detail stats
        stock_final_hoje = sum(env_test.stock_profile)
        sales = info['sales']
        real_demand = info['real_demand']
        spoilage = info['spoilage']
        lost_sales = max(0, real_demand - sales)
        overcapacity_waste = max(0, info['overflow_waste'] - spoilage)

        log_dias.append(day + 1)
        log_procura_real.append(real_demand)
        log_preco_venda.append(info['price_today'])
        log_acoes_agente.append(info['order_placed'])
        log_acoes_minmax.append(info_minmax['order_placed'])
        log_stock_inicial_agente.append(stock_inicial_hoje)
        log_stock_final_agente.append(stock_final_hoje)
        log_vendas_agente.append(sales)
        log_vendas_perdidas_agente.append(lost_sales)
        log_apodrecimento_agente.append(spoilage)
        log_excesso_agente.append(overcapacity_waste)
        log_lucro_diario_agente.append(info['profit'])
        log_lucro_acumulado_agente.append(cum_profit_agent)
        log_lucro_acumulado_minmax.append(cum_profit_minmax)
        log_markdown_agente.append(info.get('markdown_sales', 0.0))
        log_rsl_min_agente.append(env_test.get_stock_remaining_shelf_life() if not done else 0)

        flag_stockout.append(1 if stock_final_hoje <= 0 else 0)
        flag_clientes_perdidos.append(1 if lost_sales > 0 else 0)
        flag_excesso_armazem.append(1 if overcapacity_waste > 0 else 0)
        flag_apodrecimento.append(1 if spoilage > 0 else 0)

        # Aplicar atrasos de entrega à encomenda efetuada hoje (que chegaria no dia + 1)
        delay_dis = [d for d in day_disruptions if d["type"] == "delay"]
        if delay_dis:
            delay_days = int(delay_dis[0]["value"])
            for env in all_envs:
                if (day + 1) in env.in_transit:
                    qty = env.in_transit.pop(day + 1)
                    target_day = day + 1 + delay_days
                    env.in_transit[target_day] = env.in_transit.get(target_day, 0.0) + qty
            if not done:
                next_state = env_test._get_state()

        # Restauro de atributos temporários modificados
        for env in all_envs:
            env.data.at[day, 'real_value'] = orig_real_demands[env]

        state = next_state
        dias_simulados += 1

        update_triggered = False
        # --- TRIGGER FINE-TUNING ---
        if env_train is not None and dias_simulados % update_interval_days == 0 and not done:
            update_triggered = True
            update_days.append(dias_simulados)

            train_state = continual_training_step(agent, current_15d_buffer, env_train, train_state, max_order_limit,
                                                  lr_actor=online_lr_actor, lr_critic=online_lr_critic,
                                                  batch_size=online_batch_size, device=device)
            current_15d_buffer = []
            agent.actor.eval()
            log_msg += f" -> [FINE-TUNING ACTIVE: Model v{len(update_days)}]"

        # Yield daily stats to display in UI dynamically
        yield {
            "status": "running",
            "day": day + 1,
            "msg": log_msg,
            "agent_profit_cum": cum_profit_agent,
            "minmax_profit_cum": cum_profit_minmax,
            "timesupply_profit_cum": cum_profit_timesupply,
            "floatingpoint_profit_cum": cum_profit_floatingpoint,
            "agent_action": info['order_placed'],
            "minmax_action": info_minmax['order_placed'],
            "timesupply_action": action_timesupply,
            "floatingpoint_action": action_floatingpoint,
            "real_demand": real_demand,
            "stock_level": stock_final_hoje,
            "order_placed": info['order_placed'],
            "spoilage": spoilage,
            "overflow_waste": overcapacity_waste,
            "minmax_stock_level": stock_final_hoje_minmax,
            "minmax_spoilage": spoilage_mm,
            "update_triggered": update_triggered,
            "version_count": len(update_days),
            "agent_sales": sales,
            "missed_sales": lost_sales
        }

    # Done - Save final model
    os.makedirs(save_dir, exist_ok=True)
    final_model_name = "dsac_limitado_online_final"
    final_model_path = os.path.join(save_dir, final_model_name)
    agent.save(save_dir, filename_prefix=final_model_name)
    final_meta = dict(meta)
    final_meta.update({"fine_tuned_online": True, "fine_tuning_cycles": len(update_days),
                       "fine_tuned_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    _save_meta(final_model_path, final_meta)

    # Generate static report Excel
    excel_report_path = os.path.join(save_dir, "Relatorio_Simulacao.xlsx")
    try:
        df_excel = pd.DataFrame({
            'Dia': log_dias,
            'Procura_Real': log_procura_real,
            'Preco_Venda_Dia': log_preco_venda,

            # AGENTE DSAC LIMITADO
            'Acao_Agente_DSAC': log_acoes_agente,
            'Stock_Inicial_Agente': log_stock_inicial_agente,
            'Stock_Final_Agente': log_stock_final_agente,
            'Vendas_Agente': log_vendas_agente,
            'Vendas_Markdown_Agente': log_markdown_agente,
            'Vendas_Perdidas_Agente': log_vendas_perdidas_agente,
            'Desperdicio_Validade_Agente': log_apodrecimento_agente,
            'Excesso_Armazem_Agente': log_excesso_agente,
            'Vida_Util_Min_Restante_Dias': log_rsl_min_agente,
            'Lucro_Diario_Agente': log_lucro_diario_agente,
            'Lucro_Acumulado_Agente': log_lucro_acumulado_agente,

            # BASELINE MIN-MAX
            'Acao_MinMax': log_acoes_minmax,
            'Stock_Inicial_MinMax': log_stock_inicial_minmax,
            'Stock_Final_MinMax': log_stock_final_minmax,
            'Vendas_MinMax': log_vendas_minmax,
            'Vendas_Perdidas_MinMax': log_vendas_perdidas_minmax,
            'Desperdicio_Validade_MinMax': log_apodrecimento_minmax,
            'Excesso_Armazem_MinMax': log_excesso_minmax,
            'Lucro_Diario_MinMax': log_lucro_diario_minmax,
            'Lucro_Acumulado_MinMax': log_lucro_acumulado_minmax,

            # TIME SUPPLY
            'Acao_TimeSupply': log_acoes_timesupply,
            'Stock_Inicial_TimeSupply': log_stock_inicial_timesupply,
            'Stock_Final_TimeSupply': log_stock_final_timesupply,
            'Vendas_TimeSupply': log_vendas_timesupply,
            'Lucro_Diario_TimeSupply': log_lucro_diario_timesupply,
            'Lucro_Acumulado_TimeSupply': log_lucro_acumulado_timesupply,

            # FLOATING POINT
            'Acao_FloatingPoint': log_acoes_floatingpoint,
            'Stock_Inicial_FloatingPoint': log_stock_inicial_floatingpoint,
            'Stock_Final_FloatingPoint': log_stock_final_floatingpoint,
            'Vendas_FloatingPoint': log_vendas_floatingpoint,
            'Lucro_Diario_FloatingPoint': log_lucro_diario_floatingpoint,
            'Lucro_Acumulado_FloatingPoint': log_lucro_acumulado_floatingpoint,

            'Flag_Stockout': flag_stockout,
            'Flag_Clientes_Perdidos': flag_clientes_perdidos,
            'Flag_Excesso_Armazem': flag_excesso_armazem,
            'Flag_Expirado': flag_apodrecimento
        })
        df_excel.to_excel(excel_report_path, index=False)
    except Exception as e:
        yield {"status": "warning", "msg": f"[WARNING] Failed to save Excel report: {e}"}

    yield {
        "status": "complete",
        "msg": f"[COMPLETED] Simulation finished after {dias_simulados} days.",
        "cum_profit_agent": cum_profit_agent,
        "cum_profit_minmax": cum_profit_minmax,
        "cum_profit_timesupply": cum_profit_timesupply,
        "cum_profit_floatingpoint": cum_profit_floatingpoint,
        "stockout_days": sum(flag_stockout),
        "spoilage_total": sum(log_apodrecimento_agente),
        "spoilage_total_minmax": sum(log_apodrecimento_minmax),
        "lost_sales_total": sum(log_vendas_perdidas_agente),
        "lost_sales_total_minmax": sum(log_vendas_perdidas_minmax),
        "overflow_waste_total": sum(log_excesso_agente),
        "overflow_waste_total_minmax": sum(log_excesso_minmax),
        "excel_report_path": excel_report_path,
        "final_model_path": final_model_path,
        "update_days": update_days,
        "log_dias": log_dias,
        "log_lucro_acumulado_agente": log_lucro_acumulado_agente,
        "log_lucro_acumulado_minmax": log_lucro_acumulado_minmax,
        "log_lucro_acumulado_timesupply": log_lucro_acumulado_timesupply,
        "log_lucro_acumulado_floatingpoint": log_lucro_acumulado_floatingpoint,
        "log_acoes_agente": log_acoes_agente,
        "log_acoes_minmax": log_acoes_minmax,
        "log_vendas_agente": log_vendas_agente,
        "log_vendas_minmax": log_vendas_minmax,
        "log_vendas_perdidas_agente": log_vendas_perdidas_agente,
        "log_vendas_perdidas_minmax": log_vendas_perdidas_minmax,
        "log_apodrecimento_agente": log_apodrecimento_agente,
        "log_apodrecimento_minmax": log_apodrecimento_minmax,
        "log_stock_final_agente": log_stock_final_agente,
        "log_stock_final_minmax": log_stock_final_minmax,
        "log_procura_real": log_procura_real
    }


# =====================================================================
# --- PREDICTION MODELS BACKEND FUNCTIONS ---
# =====================================================================

def train_mlp_forecaster_generator(lote, df_data, save_dir="models", epochs=1000, patience=100):
    """
    Treina o modelo de previsão MLP (Multi-Layer Perceptron) para o lote epoch-por-epoch.
    Suporta até 1000 epochs com early stopping (patience=15) baseado em validação temporal.
    Salva em save_dir/sales_mlp_{lote}.joblib
    """
    import os
    import joblib
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import mean_squared_error
    
    # Ordenar por data
    df_data = df_data.sort_values(by='date').reset_index(drop=True)
    df_data['date'] = pd.to_datetime(df_data['date'])
    
    # Criar features de lag e calendário
    df_data['day_of_week'] = df_data['date'].dt.dayofweek + 1
    df_data['month'] = df_data['date'].dt.month
    
    df_data['real_value_lag1'] = df_data['sales_quantity_kg'].shift(1)
    df_data['real_value_lag7'] = df_data['sales_quantity_kg'].shift(7)
    
    if 'price_per_kg' not in df_data.columns:
        df_data['price_per_kg'] = 2.0
        
    df_mlp = df_data.dropna(subset=['real_value_lag1', 'real_value_lag7', 'sales_quantity_kg']).reset_index(drop=True)
    
    if len(df_mlp) < 8:
        raise ValueError("Histórico de vendas insuficiente para lags (mínimo 8 dias totais necessários).")
        
    X = df_mlp[['real_value_lag1', 'real_value_lag7', 'price_per_kg', 'day_of_week', 'month']].values
    y = df_mlp['sales_quantity_kg'].values
    
    # Divisão temporal (90% treino, 10% validação)
    split_idx = int(len(X) * 0.9)
    if split_idx < 5:  # se o histórico for muito pequeno, valida no próprio treino
        X_train, y_train = X, y
        X_val, y_val = X, y
    else:
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32, 16), random_state=42)
    
    best_val_loss = float('inf')
    best_coefs = None
    best_intercepts = None
    patience_counter = 0
    
    # Treino incremental (partial_fit)
    for epoch in range(1, epochs + 1):
        mlp.partial_fit(X_train, y_train)
        
        train_loss = mlp.loss_
        y_val_pred = mlp.predict(X_val)
        val_loss = float(mean_squared_error(y_val, y_val_pred))
        
        pct = int((epoch / epochs) * 100)
        yield pct, f"Epoch {epoch}/{epochs} - Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}"
        
        # Monitorizar a melhor perda de validação
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_coefs = [c.copy() for c in mlp.coefs_]
            best_intercepts = [i.copy() for i in mlp.intercepts_]
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                yield pct, f"Early Stopping na época {epoch}! Val Loss não melhora há {patience} épocas. Melhor Val Loss: {best_val_loss:.6f}"
                # Restaurar o melhor estado
                if best_coefs is not None:
                    mlp.coefs_ = best_coefs
                    mlp.intercepts_ = best_intercepts
                break
    
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"sales_mlp_{lote}.joblib")
    joblib.dump(mlp, model_path)


def train_autoformer_forecaster_generator(lote, df_data, save_dir="models"):
    """
    Treina o modelo de previsão Autoformer (PyTorch) para o lote com feedback em tempo real.
    Salva em save_dir/sales_autoformer_{lote}.pt
    """
    import os
    import autoformer_forecaster
    
    # Ordenar por data
    df_data = df_data.sort_values(by='date').reset_index(drop=True)
    df_data['date'] = pd.to_datetime(df_data['date'])
    
    df_auto = df_data.copy()
    if 'date' in df_auto.columns:
        df_auto = df_auto.rename(columns={"date": "Data"})
    if 'sales_quantity_kg' in df_auto.columns:
        df_auto = df_auto.rename(columns={"sales_quantity_kg": "Valor"})
        
    # Consumir o gerador interno do Autoformer
    gen = autoformer_forecaster.train_model_generator(df_auto)
    for pct, log_msg, model_bytes in gen:
        if model_bytes is not None:
            # Ao alcançar 100%, salvar o arquivo de pesos
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f"sales_autoformer_{lote}.pt")
            with open(model_path, 'wb') as f:
                f.write(model_bytes)
        yield pct, log_msg


def run_forecast_inference(lote, model_type, df_history, horizon_days=30, save_dir="models"):
    """
    Executa a inferência autoregressiva (horizon_days = 15 ou 30) para o lote.
    Retorna uma lista de tuplos (date, value) com valores inteiros.
    """
    import os
    import joblib
    import datetime
    
    df_history = df_history.sort_values(by='date').reset_index(drop=True)
    df_history['date'] = pd.to_datetime(df_history['date'])
    
    start_date = df_history['date'].max() + datetime.timedelta(days=1)
    predictions = []
    
    avg_price = float(df_history['price_per_kg'].mean() if 'price_per_kg' in df_history.columns else 2.0)
    
    if model_type == 'autoformer':
        import autoformer_forecaster
        model_path = os.path.join(save_dir, f"sales_autoformer_{lote}.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo Autoformer para o lote '{lote}' não foi encontrado em '{model_path}'.")
            
        with open(model_path, 'rb') as f:
            model_bytes = f.read()
            
        running_history = list(df_history['sales_quantity_kg'].tail(30).values)
        if len(running_history) < 30:
            pad_size = 30 - len(running_history)
            if len(running_history) == 0:
                running_history = [10.0] * 30
            else:
                running_history = [running_history[0]] * pad_size + running_history
                
        preds_list = autoformer_forecaster.predict_horizon(model_bytes, running_history, avg_price, horizon_days, start_date=start_date)
        
        for step in range(horizon_days):
            current_date = start_date + datetime.timedelta(days=step)
            # Garantir valor inteiro arredondado
            val = int(round(max(0.0, float(preds_list[step]))))
            predictions.append((current_date, val))
            
    else: # mlp
        model_path = os.path.join(save_dir, f"sales_mlp_{lote}.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo MLP para o lote '{lote}' não foi encontrado em '{model_path}'.")
            
        mlp = joblib.load(model_path)
        
        running_history = list(df_history['sales_quantity_kg'].tail(7).values)
        if len(running_history) < 7:
            pad_size = 7 - len(running_history)
            if len(running_history) == 0:
                running_history = [10.0] * 7
            else:
                running_history = [running_history[0]] * pad_size + running_history
                
        for step in range(horizon_days):
            current_date = start_date + datetime.timedelta(days=step)
            day_of_week = current_date.weekday() + 1
            month = current_date.month
            
            lag1 = running_history[-1]
            lag7 = running_history[-7]
            
            X_pred = np.array([[lag1, lag7, avg_price, day_of_week, month]])
            y_pred = float(mlp.predict(X_pred)[0])
            # Garantir valor inteiro arredondado
            y_pred_int = int(round(max(0.0, y_pred)))
            
            predictions.append((current_date, y_pred_int))
            running_history.append(y_pred_int)
            
    return predictions


def populate_prediction_column(lote, model_type, df_data, save_dir="models"):
    """
    Popula ou adiciona a coluna 'prediction' no DataFrame do histórico de treino/teste do Buyer Agent,
    usando o modelo de previsão previamente treinado.
    """
    import os
    import joblib
    import datetime
    
    df_data = df_data.sort_values(by='date').reset_index(drop=True)
    df_data['date'] = pd.to_datetime(df_data['date'])
    
    df_res = df_data.copy()
    avg_price = float(df_res['price_per_kg'].mean() if 'price_per_kg' in df_res.columns else 2.0)
    
    predictions = []
    
    if model_type == 'autoformer':
        import autoformer_forecaster
        model_path = os.path.join(save_dir, f"sales_autoformer_{lote}.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo Autoformer para o lote '{lote}' não foi encontrado em '{model_path}'.")
            
        with open(model_path, 'rb') as f:
            model_bytes = f.read()
            
        for idx in range(len(df_res)):
            if idx < 30:
                predictions.append(int(round(float(df_res.loc[idx, 'sales_quantity_kg']))))
            else:
                running_history = list(df_res.loc[idx-30:idx-1, 'sales_quantity_kg'].values)
                current_date = df_res.loc[idx, 'date']
                pred = autoformer_forecaster.predict_horizon(model_bytes, running_history, avg_price, horizon_days=1, start_date=current_date)[0]
                predictions.append(int(round(max(0.0, float(pred)))))
                
    else: # mlp
        model_path = os.path.join(save_dir, f"sales_mlp_{lote}.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo MLP para o lote '{lote}' não foi encontrado em '{model_path}'.")
            
        mlp = joblib.load(model_path)
        
        for idx in range(len(df_res)):
            if idx < 7:
                predictions.append(int(round(float(df_res.loc[idx, 'sales_quantity_kg']))))
            else:
                lag1 = float(df_res.loc[idx-1, 'sales_quantity_kg'])
                lag7 = float(df_res.loc[idx-7, 'sales_quantity_kg'])
                day_of_week = df_res.loc[idx, 'date'].weekday() + 1
                month = df_res.loc[idx, 'date'].month
                price_val = float(df_res.loc[idx, 'price_per_kg'] if 'price_per_kg' in df_res.columns else avg_price)
                
                X_pred = np.array([[lag1, lag7, price_val, day_of_week, month]])
                pred = float(mlp.predict(X_pred)[0])
                predictions.append(int(round(max(0.0, pred))))
                
    df_res['prediction'] = predictions
    return df_res

