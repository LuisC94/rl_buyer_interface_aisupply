import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
import math
import os
from loguru import logger
import numba

class EnvRunningStat:
    """ Estatístico Dinâmico Independente para o MORL (Welford's Algorithm) """
    def __init__(self, shape=()):
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
    def push(self, x):
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.S = 0.0
        else:
            if isinstance(self.mean, np.ndarray):
                old_mean = self.mean.copy()
            else:
                old_mean = self.mean
                
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
    @property
    def variance(self):
        return self.S / (self.n - 1) if self.n > 1 else np.square(self.mean)
    @property
    def std(self):
        return np.sqrt(self.variance)

@numba.njit(nogil=True, fastmath=True)
def advance_batch_one_day_jit(dureza, brix, mold, E_int, age, quantity, T_c, RH_pct, E_ext_ppm, preset):
    # R = 8.314, dt = 0.05, steps = 20
    R = 8.314
    dt = 0.05
    steps = 20
    
    T_K = T_c + 273.15
    Tref_K = preset[0] + 273.15
    
    kT_firm = preset[2] * math.exp((-preset[1] / R) * (1.0 / T_K - 1.0 / Tref_K))
    
    RH_ref = preset[5]
    RH_deficit = max(0.0, (RH_ref - RH_pct) / 100.0)
    kRH = 1.0 + preset[4] * RH_deficit
    
    Ea_E_J = preset[18]
    Eref_prod = preset[13]
    E_decay = preset[17]
    E_t0 = preset[14]
    E_g = preset[15]
    E_auto = preset[16]
    E_decay_val = preset[17]
    E_ext_shift = preset[19]
    
    t0_eff = E_t0 - E_ext_shift * math.log1p(max(0.0, E_ext_ppm))
    prod_T = math.exp((-Ea_E_J / R) * (1.0 / T_K - 1.0 / Tref_K))
    
    dureza_min = preset[6]
    alpha_E = preset[3]
    
    brix_min = preset[7]
    brix_max = preset[8]
    r0 = preset[9]
    alpha_bE = 0.25
    bRH = max(0.0, (RH_ref - RH_pct) / 100.0)
    rRH = 1.0 - 0.6 * bRH
    rT = prod_T
    
    RH_mold_thr = preset[20]
    mold_rate_ref = preset[21]
    mold_sens_RH = preset[22]
    mold_max_penalty = preset[23]
    Ea_mold_J = preset[24]
    
    mold_T = math.exp((-Ea_mold_J / R) * (1.0 / T_K - 1.0 / Tref_K))
    RH_excess = max(0.0, (RH_pct - RH_mold_thr) / 100.0)
    RH_factor = 1.0 - math.exp(-mold_sens_RH * RH_excess)
    
    for _ in range(steps):
        current_age = age + (_ * dt)
        ramp = 1.0 / (1.0 + math.exp(-E_g * (current_age - t0_eff)))
        prod = Eref_prod * prod_T * ramp
        dE = (prod * (1.0 + E_auto * E_int) - E_decay_val * E_int) * dt
        E_int = max(0.0, E_int + dE)
        
        E_total = E_ext_ppm + E_int
        
        kE = 1.0 + alpha_E * E_total
        k = kT_firm * kRH * kE
        dD = (-k * (dureza - dureza_min)) * dt
        dureza = max(dureza_min, dureza + dD)
        
        r = r0 * rT * rRH * (1.0 + alpha_bE * E_total)
        x = max(0.0, brix - brix_min)
        K = max(1e-6, brix_max - brix_min)
        db = (r * x * (1.0 - x / K)) * dt
        brix = min(brix_max, max(brix_min, brix + db))
        
        rate = mold_rate_ref * mold_T * RH_factor
        dm = (rate * (1.0 - mold)) * dt
        mold = min(1.0, max(0.0, mold + dm))
        
    firm_score = 1.0 / (1.0 + math.exp(-0.35 * (dureza - preset[10])))
    brix_score = math.exp(-((brix - preset[11])**2) / 2.0)
    quality_base = 100.0 * (0.65 * firm_score + 0.35 * brix_score)
    
    mold_penalty = preset[23] * mold
    quality = quality_base * (1.0 - mold_penalty)
    
    return dureza, brix, mold, E_int, age + 1.0, quality

@numba.njit(nogil=True, fastmath=True)
def project_batch_rsl_jit(dureza, brix, mold, E_int, age, quantity, quality, T_c, RH_pct, E_ext_ppm, preset):
    days_remaining = 0
    max_projection_days = 60
    
    cur_dureza = dureza
    cur_brix = brix
    cur_mold = mold
    cur_E_int = E_int
    cur_age = age
    cur_quality = quality
    
    while days_remaining < max_projection_days:
        if cur_quality < 30.0:
            break
        cur_dureza, cur_brix, cur_mold, cur_E_int, cur_age, cur_quality = advance_batch_one_day_jit(
            cur_dureza, cur_brix, cur_mold, cur_E_int, cur_age, quantity, T_c, RH_pct, E_ext_ppm, preset
        )
        days_remaining += 1
        
    return days_remaining

class StockEnvironment:
    """
    OpenAI Gym-style Environment for Supply Chain Reinforcement Learning.
    Constrained version: Limits active ordering actions to the maximum historical sales in training split.
    Uses FruitModel2 biological decay presets and active batch FEFO.
    """
    
    # Presets biológicos por fruta retirados de FruitModel2.ipynb
    PRESETS = {
        "kiwi_hayward": {
            "label": "Kiwi (Hayward)",
            "Tref_C": 5.0, "Ea_J": 60000.0, "k_firm_ref": 0.15, "alpha_E": 1.8,
            "beta_RH": 1.2, "RH_ref": 90.0,
            "dureza_min": 3.0, "dureza_0_default": 45.0,
            "brix_min": 11.0, "brix_max": 17.0, "brix_g": 0.35, "brix_0_default": 11.0,
            "qual_firm_threshold": 8.0, "qual_brix_target": 15.0,
            "E0_int": 0.02, "Eref_prod": 0.12, "E_t0": 10.0, "E_g": 0.9, "E_auto": 0.35,
            "E_decay": 0.7, "Ea_E_J": 52000.0, "E_ext_shift": 2.0,
            "RH_mold_thr": 95.0, "mold_rate_ref": 0.05, "mold_sens_RH": 9.0,
            "mold_max_penalty": 0.65, "Ea_mold_J": 43000.0
        },
        "maca_golden": {
            "label": "Maçã (Golden)",
            "Tref_C": 5.0, "Ea_J": 50000.0, "k_firm_ref": 0.18, "alpha_E": 0.8,
            "beta_RH": 0.8, "RH_ref": 90.0,
            "dureza_min": 12.0, "dureza_0_default": 72.0,
            "brix_min": 11.5, "brix_max": 15.5, "brix_g": 0.18, "brix_0_default": 12.0,
            "qual_firm_threshold": 35.0, "qual_brix_target": 13.5,
            "E0_int": 0.01, "Eref_prod": 0.1, "E_t0": 18.0, "E_g": 0.6, "E_auto": 0.35,
            "E_decay": 0.55, "Ea_E_J": 52000.0, "E_ext_shift": 1.8,
            "RH_mold_thr": 95.0, "mold_rate_ref": 0.04, "mold_sens_RH": 8.0,
            "mold_max_penalty": 0.60, "Ea_mold_J": 42000.0
        },
        "maca_reineta": {
            "label": "Maçã (Reineta)",
            "Tref_C": 5.0, "Ea_J": 52000.0, "k_firm_ref": 0.18, "alpha_E": 1.1,
            "beta_RH": 1.0, "RH_ref": 90.0,
            "dureza_min": 10.0, "dureza_0_default": 65.0,
            "brix_min": 11.0, "brix_max": 14.0, "brix_g": 0.16, "brix_0_default": 11.5,
            "qual_firm_threshold": 30.0, "qual_brix_target": 12.5,
            "E0_int": 0.01, "Eref_prod": 0.13, "E_t0": 14.0, "E_g": 0.7, "E_auto": 0.40,
            "E_decay": 0.6, "Ea_E_J": 52000.0, "E_ext_shift": 2.0,
            "RH_mold_thr": 95.0, "mold_rate_ref": 0.05, "mold_sens_RH": 9.0,
            "mold_max_penalty": 0.65, "Ea_mold_J": 43000.0
        },
        "maca_gala": {
            "label": "Maçã (Gala)",
            "Tref_C": 5.0, "Ea_J": 48000.0, "k_firm_ref": 0.20, "alpha_E": 1.3,
            "beta_RH": 0.9, "RH_ref": 90.0,
            "dureza_min": 9.0, "dureza_0_default": 60.0,
            "brix_min": 12.5, "brix_max": 17.0, "brix_g": 0.25, "brix_0_default": 13.0,
            "qual_firm_threshold": 28.0, "qual_brix_target": 14.5,
            "E0_int": 0.015, "Eref_prod": 0.18, "E_t0": 10.0, "E_g": 0.9, "E_auto": 0.5,
            "E_decay": 0.65, "Ea_E_J": 52000.0, "E_ext_shift": 2.2,
            "RH_mold_thr": 95.0, "mold_rate_ref": 0.05, "mold_sens_RH": 9.0,
            "mold_max_penalty": 0.65, "Ea_mold_J": 43000.0
        },
        "maca_fuji": {
            "label": "Maçã (Fuji)",
            "Tref_C": 5.0, "Ea_J": 47000.0, "k_firm_ref": 0.18, "alpha_E": 0.6,
            "beta_RH": 0.7, "RH_ref": 90.0,
            "dureza_min": 15.0, "dureza_0_default": 80.0,
            "brix_min": 13.0, "brix_max": 19.0, "brix_g": 0.15, "brix_0_default": 14.0,
            "qual_firm_threshold": 40.0, "qual_brix_target": 16.0,
            "E0_int": 0.008, "Eref_prod": 0.06, "E_t0": 25.0, "E_g": 0.5, "E_auto": 0.25,
            "E_decay": 0.45, "Ea_E_J": 52000.0, "E_ext_shift": 1.4,
            "RH_mold_thr": 95.0, "mold_rate_ref": 0.035, "mold_sens_RH": 8.0,
            "mold_max_penalty": 0.55, "Ea_mold_J": 42000.0
        }
    }

    def __init__(self, excel_path, is_training=True, train_split=0.6, max_capacity=1000, shared_stats=None, completely_unconstrained=False, include_future_forecasts=False, enable_lifecycle=True, use_web_service=False, web_service_url="http://localhost:8003", truck_capacity_m3=80.0, waste_penalty_factor=None, latent_risk_factor=None):
        self.is_training = is_training
        self.completely_unconstrained = completely_unconstrained
        self.include_future_forecasts = include_future_forecasts
        self.enable_lifecycle = enable_lifecycle
        
        # Penalizações Universais de Desperdício e Risco Biológico (Spoilage):
        # Durante o treino: padrão 5.0x e 0.30 para forçar todos os modelos a combater o desperdício
        # Durante o teste: padrão 1.0x e 0.00 para avaliação do lucro comercial líquido em euros
        if waste_penalty_factor is not None:
            self.waste_penalty_factor = float(waste_penalty_factor)
        else:
            self.waste_penalty_factor = 5.0 if is_training else 1.0

        if latent_risk_factor is not None:
            self.latent_risk_factor = float(latent_risk_factor)
        else:
            self.latent_risk_factor = 0.30 if is_training else 0.0
        
        # Padrão: SEMPRE motor local JIT em memória de ultra-alta velocidade
        # (O Web Service só é acionado se use_web_service=True for passado explicitamente)
        self.use_web_service = bool(use_web_service)
            
        self.web_service_url = web_service_url
        self.lifecycle_client = None
        self._next_lot_id = 1
        
        if self.use_web_service:
            try:
                from lifecycle_client import LifecycleServiceClient
                self.lifecycle_client = LifecycleServiceClient(base_url=self.web_service_url)
                logger.info(f"StockEnvironment: Ligado ao Lifecycle Web Service em {self.web_service_url}")
            except Exception as e:
                logger.warning(f"StockEnvironment: Não foi possível carregar LifecycleServiceClient ({e}). Usando motor local JIT.")
                self.lifecycle_client = None
        
        # 1. Load Data
        self.df = pd.read_excel(excel_path)
        
        # 2. Train/Test Split
        split_index = int(len(self.df) * train_split)
        
        # Determinar dinamicamente o limite de encomenda
        if self.completely_unconstrained:
            self.max_order_limit = float(max_capacity)
        else:
            train_data_df = self.df.iloc[:split_index] if split_index > 0 else self.df
            if 'real_value' in train_data_df.columns and len(train_data_df) > 0 and not np.isnan(train_data_df['real_value'].max()):
                self.max_order_limit = float(train_data_df['real_value'].max())
            else:
                self.max_order_limit = float(self.df['real_value'].max()) if ('real_value' in self.df.columns and not np.isnan(self.df['real_value'].max())) else float(max_capacity)
            
        if is_training:
            self.data = self.df.iloc[:split_index].reset_index(drop=True)
            logger.info(f"Environment Constrained initialized in TRAINING Mode: {len(self.data)} days. Max Order Limit: {self.max_order_limit}")
        else:
            self.data = self.df.iloc[split_index:].reset_index(drop=True)
            logger.info(f"Environment Constrained initialized in TESTING Mode: {len(self.data)} days. Max Order Limit: {self.max_order_limit}")
            
        self.max_steps = len(self.data) - 1
        self.current_step = 0
        
        # Obter o preset biológico diretamente da coluna 'fruit_type' do dataset
        self.fruit_key = str(self.df['fruit_type'].iloc[0]).strip().lower()
        if self.fruit_key not in self.PRESETS:
            matched = False
            for k in self.PRESETS.keys():
                if k in self.fruit_key or self.fruit_key in k:
                    self.fruit_key = k
                    matched = True
                    break
            if not matched:
                self.fruit_key = "kiwi_hayward"
            
        # 3. Physics & Fixed Constraints
        self.max_capacity = max_capacity
        self.stock_inicial = 100.0
        
        # Extrair Volume do Produto (m3) e Preço Médio (se a coluna não existir, usa Fallbacks)
        self.product_volume_m3 = self.data['volume'].iloc[0] if 'volume' in self.data.columns else 0.002
        self.avg_price = self.data['price'].mean() if 'price' in self.data.columns else 2.0
        
        # --- LOGÍSTICA VOLUMÉTRICA UNIVERSAL E FROTAS (ORACLE RETAIL AIP/OTM/RMS) ---
        self.CUSTO_ARMAZEM_POR_M3 = 20.00         # Custo de 1m³ por dia no armazém
        self.CUSTO_TRANSPORTE_POR_M3 = 50.00      # Custo base legado por m3
        self.TRUCK_CAPACITY_M3 = float(truck_capacity_m3) # Capacidade física volumétrica do camião padrão
        self.TRUCK_CAPACITY = max(1, int(self.TRUCK_CAPACITY_M3 / max(1e-6, self.product_volume_m3)))
        
        # Parâmetros de Frotas e Acondicionamento (Carrinha, Intermédio, TIR)
        self.boxes_per_pallet = int(self.data['boxes_per_pallet'].iloc[0]) if ('boxes_per_pallet' in self.data.columns and not np.isnan(self.data['boxes_per_pallet'].iloc[0])) else 50
        self.cogs_ratio = 0.60
        self.loose_box_fee = 0.20             # Taxa calibrada de manuseamento manual por caixa solta (0.20€)
        self.dead_freight_factor = 0.10       # Fator calibrado de dead freight (0.10 em vez de 0.50)
        self.markdown_discount = 0.40         # Desconto aplicado a stock em fase crítica (G3)
        self.wh_congestion_threshold = 0.80   # Limiar de 80% de capacidade do armazém para penalização quadrática
        self.consecutive_oos_days = 0         # Contador de dias seguidos em rutura de stock
        
        self.stock_profile = [0.0, 0.0, 0.0, 0.0] # [G0, G1, G2, G3]
        self.active_batches = []                  # Lotes físicos ativos: [{'quantity', 'dureza', 'brix', 'mold', 'E_int', 'age', 'quality'}]
        
        # Pre-compute preset array for Numba JIT acceleration
        p = self.PRESETS[self.fruit_key]
        self.preset_array = np.array([
            float(p["Tref_C"]),                 # 0
            float(p["Ea_J"]),                   # 1
            float(p["k_firm_ref"]),             # 2
            float(p["alpha_E"]),                # 3
            float(p["beta_RH"]),                # 4
            float(p["RH_ref"]),                 # 5
            float(p["dureza_min"]),             # 6
            float(p["brix_min"]),               # 7
            float(p["brix_max"]),               # 8
            float(p["brix_g"]),                 # 9
            float(p["qual_firm_threshold"]),    # 10
            float(p["qual_brix_target"]),       # 11
            float(p.get("E0_int", 0.01)),       # 12
            float(p.get("Eref_prod", 0.12)),    # 13
            float(p.get("E_t0", 10.0)),         # 14
            float(p.get("E_g", 0.9)),           # 15
            float(p.get("E_auto", 0.35)),       # 16
            float(p.get("E_decay", 0.7)),       # 17
            float(p.get("Ea_E_J", 52000.0)),    # 18
            float(p.get("E_ext_shift", 2.0)),   # 19
            float(p["RH_mold_thr"]),            # 20
            float(p["mold_rate_ref"]),          # 21
            float(p["mold_sens_RH"]),           # 22
            float(p["mold_max_penalty"]),       # 23
            float(p["Ea_mold_J"])               # 24
        ], dtype=np.float64)
        
        self.in_transit = {}                      # Dictionary to track {day_of_arrival: quantity}
        
        # --- GLOBAL/LOCAL STATISTICS (Robust Z-Score) ---
        if shared_stats is not None:
            self.stat_profit = shared_stats['econ']
        else:
            self.stat_profit = EnvRunningStat()
            
        self.eps = 1e-8
        self.clip_val = 10.0

        # Scaler for Neural Network standard formatting [-1, 1] ou [0, 1]
        self.scaler = MinMaxScaler()
        
        min_array = [0.0]*5 + [0.0]*4
        max_array = [float(self.max_capacity)]*5 + [100.0]*4
        
        self.scaler.fit(np.array([
            min_array,
            max_array
        ]))

    def _get_current_date_str(self, step=None):
        target_step = self.current_step if step is None else step
        if 0 <= target_step < len(self.data):
            row = self.data.iloc[target_step]
            if 'day' in row and pd.notnull(row['day']):
                return pd.to_datetime(row['day']).strftime("%Y-%m-%d")
            elif 'date' in row and pd.notnull(row['date']):
                return pd.to_datetime(row['date']).strftime("%Y-%m-%d")
        return f"2025-01-{(target_step % 28) + 1:02d}"

    def reset(self, start_step=None):
        """ Resets the world to a starting step (defaults to Day 0) """
        if start_step is not None:
            self.current_step = start_step
        else:
            self.current_step = 0
        p = self.PRESETS[self.fruit_key]
        self._next_lot_id = 1
        current_date_str = self._get_current_date_str()
        
        # Inicializa o armazém com o lote inicial
        self.active_batches = [{
            'lot_id': f"LOT-{self._next_lot_id}",
            'harvest_date': current_date_str,
            'initial_quantity': self.stock_inicial,
            'quantity': self.stock_inicial,
            'initial_dureza': float(p["dureza_0_default"]),
            'dureza': float(p["dureza_0_default"]),
            'initial_brix': float(p["brix_0_default"]),
            'brix': float(p["brix_0_default"]),
            'mold': 0.0,
            'E_int': float(p.get("E0_int", 0.01)),
            'age': 0.0,
            'quality': 100.0,
            'daily_readings': []
        }]
        self._next_lot_id += 1
        
        self.in_transit = {}
        self.last_order_qty = 0.0
        self.consecutive_oos_days = 0
        self._update_stock_profile_from_batches()
        return self._get_state()

    def advance_batch_one_day(self, batch, T_c, RH_pct, E_ext_ppm):
        """ Avança a maturação biológica de um lote por 1 dia usando dt = 0.05 """
        dureza, brix, mold, E_int, age, quality = advance_batch_one_day_jit(
            float(batch['dureza']),
            float(batch['brix']),
            float(batch['mold']),
            float(batch.get('E_int', self.preset_array[12])),
            float(batch.get('age', 0.0)),
            float(batch['quantity']),
            float(T_c),
            float(RH_pct),
            float(E_ext_ppm),
            self.preset_array
        )
        
        history = list(batch.get('daily_readings', []))
        history.append({
            "date": self._get_current_date_str(),
            "temperature_celsius": float(T_c),
            "humidity_percent": float(RH_pct),
            "ethylene_ppm": float(E_ext_ppm),
            "source": "SENSOR_RECORDING"
        })
        
        return {
            'lot_id': batch.get('lot_id', 'LOT-1'),
            'harvest_date': batch.get('harvest_date', self._get_current_date_str()),
            'initial_quantity': batch.get('initial_quantity', batch['quantity']),
            'quantity': batch['quantity'],
            'initial_dureza': batch.get('initial_dureza', dureza),
            'dureza': dureza,
            'initial_brix': batch.get('initial_brix', brix),
            'brix': brix,
            'mold': mold,
            'E_int': E_int,
            'age': age,
            'quality': quality,
            'daily_readings': history
        }

    def project_batch_rsl(self, batch):
        """ Projeta os dias restantes até que a qualidade caia abaixo de 30.0 """
        if not self.enable_lifecycle:
            return 60
        row = self.data.iloc[self.current_step]
        T_c = float(row['temperature'] if 'temperature' in row else 22.0)
        RH_pct = float(row['humidity'] if 'humidity' in row else 70.0)
        E_ext_ppm = float(row['ethylene'] if 'ethylene' in row else 0.8)
        
        # 1. Se estiver configurado para usar o Web Service REST em inferência:
        if self.use_web_service and self.lifecycle_client is not None:
            lot_id = str(batch.get('lot_id', 'LOT-1'))
            harvest_date = str(batch.get('harvest_date', self._get_current_date_str()))
            p = self.PRESETS[self.fruit_key]
            initial_brix = float(batch.get('initial_brix', p["brix_0_default"]))
            initial_firmness = float(batch.get('initial_dureza', p["dureza_0_default"]))
            daily_readings = list(batch.get('daily_readings', []))
            
            if not daily_readings:
                daily_readings = [{
                    "date": self._get_current_date_str(),
                    "temperature_celsius": T_c,
                    "humidity_percent": RH_pct,
                    "ethylene_ppm": E_ext_ppm,
                    "source": "SENSOR_RECORDING"
                }]
                
            res = self.lifecycle_client.query_lot_lifecycle(
                lot_id=lot_id,
                fruit_type=self.fruit_key,
                culture_name=self.fruit_key,
                harvest_date=harvest_date,
                initial_quantity_kg=float(batch.get('initial_quantity', batch['quantity'])),
                current_stock_kg=float(batch['quantity']),
                initial_brix=initial_brix,
                initial_firmness=initial_firmness,
                daily_readings=daily_readings
            )
            if res is not None and "remaining_lifetime" in res:
                self._logged_service_down = False
                return float(res["remaining_lifetime"])
            else:
                if not getattr(self, '_logged_service_down', False):
                    logger.warning(f"[AVISO] O Lifecycle Web Service está em baixo ou inacessível em '{self.web_service_url}'. A usar fallback para o motor local JIT.")
                    self._logged_service_down = True
        elif self.use_web_service and self.lifecycle_client is None:
            if not getattr(self, '_logged_service_down', False):
                logger.warning(f"[AVISO] O Lifecycle Web Service está em baixo ou inacessível em '{self.web_service_url}'. A usar fallback para o motor local JIT.")
                self._logged_service_down = True
        
        # 2. Fallback / Modo Treino nativo em memória ultra-rápido (JIT):
        days_remaining = project_batch_rsl_jit(
            float(batch['dureza']),
            float(batch['brix']),
            float(batch['mold']),
            float(batch.get('E_int', self.preset_array[12])),
            float(batch.get('age', 0.0)),
            float(batch['quantity']),
            float(batch.get('quality', 100.0)),
            T_c,
            RH_pct,
            E_ext_ppm,
            self.preset_array
        )
        return days_remaining

    def get_stock_remaining_shelf_life(self):
        """ Retorna o menor RSL estimado entre os lotes ativos em armazém """
        valid_batches = [b for b in self.active_batches if b['quantity'] > 0]
        if not valid_batches:
            return 0
        rsls = [self.project_batch_rsl(b) for b in valid_batches]
        return min(rsls) if rsls else 0

    def get_min_required_order_shelf_life(self, order_quantity):
        """ Retorna o shelf-life mínimo requerido para a nova encomenda sob FEFO """
        if order_quantity <= 0:
            return 0
            
        total_current_stock = sum(b['quantity'] for b in self.active_batches if b['quantity'] > 0)
        target_qty = total_current_stock + order_quantity
        
        accumulated_demand = 0.0
        days_ahead = 0
        step_idx = self.current_step + 1
        
        while accumulated_demand < target_qty:
            if step_idx <= self.max_steps:
                pred_demand = self.data.iloc[step_idx]['prediction']
            else:
                pred_demand = self.data.iloc[-1]['prediction']
                
            accumulated_demand += pred_demand
            days_ahead += 1
            step_idx += 1
            
            if days_ahead > 60:
                break
                
        return days_ahead

    def _update_stock_profile_from_batches(self):
        """ Atualiza self.stock_profile (G0-G3) somando os lotes pela percentagem do tempo de vida restante """
        if not self.enable_lifecycle:
            total_stock = sum(b['quantity'] for b in self.active_batches if b['quantity'] > 0)
            self.stock_profile = [total_stock, 0.0, 0.0, 0.0]
            return
            
        self.stock_profile = [0.0, 0.0, 0.0, 0.0]
        
        # Determinar o RSL máximo de um lote fresco sob as condições climáticas de hoje
        p = self.PRESETS[self.fruit_key]
        ref_batch = {
            'quantity': 1.0,
            'dureza': float(p["dureza_0_default"]),
            'brix': float(p["brix_0_default"]),
            'mold': 0.0,
            'E_int': float(p.get("E0_int", 0.01)),
            'age': 0.0,
            'quality': 100.0
        }
        max_rsl = self.project_batch_rsl(ref_batch)
        if max_rsl <= 0:
            max_rsl = 1  # Evita divisão por zero
            
        for b in self.active_batches:
            if b['quantity'] <= 0:
                continue
            rsl = self.project_batch_rsl(b)
            ratio = rsl / max_rsl
            
            # Divisão por faixas percentuais:
            # G0: >= 75% da vida útil restante (muito fresco)
            # G1: [50%, 75%[ da vida útil restante
            # G2: [25%, 50%[ da vida útil restante
            # G3: < 25% da vida útil restante (zona crítica)
            if ratio >= 0.75:
                self.stock_profile[0] += b['quantity']
            elif ratio >= 0.50:
                self.stock_profile[1] += b['quantity']
            elif ratio >= 0.25:
                self.stock_profile[2] += b['quantity']
            else:
                self.stock_profile[3] += b['quantity']

    def _get_state(self):
        """ Builds the Observation Vector (What the Actor SEES). """
        row = self.data.iloc[self.current_step]
        prediction_today = row['prediction']
        price_today = row['price'] if 'price' in row else 10.0
        
        if self.current_step < self.max_steps:
            prediction_tomorrow = self.data.iloc[self.current_step + 1]['prediction']
        else:
            prediction_tomorrow = prediction_today
            
        total_in_transit = sum(self.in_transit.values())
        
        # --- HISTORICAL DEMAND ---
        if self.current_step >= 1:
            real_t_minus_1 = self.data.iloc[self.current_step - 1]['real_value']
        else:
            real_t_minus_1 = prediction_today 
            
        if self.current_step >= 2:
            real_t_minus_2 = self.data.iloc[self.current_step - 2]['real_value']
        else:
            real_t_minus_2 = real_t_minus_1
            
        # --- CALENDAR MATH ---
        day_of_year = (self.current_step % 365) + 1  
        day_of_week = (self.current_step % 7) + 1    
        month = min(12, int((day_of_year / 30.416) + 1)) 
        
        sin_day = math.sin(2 * math.pi * day_of_week / 7.0)
        cos_day = math.cos(2 * math.pi * day_of_week / 7.0)
        
        sin_month = math.sin(2 * math.pi * month / 12.0)
        cos_month = math.cos(2 * math.pi * month / 12.0)

        # --- Z-SCORE PRICE ---
        window_prices = []
        for i in range(15):
            idx = max(0, self.current_step - i)
            p = self.data.iloc[idx]['price'] if 'price' in self.data.columns else 10.0
            window_prices.append(p)
            
        media_15dias = np.mean(window_prices)
        std_15dias = np.std(window_prices)
        preco_relativo = (price_today - media_15dias) / (std_15dias + 1e-8)

        # --- SUPER FEATURES ---
        stock_total_atual = sum(self.stock_profile)
        
        cobertura_dias = stock_total_atual / (prediction_today + 1e-8)
        cobertura_norm = np.clip(cobertura_dias, 0, 7) / 7.0
        
        urgencia_norm = self.stock_profile[3] / (stock_total_atual + 1e-8)
        
        if self.current_step >= 1:
            prediction_yesterday = self.data.iloc[self.current_step - 1]['prediction']
        else:
            prediction_yesterday = prediction_today
            
        erro_previsao = (real_t_minus_1 - prediction_yesterday) / (prediction_yesterday + 1e-8)
        erro_norm = np.clip(erro_previsao, -1.0, 1.0)

        via1_absolutas = [
            self.stock_profile[0],
            self.stock_profile[1],
            self.stock_profile[2],
            self.stock_profile[3],
            total_in_transit, 
            prediction_today, 
            prediction_tomorrow,
            real_t_minus_1,
            real_t_minus_2
        ]
        
        scaled_via1 = self.scaler.transform([via1_absolutas])[0]
        preco_relativo_safe = np.clip(preco_relativo, -3.0, 3.0)

        via2_bypass = [
            preco_relativo_safe,
            sin_day,
            cos_day,
            sin_month,
            cos_month,
            cobertura_norm,
            urgencia_norm,
            erro_norm
        ]
        
        final_state = np.concatenate([scaled_via1, via2_bypass])
        
        # --- 6 NEW FEATURES (DIMENSION 23) ---
        cum_5d = 0.0
        cum_10d = 0.0
        cum_15d = 0.0
        cum_20d = 0.0
        for i in range(1, 21):
            idx = self.current_step + i
            if idx < len(self.data):
                f_val = self.data.iloc[idx]['prediction']
            else:
                f_val = self.data.iloc[-1]['prediction']
            
            if i <= 5:
                cum_5d += f_val
            if i <= 10:
                cum_10d += f_val
            if i <= 15:
                cum_15d += f_val
            if i <= 20:
                cum_20d += f_val
                
        cum_5d_norm = np.clip(cum_5d / 500.0, 0.0, 1.0)
        cum_10d_norm = np.clip(cum_10d / 500.0, 0.0, 1.0)
        cum_15d_norm = np.clip(cum_15d / 500.0, 0.0, 1.0)
        cum_20d_norm = np.clip(cum_20d / 500.0, 0.0, 1.0)
        
        min_rsl = self.get_stock_remaining_shelf_life()
        min_rsl_norm = np.clip(float(min_rsl) / 60.0, 0.0, 1.0)
        
        arrival_day_tomorrow = self.current_step + 1
        in_transit_tomorrow_qty = self.in_transit.get(arrival_day_tomorrow, 0.0)
        in_transit_tomorrow_norm = np.clip(in_transit_tomorrow_qty / 500.0, 0.0, 1.0)
        capacity_utilization = np.clip(stock_total_atual / 500.0, 0.0, 1.0)
        
        if self.last_order_qty > 0:
            if self.last_order_qty <= 200.0:
                last_efficiency = self.last_order_qty / 200.0
            elif self.last_order_qty <= 850.0:
                last_efficiency = self.last_order_qty / 850.0
            else:
                last_efficiency = min(1.0, self.last_order_qty / 2000.0)
        else:
            last_efficiency = 1.0
            
        extra_features = [
            cum_5d_norm,
            cum_10d_norm,
            cum_15d_norm,
            cum_20d_norm,
            min_rsl_norm,
            in_transit_tomorrow_norm,
            capacity_utilization,
            last_efficiency
        ]
        
        # --- 5 NOVAS FEATURES COMPETITIVAS DE MERCADO (DIMENSÃO 25 -> 30) ---
        # 1. Dias consecutivos de rutura de stock (elimina o ponto cego da reputação POMDP)
        consecutive_oos_norm = np.clip(float(self.consecutive_oos_days) / 10.0, 0.0, 1.0)
        
        # 2 e 3. Condições climáticas para antecipação de degradação biológica (Temperatura e Humidade)
        T_c = float(row['temperature']) if 'temperature' in row else 22.0
        RH_pct = float(row['humidity']) if 'humidity' in row else 70.0
        temp_norm = np.clip((T_c - 5.0) / 30.0, 0.0, 1.0)
        humidity_norm = np.clip(RH_pct / 100.0, 0.0, 1.0)
        
        # 4. Concentração externa de etileno (acelera o amadurecimento e perda de firmeza)
        E_ext_ppm = float(row['ethylene']) if 'ethylene' in row else 0.8
        ethylene_norm = np.clip(E_ext_ppm / 3.0, 0.0, 1.0)
        
        # 5. Distância para completar palete fechada (sensibilidade aos custos de caixas avulsas)
        b_per_pal = float(self.boxes_per_pallet) if hasattr(self, 'boxes_per_pallet') and self.boxes_per_pallet > 0 else 50.0
        rem = float(prediction_today) % b_per_pal
        pallet_gap_norm = float((b_per_pal - rem) / b_per_pal) if rem > 0 else 0.0
        
        competitive_features = [
            consecutive_oos_norm,
            temp_norm,
            humidity_norm,
            ethylene_norm,
            pallet_gap_norm
        ]
        
        final_state = np.concatenate([final_state, extra_features, competitive_features])
        return final_state

    def get_checkpoint(self):
        return {
            'current_step': self.current_step,
            'stock_profile': list(self.stock_profile),
            'in_transit': {k: v for k, v in self.in_transit.items()},
            'active_batches': [dict(b) for b in self.active_batches],
            'last_order_qty': self.last_order_qty,
            'consecutive_oos_days': self.consecutive_oos_days
        }

    def load_checkpoint(self, checkpoint):
        self.current_step = checkpoint['current_step']
        self.stock_profile = list(checkpoint['stock_profile'])
        self.in_transit = {k: v for k, v in checkpoint['in_transit'].items()}
        self.last_order_qty = checkpoint.get('last_order_qty', 0.0)
        self.consecutive_oos_days = checkpoint.get('consecutive_oos_days', 0)
        if 'active_batches' in checkpoint:
            self.active_batches = [dict(b) for b in checkpoint['active_batches']]
        else:
            p = self.PRESETS[self.fruit_key]
            self.active_batches = [{
                'quantity': sum(self.stock_profile),
                'dureza': float(p["dureza_0_default"]),
                'brix': float(p["brix_0_default"]),
                'mold': 0.0,
                'E_int': float(p.get("E0_int", 0.01)),
                'age': 0.0,
                'quality': 100.0
            }]

    def clone(self):
        """ Cria um clone isolado e ultraleve do ambiente para rollouts paralelos em threads separadas """
        new_env = StockEnvironment.__new__(StockEnvironment)
        new_env.is_training = self.is_training
        new_env.completely_unconstrained = self.completely_unconstrained
        new_env.include_future_forecasts = self.include_future_forecasts
        new_env.enable_lifecycle = self.enable_lifecycle
        new_env.waste_penalty_factor = self.waste_penalty_factor
        new_env.latent_risk_factor = self.latent_risk_factor
        new_env.use_web_service = False
        new_env.web_service_url = self.web_service_url
        new_env.lifecycle_client = None
        new_env.df = self.df
        new_env.data = self.data
        new_env.max_order_limit = self.max_order_limit
        new_env.max_steps = self.max_steps
        new_env.current_step = self.current_step
        new_env.fruit_key = self.fruit_key
        new_env.preset_array = self.preset_array
        new_env.max_capacity = self.max_capacity
        new_env.product_volume_m3 = self.product_volume_m3
        new_env.CUSTO_TRANSPORTE_POR_M3 = self.CUSTO_TRANSPORTE_POR_M3
        new_env.CUSTO_ARMAZEM_POR_M3 = self.CUSTO_ARMAZEM_POR_M3
        new_env.TRUCK_CAPACITY_M3 = self.TRUCK_CAPACITY_M3
        new_env.TRUCK_CAPACITY = self.TRUCK_CAPACITY
        new_env.boxes_per_pallet = self.boxes_per_pallet
        new_env.cogs_ratio = self.cogs_ratio
        new_env.loose_box_fee = self.loose_box_fee
        new_env.dead_freight_factor = self.dead_freight_factor
        new_env.markdown_discount = self.markdown_discount
        new_env.wh_congestion_threshold = self.wh_congestion_threshold
        new_env.consecutive_oos_days = self.consecutive_oos_days
        new_env.eps = self.eps
        new_env.clip_val = self.clip_val
        new_env.stock_inicial = self.stock_inicial
        new_env.scaler = self.scaler
        new_env.stat_profit = self.stat_profit
        new_env._next_lot_id = self._next_lot_id
        
        # Deepcopy dos estados dinâmicos para isolamento total de threads
        new_env.stock_profile = list(self.stock_profile)
        new_env.in_transit = {k: v for k, v in self.in_transit.items()}
        new_env.active_batches = [dict(b) for b in self.active_batches]
        new_env.last_order_qty = self.last_order_qty
        return new_env

    def step(self, action_quantity, update_stats=True):
        """
        The Core Physics Engine (With biological presets, FEFO and active batching)
        """
        current_total_stock = sum(b['quantity'] for b in self.active_batches if b['quantity'] > 0)
        
        # 1. Enforce Capacity Limits, Action Capping (max_order_limit), and Integer Constraints
        if self.completely_unconstrained:
            raw_order = max(0, action_quantity)
        else:
            raw_order = max(0, min(action_quantity, self.max_order_limit, self.max_capacity - current_total_stock))
        order_qty = round(raw_order) 
        
        # 2. Process Arrivals
        arrived_today = self.in_transit.pop(self.current_step, 0)
        
        potential_stock = current_total_stock + arrived_today
        overflow_waste = max(0, potential_stock - self.max_capacity)
        
        accepted_arrivals = arrived_today - overflow_waste
        if accepted_arrivals > 0:
            p = self.PRESETS[self.fruit_key]
            current_date_str = self._get_current_date_str()
            self.active_batches.append({
                'lot_id': f"LOT-{self._next_lot_id}",
                'harvest_date': current_date_str,
                'initial_quantity': float(accepted_arrivals),
                'quantity': float(accepted_arrivals),
                'initial_dureza': float(p["dureza_0_default"]),
                'dureza': float(p["dureza_0_default"]),
                'initial_brix': float(p["brix_0_default"]),
                'brix': float(p["brix_0_default"]),
                'mold': 0.0,
                'E_int': float(p.get("E0_int", 0.01)),
                'age': 0.0,
                'quality': 100.0,
                'daily_readings': []
            })
            self._next_lot_id += 1

        # 3. The Oracle 
        row = self.data.iloc[self.current_step]
        real_demand = row['real_value']
        price_today = row['price'] if 'price' in row else 10.0
        
        # 4. Sell products (FEFO - Consome primeiro os lotes com menor RSL)
        self.active_batches.sort(key=lambda b: self.project_batch_rsl(b))
        remaining_demand = real_demand
        sales = 0.0
        markdown_sales = 0.0
        
        # Determinar o RSL de referência para distinguir vendas normais vs markdown (G3 < 25%)
        p = self.PRESETS[self.fruit_key]
        ref_batch = {
            'quantity': 1.0,
            'dureza': float(p["dureza_0_default"]),
            'brix': float(p["brix_0_default"]),
            'mold': 0.0,
            'E_int': float(p.get("E0_int", 0.01)),
            'age': 0.0,
            'quality': 100.0
        }
        max_rsl = max(1, self.project_batch_rsl(ref_batch))
        
        for b in self.active_batches:
            if b['quantity'] <= 0:
                continue
            take = min(b['quantity'], remaining_demand)
            b['quantity'] -= take
            sales += take
            
            # Se o lote tiver menos de 25% de vida restante (escalão crítico G3), conta como venda markdown
            b_rsl = self.project_batch_rsl(b)
            if (b_rsl / max_rsl) < 0.25:
                markdown_sales += take
                
            remaining_demand -= take
            if remaining_demand <= 0:
                break
                
        regular_sales = sales - markdown_sales
        missed_sales = remaining_demand
        
        # 5. Place New Order
        if order_qty > 0:
            arrival_day = self.current_step + 1 
            self.in_transit[arrival_day] = self.in_transit.get(arrival_day, 0) + order_qty

        # 6. Aging and Decay for each active batch
        spoilage = 0.0
        if not self.enable_lifecycle:
            # Sem lifecycle: sem apodrecimento biológico nem degradação de firmeza
            for b in self.active_batches:
                b['age'] = b.get('age', 0.0) + 1.0
        else:
            T_c = row['temperature'] if 'temperature' in row else 22.0
            RH_pct = row['humidity'] if 'humidity' in row else 70.0
            E_ext_ppm = row['ethylene'] if 'ethylene' in row else 0.8
            
            updated_batches = []
            for b in self.active_batches:
                if b['quantity'] <= 0:
                    continue
                b_next = self.advance_batch_one_day(b, T_c, RH_pct, E_ext_ppm)
                if b_next['quality'] < 30.0:
                    spoilage += b_next['quantity']
                else:
                    updated_batches.append(b_next)
            self.active_batches = updated_batches
        
        # Re-populate stock profile G0-G3
        self._update_stock_profile_from_batches()
        final_daily_stock = sum(self.stock_profile)

        # 7. Financial Math (Alinhado com Oracle Retail: AIP, OTM, RMS, RPM)
        # 7.0 Margem Comercial Líquida (Sem dedução de COGS; Considerando Desconto de Fim de Vida G3)
        unit_cost = float(row['cogs_unit']) if ('cogs_unit' in row and not np.isnan(row['cogs_unit'])) else price_today
        revenue_regular = regular_sales * price_today
        revenue_markdown = markdown_sales * (price_today * (1.0 - self.markdown_discount))
        gross_revenue = revenue_regular + revenue_markdown
        cogs = 0.0
        gross_margin = gross_revenue
        
        # 7.1 Armazenagem Base (Sem penalização de congestão de armazém)
        volume_stock_final = final_daily_stock * self.product_volume_m3
        storage_base_cost = volume_stock_final * self.CUSTO_ARMAZEM_POR_M3
        congestion_penalty = 0.0
        storage_cost = storage_base_cost
        
        # 7.2 Logística Dinâmica por Escalão de Encomenda (Carrinha / Intermédio / TIR - Calibrados)
        if order_qty > 0:
            order_volume_m3 = order_qty * self.product_volume_m3
            if order_qty <= 200:
                vehicle_type = "CARRINHA_ABERTA"
                cap_boxes = 200.0
                c_fixo = 15.00
                c_var = 18.00
                packaging_mode = "grade"
            elif order_qty <= 850:
                vehicle_type = "CAMIAO_INTERMEDIO"
                cap_boxes = 850.0
                c_fixo = 45.00
                c_var = 12.00
                packaging_mode = "palete"
            else:
                vehicle_type = "CAMIAO_TIR"
                cap_boxes = 2000.0
                c_fixo = 120.00
                c_var = 6.00
                packaging_mode = "palete"
                
            frete_base = c_fixo + (order_volume_m3 * c_var)
            occupancy = min(1.0, order_qty / cap_boxes)
            dead_freight_penalty = c_fixo * (1.0 - occupancy) * self.dead_freight_factor
            
            if packaging_mode == "palete":
                loose_boxes = order_qty % self.boxes_per_pallet
                loose_handling_fee = loose_boxes * self.loose_box_fee
            else:
                loose_boxes = 0
                loose_handling_fee = 0.0
                
            over_capacity = max(0.0, order_qty - cap_boxes)
            over_capacity_penalty = over_capacity * 10.0
            
            transport_cost = frete_base + dead_freight_penalty + loose_handling_fee + over_capacity_penalty
        else:
            vehicle_type = "NENHUM"
            packaging_mode = "N/A"
            transport_cost = 0.0
            dead_freight_penalty = 0.0
            loose_handling_fee = 0.0
            loose_boxes = 0
            occupancy = 1.0
            over_capacity_penalty = 0.0
            
        # 7.3 Desperdício Efetivo e Risco Preventivo de Spoilage
        total_lost_boxes = overflow_waste + spoilage
        base_waste_cost = total_lost_boxes * unit_cost
        direct_waste_cost = base_waste_cost * self.waste_penalty_factor

        # Penalização preventiva de risco latente (stock crítico em fase final G3 < 25% RSL)
        if self.latent_risk_factor > 0 and hasattr(self, 'stock_profile') and len(self.stock_profile) > 3:
            latent_risk_penalty = float(self.stock_profile[3] * self.latent_risk_factor)
        else:
            latent_risk_penalty = 0.0
        
        # 7.4 Nível de Serviço e Ruptura Consecutiva Equilibrada (Linear com teto, sem abismo)
        if missed_sales > 0:
            self.consecutive_oos_days += 1
        else:
            self.consecutive_oos_days = 0
            
        margin_per_unit = price_today
        stockout_cost = missed_sales * (margin_per_unit * 1.0)
        reputation_penalty = min(self.consecutive_oos_days, 5) * (margin_per_unit * 1.0)
        zero_stock_cost = 0.0
        
        # Lucro Comercial Puro (Sem fatores de shaping de treino)
        commercial_profit = (
            gross_margin
            - storage_cost
            - transport_cost
            - base_waste_cost
            - stockout_cost
            - reputation_penalty
            - zero_stock_cost
        )

        # Lucro Operacional Líquido (Com penalização modelada para o agente)
        daily_profit = (
            gross_margin
            - storage_cost
            - transport_cost
            - direct_waste_cost
            - latent_risk_penalty
            - stockout_cost
            - reputation_penalty
            - zero_stock_cost
        )
        
        if update_stats:
            self.stat_profit.push(daily_profit)
        
        # Z-Score contínuo robusto: transforma lucros relativos melhores em recompensas positivas (+Z)
        mean_p = self.stat_profit.mean
        std_p = self.stat_profit.std
        if self.stat_profit.n > 10 and std_p > 1e-4:
            reward_zscore = (daily_profit - mean_p) / (std_p + self.eps)
            reward_scaled = float(np.clip(reward_zscore, -self.clip_val, self.clip_val))
        else:
            reward_scaled = daily_profit / 100.0
        
        self.last_order_qty = float(order_qty)
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        next_state = self._get_state() if not done else np.zeros(30)
        
        info = {
            'sales': sales,
            'markdown_sales': markdown_sales,
            'regular_sales': regular_sales,
            'gross_margin': gross_margin,
            'cogs': cogs,
            'real_demand': real_demand,
            'missed_sales': missed_sales,
            'stockout_cost': stockout_cost,
            'reputation_penalty': reputation_penalty,
            'consecutive_oos_days': self.consecutive_oos_days,
            'transport_cost': transport_cost,
            'vehicle_type': vehicle_type,
            'packaging_mode': packaging_mode,
            'dead_freight_penalty': dead_freight_penalty,
            'loose_handling_fee': loose_handling_fee,
            'loose_boxes': loose_boxes,
            'vehicle_occupancy': occupancy,
            'storage_cost': storage_cost,
            'congestion_penalty': congestion_penalty,
            'order_placed': order_qty,
            'arrived_today': arrived_today,
            'overflow_waste': total_lost_boxes,
            'spoilage': spoilage,
            'base_waste_cost': base_waste_cost,
            'direct_waste_cost': direct_waste_cost,
            'latent_risk_penalty': latent_risk_penalty,
            'zero_stock_penalty': zero_stock_cost,
            'final_stock': final_daily_stock,
            'profit': daily_profit,
            'commercial_profit': commercial_profit,
            'price_today': price_today,
            'unit_cost': unit_cost
        }
        
        return next_state, reward_scaled, done, info
