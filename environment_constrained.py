import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
import math
import os

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
            "Tref_C": 5.0, "Ea_J": 60000.0, "k_firm_ref": 0.06, "alpha_E": 1.8,
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
            "Tref_C": 5.0, "Ea_J": 50000.0, "k_firm_ref": 0.025, "alpha_E": 0.8,
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
            "Tref_C": 5.0, "Ea_J": 52000.0, "k_firm_ref": 0.035, "alpha_E": 1.1,
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
            "Tref_C": 5.0, "Ea_J": 48000.0, "k_firm_ref": 0.04, "alpha_E": 1.3,
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
            "Tref_C": 5.0, "Ea_J": 47000.0, "k_firm_ref": 0.018, "alpha_E": 0.6,
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

    def __init__(self, excel_path, is_training=True, train_split=0.6, max_capacity=1000, shared_stats=None,
                 holding_cost=0.70, transport_cost=10.0, fixed_transport_cost=10.0,
                 stockout_penalty=0.25, waste_penalty=1.0, zero_stock_penalty=5.0):
        # 1. Load Data
        if isinstance(excel_path, pd.DataFrame):
            self.df = excel_path
            excel_name = "custom_dataframe.xlsx"
        else:
            self.df = pd.read_excel(excel_path)
            excel_name = os.path.basename(excel_path).lower()
            
        # 2. Train/Test Split
        split_index = int(len(self.df) * train_split)
        
        # Determinar dinamicamente o limite máximo histórico de procura/vendas no treino
        train_data_df = self.df.iloc[:split_index]
        if 'real_value' in train_data_df.columns:
            self.max_order_limit = float(train_data_df['real_value'].max())
        else:
            self.max_order_limit = 166.0 # Fallback seguro
            
        if is_training:
            self.data = self.df.iloc[:split_index].reset_index(drop=True)
        else:
            self.data = self.df.iloc[split_index:].reset_index(drop=True)
            
        self.max_steps = len(self.data) - 1
        self.current_step = 0
        
        # Mapeamento do SKU para chave do preset biológico
        if "3_080" in excel_name:
            self.fruit_key = "maca_gala"
        elif "3_090" in excel_name:
            self.fruit_key = "maca_fuji"
        elif "3_252" in excel_name:
            self.fruit_key = "kiwi_hayward"
        elif "3_586" in excel_name or "2_586" in excel_name:
            self.fruit_key = "maca_golden"
        elif "911753" in excel_name:
            self.fruit_key = "maca_reineta"
        else:
            self.fruit_key = "maca_gala"
            
        # 3. Physics & Fixed Constraints
        self.max_capacity = max_capacity
        self.stock_inicial = 100.0
        
        # --- LOGÍSTICA VOLUMÉTRICA UNIVERSAL (M3) ---
        self.CUSTO_ARMAZEM_POR_M3 = holding_cost
        self.CUSTO_TRANSPORTE_POR_M3 = transport_cost
        self.TAXA_PARAGEM_CAMIAO = fixed_transport_cost
        self.STOCKOUT_PENALTY_MULT = stockout_penalty
        self.WASTE_PENALTY_MULT = waste_penalty
        self.ZERO_STOCK_PENALTY_MULT = zero_stock_penalty
        
        # Extrair Volume do Produto (m3) e Preço Médio (se a coluna não existir, usa Fallbacks)
        self.product_volume_m3 = self.data['volume'].iloc[0] if 'volume' in self.data.columns else 0.002
        self.avg_price = self.data['price'].mean() if 'price' in self.data.columns else 2.0
        
        self.stock_profile = [0.0, 0.0, 0.0, 0.0] # [G0, G1, G2, G3]
        self.active_batches = []                  # Lotes físicos ativos: [{'quantity', 'dureza', 'brix', 'mold', 'E_int', 'age', 'quality'}]
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

    def reset(self):
        """ Resets the world to Day 0 """
        self.current_step = 0
        p = self.PRESETS[self.fruit_key]
        
        # Inicializa o armazém com o lote inicial no dia 0
        self.active_batches = [{
            'quantity': self.stock_inicial,
            'dureza': float(p["dureza_0_default"]),
            'brix': float(p["brix_0_default"]),
            'mold': 0.0,
            'E_int': float(p.get("E0_int", 0.01)),
            'age': 0.0,
            'quality': 100.0
        }]
        
        self.in_transit = {}
        self._update_stock_profile_from_batches()
        return self._get_state()

    def advance_batch_one_day(self, batch, T_c, RH_pct, E_ext_ppm):
        """ Avança a maturação biológica de um lote por 1 dia usando dt = 0.05 """
        p = self.PRESETS[self.fruit_key]
        R = 8.314
        dt = 0.05
        steps = int(1.0 / dt)
        
        T_K = T_c + 273.15
        Tref_K = p["Tref_C"] + 273.15
        
        def k_temp_scaling(Ea, T, Tref):
            return math.exp((-Ea / R) * (1.0 / T - 1.0 / Tref))
            
        def sigmoid(x):
            return 1.0 / (1.0 + math.exp(-x))
            
        kT_firm = p["k_firm_ref"] * k_temp_scaling(p["Ea_J"], T_K, Tref_K)
        
        RH_ref = p["RH_ref"]
        RH_deficit = max(0.0, (RH_ref - RH_pct) / 100.0)
        kRH = 1.0 + p["beta_RH"] * RH_deficit
        
        Ea_E_J = float(p.get("Ea_E_J", 52000.0))
        Eref_prod = float(p.get("Eref_prod", 0.08))
        E_decay = float(p.get("E_decay", 0.7))
        E_t0 = float(p.get("E_t0", 15.0))
        E_g = float(p.get("E_g", 0.8))
        E_auto = float(p.get("E_auto", 0.35))
        E_ext_shift = float(p.get("E_ext_shift", 2.0))
        
        t0_eff = E_t0 - E_ext_shift * math.log1p(max(0.0, E_ext_ppm))
        prod_T = k_temp_scaling(Ea_E_J, T_K, Tref_K)
        
        dureza_min = float(p["dureza_min"])
        alpha_E = float(p["alpha_E"])
        
        brix_min = float(p["brix_min"])
        brix_max = float(p["brix_max"])
        r0 = float(p["brix_g"])
        alpha_bE = 0.25
        bRH = max(0.0, (RH_ref - RH_pct) / 100.0)
        rRH = 1.0 - 0.6 * bRH
        rT = k_temp_scaling(Ea_E_J, T_K, Tref_K)
        
        RH_mold_thr = float(p["RH_mold_thr"])
        mold_rate_ref = float(p["mold_rate_ref"])
        mold_sens_RH = float(p["mold_sens_RH"])
        mold_max_penalty = float(p["mold_max_penalty"])
        Ea_mold_J = float(p["Ea_mold_J"])
        
        mold_T = k_temp_scaling(Ea_mold_J, T_K, Tref_K)
        RH_excess = max(0.0, (RH_pct - RH_mold_thr) / 100.0)
        RH_factor = 1.0 - math.exp(-mold_sens_RH * RH_excess)
        
        dureza = batch['dureza']
        brix = batch['brix']
        mold = batch['mold']
        E_int = batch.get('E_int', float(p.get("E0_int", 0.01)))
        age = batch.get('age', 0.0)
        
        for _ in range(steps):
            current_age = age + (_ * dt)
            ramp = sigmoid(E_g * (current_age - t0_eff))
            prod = Eref_prod * prod_T * ramp
            dE = (prod * (1.0 + E_auto * E_int) - E_decay * E_int) * dt
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
            
        firm_score = 1.0 / (1.0 + math.exp(-0.35 * (dureza - float(p["qual_firm_threshold"]))))
        brix_score = math.exp(-((brix - float(p["qual_brix_target"]))**2) / 2.0)
        quality_base = 100.0 * (0.65 * firm_score + 0.35 * brix_score)
        
        mold_penalty = mold_max_penalty * mold
        quality = quality_base * (1.0 - mold_penalty)
        
        return {
            'quantity': batch['quantity'],
            'dureza': dureza,
            'brix': brix,
            'mold': mold,
            'E_int': E_int,
            'age': age + 1.0,
            'quality': quality
        }

    def project_batch_rsl(self, batch):
        """ Projeta os dias restantes até que a qualidade caia abaixo de 30.0 """
        b = {
            'quantity': batch['quantity'],
            'dureza': batch['dureza'],
            'brix': batch['brix'],
            'mold': batch['mold'],
            'E_int': batch.get('E_int', 0.01),
            'age': batch.get('age', 0.0),
            'quality': batch.get('quality', 100.0)
        }
        
        row = self.data.iloc[self.current_step]
        T_c = row['temperature'] if 'temperature' in row else 1.5
        RH_pct = row['humidity'] if 'humidity' in row else 92.0
        E_ext_ppm = row['ethylene'] if 'ethylene' in row else 0.05
        
        days_remaining = 0
        max_projection_days = 60
        
        while days_remaining < max_projection_days:
            if b['quality'] < 30.0:
                break
            b = self.advance_batch_one_day(b, T_c, RH_pct, E_ext_ppm)
            days_remaining += 1
            
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
        """ Atualiza self.stock_profile (G0-G3) somando os lotes pelo seu RSL """
        self.stock_profile = [0.0, 0.0, 0.0, 0.0]
        for b in self.active_batches:
            if b['quantity'] <= 0:
                continue
            rsl = self.project_batch_rsl(b)
            if rsl >= 4:
                self.stock_profile[0] += b['quantity']
            elif rsl == 3:
                self.stock_profile[1] += b['quantity']
            elif rsl == 2:
                self.stock_profile[2] += b['quantity']
            elif rsl == 1:
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
        return final_state

    def get_checkpoint(self):
        return {
            'current_step': self.current_step,
            'stock_profile': list(self.stock_profile),
            'in_transit': {k: v for k, v in self.in_transit.items()},
            'active_batches': [dict(b) for b in self.active_batches]
        }

    def load_checkpoint(self, checkpoint):
        self.current_step = checkpoint['current_step']
        self.stock_profile = list(checkpoint['stock_profile'])
        self.in_transit = {k: v for k, v in checkpoint['in_transit'].items()}
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

    def step(self, action_quantity, update_stats=True):
        """
        The Core Physics Engine (With biological presets, FEFO and active batching)
        """
        current_total_stock = sum(b['quantity'] for b in self.active_batches if b['quantity'] > 0)
        
        # 1. Enforce Capacity Limits, Action Capping (max_order_limit), and Integer Constraints
        raw_order = max(0, min(action_quantity, self.max_order_limit, self.max_capacity))
        order_qty = round(raw_order) 
        
        # 2. Process Arrivals
        arrived_today = self.in_transit.pop(self.current_step, 0)
        
        potential_stock = current_total_stock + arrived_today
        overflow_waste = max(0, potential_stock - self.max_capacity)
        
        accepted_arrivals = arrived_today - overflow_waste
        if accepted_arrivals > 0:
            p = self.PRESETS[self.fruit_key]
            self.active_batches.append({
                'quantity': float(accepted_arrivals),
                'dureza': float(p["dureza_0_default"]),
                'brix': float(p["brix_0_default"]),
                'mold': 0.0,
                'E_int': float(p.get("E0_int", 0.01)),
                'age': 0.0,
                'quality': 100.0
            })

        # 3. The Oracle 
        row = self.data.iloc[self.current_step]
        real_demand = row['real_value']
        price_today = row['price'] if 'price' in row else 10.0
        
        # 4. Sell products (FEFO - Consome primeiro os lotes com menor RSL)
        self.active_batches.sort(key=lambda b: self.project_batch_rsl(b))
        remaining_demand = real_demand
        sales = 0
        
        for b in self.active_batches:
            if b['quantity'] <= 0:
                continue
            take = min(b['quantity'], remaining_demand)
            b['quantity'] -= take
            sales += take
            remaining_demand -= take
            if remaining_demand <= 0:
                break
                
        missed_sales = remaining_demand
        
        # 5. Place New Order
        if order_qty > 0:
            arrival_day = self.current_step + 1 
            self.in_transit[arrival_day] = self.in_transit.get(arrival_day, 0) + order_qty

        # 6. Aging and Decay for each active batch
        T_c = row['temperature'] if 'temperature' in row else 1.5
        RH_pct = row['humidity'] if 'humidity' in row else 92.0
        E_ext_ppm = row['ethylene'] if 'ethylene' in row else 0.05
        
        spoilage = 0.0
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

        # 7. Financial Math
        gross_profit = sales * price_today
        volume_stock_final = final_daily_stock * self.product_volume_m3
        storage_cost = volume_stock_final * self.CUSTO_ARMAZEM_POR_M3
        
        if order_qty > 0:
            order_volume_m3 = order_qty * self.product_volume_m3
            transport_cost = self.TAXA_PARAGEM_CAMIAO + (order_volume_m3 * self.CUSTO_TRANSPORTE_POR_M3)
        else:
            transport_cost = 0.0
            
        stockout_cost = missed_sales * (price_today * self.STOCKOUT_PENALTY_MULT) 
        total_lost_boxes = overflow_waste + spoilage
        waste_cost = total_lost_boxes * (price_today * self.WASTE_PENALTY_MULT)
        zero_stock_cost = (price_today * self.ZERO_STOCK_PENALTY_MULT) if final_daily_stock <= 0 else 0.0

        daily_profit = gross_profit - storage_cost - transport_cost - stockout_cost - waste_cost - zero_stock_cost
        
        if update_stats:
            self.stat_profit.push(daily_profit)
        
        reward_z = (daily_profit - self.stat_profit.mean) / (self.stat_profit.std + self.eps)
        reward_z = np.clip(reward_z, -self.clip_val, self.clip_val)
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        next_state = self._get_state() if not done else np.zeros(17)
        
        info = {
            'sales': sales,
            'real_demand': real_demand,
            'order_placed': order_qty,
            'arrived_today': arrived_today,
            'overflow_waste': total_lost_boxes,
            'spoilage': spoilage,
            'zero_stock_penalty': zero_stock_cost,
            'profit': daily_profit,
            'price_today': price_today
        }
        
        return next_state, reward_z, done, info
