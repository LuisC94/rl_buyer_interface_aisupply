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
    Uses clean age-based shelf life decay logic.
    """
    
    def __init__(self, excel_path, is_training=True, train_split=0.6, max_capacity=1000, shared_stats=None,
                 holding_cost=0.70, transport_cost=10.0, fixed_transport_cost=10.0,
                 stockout_penalty=0.25, waste_penalty=1.0, zero_stock_penalty=5.0,
                 max_shelf_life=15.0):
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
        
        self.max_shelf_life = float(max_shelf_life)
            
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
        
        # Inicializa o armazém com o lote inicial no dia 0
        self.active_batches = [{
            'quantity': self.stock_inicial,
            'age': 0.0,
            'quality': 100.0
        }]
        
        self.in_transit = {}
        self._update_stock_profile_from_batches()
        return self._get_state()

    def advance_batch_one_day(self, batch):
        """ Avança a idade do lote por 1 dia. Qualidade decresce linearmente com a idade. """
        max_life = self.max_shelf_life
        age = batch.get('age', 0.0) + 1.0
        quality = max(0.0, 100.0 * (1.0 - age / max_life))
        return {
            'quantity': batch['quantity'],
            'age': age,
            'quality': quality
        }

    def project_batch_rsl(self, batch):
        """ Projeta os dias restantes até que a qualidade caia abaixo de 30.0 """
        max_life = self.max_shelf_life
        age = batch.get('age', 0.0)
        limit_age = 0.7 * max_life
        rsl = max(0, int(math.ceil(limit_age - age)))
        return rsl

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
            self.active_batches = [{
                'quantity': sum(self.stock_profile),
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
            self.active_batches.append({
                'quantity': float(accepted_arrivals),
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
        spoilage = 0.0
        updated_batches = []
        
        for b in self.active_batches:
            if b['quantity'] <= 0:
                continue
            b_next = self.advance_batch_one_day(b)
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
