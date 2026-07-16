import os
import random
import torch
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt

try:
    from environment_constrained import StockEnvironment, EnvRunningStat
except ImportError:
    from rl_buyer_interface.environment_constrained import StockEnvironment, EnvRunningStat

try:
    from agent.ppo_agent import ParallelPPOAgent
except ImportError:
    from rl_buyer_interface.agent.ppo_agent import ParallelPPOAgent

# --- WORKER FOR MULTI-CORE PPO ---
def ppo_worker(worker_id, excel_path, train_split, num_envs, capacity, weights_queue, results_queue, shared_stats, horizon=90,
               holding_cost=0.70, transport_cost=10.0, fixed_transport_cost=10.0,
               stockout_penalty=0.25, waste_penalty=1.0, zero_stock_penalty=5.0):
    """ Worker that manages a block of PPO environments in synchronization """
    # Set seed for this worker to ensure diversity
    worker_seed = 42 + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    
    envs = [StockEnvironment(excel_path=excel_path, is_training=True, train_split=train_split, 
                             max_capacity=capacity, shared_stats=shared_stats,
                             holding_cost=holding_cost, transport_cost=transport_cost, fixed_transport_cost=fixed_transport_cost,
                             stockout_penalty=stockout_penalty, waste_penalty=waste_penalty, zero_stock_penalty=zero_stock_penalty) for _ in range(num_envs)]
    
    max_order_limit = envs[0].max_order_limit
    agent = ParallelPPOAgent(state_dim=17, action_dim=1, max_action=max_order_limit)
    agent.device = torch.device('cpu') 
    agent.policy_old_actor.to('cpu')
    agent.policy_old_critic.to('cpu')
    
    states = [env.reset() for env in envs]
    states_matrix = np.array(states)
    
    while True:
        weights = weights_queue.get()
        if weights is None: 
            break
        
        agent.policy_old_actor.load_state_dict(weights['actor'])
        worker_memory = {
            'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'dones': [], 'profits': []
        }
        
        for step in range(horizon):
            with torch.no_grad():
                st_t = torch.FloatTensor(states_matrix).to('cpu')
                action_mean, log_std = agent.policy_old_actor(st_t)
                dist = torch.distributions.Normal(action_mean, torch.exp(torch.clamp(log_std, -2.3, 1.5)))
                action_percent = dist.sample()
                action_logprob = dist.log_prob(action_percent)
                physical_actions = torch.round(torch.clamp(action_percent * max_order_limit, 0, max_order_limit)).numpy().flatten()

            worker_memory['states'].append(st_t)
            worker_memory['actions'].append(action_percent)
            worker_memory['logprobs'].append(action_logprob)
            
            next_states_list = []
            rewards_list = []
            dones_list = []
            profits_list = []
            
            for i in range(num_envs):
                ns, r, d, info = envs[i].step(physical_actions[i])
                if d:
                    ns = envs[i].reset()
                next_states_list.append(ns)
                rewards_list.append(r)
                dones_list.append(d)
                profits_list.append(info['profit'])
            
            worker_memory['rewards'].append(rewards_list)
            worker_memory['dones'].append(dones_list)
            worker_memory['profits'].append(profits_list)
            
            states_matrix = np.array(next_states_list)
            
        with torch.no_grad():
            st_final = torch.FloatTensor(states_matrix).to('cpu')
            worker_memory['states'].append(st_final)
        
        results_queue.put({
            'worker_id': worker_id,
            'memory': worker_memory,
            'total_profit': np.sum(worker_memory['profits'])
        })

# --- SINGLE CORE TRAINING GENERATOR ---
def train_single_core_generator(seed, excel_path, train_split, max_capacity, lr_actor, lr_critic, gamma, k_epochs, eps_clip, batch_size, max_episodes_total, num_envs, horizon=90, save_dir="modelos_producao_constrained",
                                holding_cost=0.70, transport_cost=10.0, fixed_transport_cost=10.0,
                                stockout_penalty=0.25, waste_penalty=1.0, zero_stock_penalty=5.0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    yield "[INFO] Initializing training in SINGLE-CORE mode..."
    
    # 1. Determinar dinamicamente o limite de encomenda
    df_temp = pd.read_excel(excel_path)
    split_idx = int(len(df_temp) * train_split)
    MAX_ORDER_LIMIT = float(df_temp.iloc[:split_idx]['real_value'].max())
    
    yield f"[INFO] Max Daily Order Limit (MAX_ORDER_LIMIT) = {MAX_ORDER_LIMIT} units (Warehouse Capacity = {max_capacity} units)"
    yield f"[INFO] State dimension = 17, Action dimension = 1"
    
    shared_stats = {
        'econ': EnvRunningStat(), 'eco': EnvRunningStat(), 'risk': EnvRunningStat()
    }
    
    envs = [StockEnvironment(excel_path=excel_path, is_training=True, train_split=train_split, 
                             max_capacity=max_capacity, shared_stats=shared_stats,
                             holding_cost=holding_cost, transport_cost=transport_cost, fixed_transport_cost=fixed_transport_cost,
                             stockout_penalty=stockout_penalty, waste_penalty=waste_penalty, zero_stock_penalty=zero_stock_penalty) for _ in range(num_envs)]
    
    agent = ParallelPPOAgent(state_dim=17, action_dim=1, max_action=MAX_ORDER_LIMIT, 
                             lr_actor=lr_actor, lr_critic=lr_critic, gamma=gamma, K_epochs=k_epochs, eps_clip=eps_clip, batch_size=batch_size)
    
    os.makedirs(save_dir, exist_ok=True)
    
    states = [env.reset() for env in envs]
    states_matrix = np.array(states)
    
    episodes_played = 0
    iteration = 0
    
    losses_total = []
    losses_actor = []
    losses_critic = []
    
    yield "[START] Training started successfully."
    
    while episodes_played < max_episodes_total:
        iteration += 1
        
        # Rollout
        for step in range(horizon):
            with torch.no_grad():
                st_t = torch.FloatTensor(states_matrix).to(agent.device)
                action_mean, log_std = agent.policy_old_actor(st_t)
                dist = torch.distributions.Normal(action_mean, torch.exp(torch.clamp(log_std, -2.3, 1.5)))
                action_percent = dist.sample()
                action_logprob = dist.log_prob(action_percent)
                physical_actions = torch.round(torch.clamp(action_percent * MAX_ORDER_LIMIT, 0, MAX_ORDER_LIMIT)).cpu().numpy().flatten()

            agent.buffer.states.append(st_t)
            agent.buffer.actions.append(action_percent)
            agent.buffer.logprobs.append(action_logprob)
            
            next_states_list = []
            rewards_list = []
            dones_list = []
            profits_list = []
            
            for i in range(num_envs):
                ns, r, d, info = envs[i].step(physical_actions[i])
                if d:
                    ns = envs[i].reset()
                next_states_list.append(ns)
                rewards_list.append(r)
                dones_list.append(d)
                profits_list.append(info['profit'])
            
            agent.buffer.rewards.append(rewards_list)
            agent.buffer.is_terminals.append(dones_list)
            
            states_matrix = np.array(next_states_list)
            
        with torch.no_grad():
            st_final = torch.FloatTensor(states_matrix).to(agent.device)
            agent.buffer.states.append(st_final)
            
        # Update
        loss_t, loss_a, loss_c = agent.update()
        losses_total.append(loss_t)
        losses_actor.append(loss_a)
        losses_critic.append(loss_c)
        
        episodes_played += num_envs
        avg_profit = np.mean([np.sum(profits_list) / (horizon/30.0) for _ in range(num_envs)]) # aproximado por env
        
        yield f"Episodes: {episodes_played}/{max_episodes_total} | Batch Profit Avg: {avg_profit:.2f}€ | Loss: {loss_t:.4f}"
        
        # Save checkpoints periodically
        if iteration % 5 == 0 or episodes_played >= max_episodes_total:
            checkpoint_path = os.path.join(save_dir, "ppo_constrained_checkpoint")
            agent.save(checkpoint_path)
            econ_state = {
                'n': shared_stats['econ'].n,
                'mean': shared_stats['econ'].mean,
                'S': shared_stats['econ'].S
            }
            torch.save(econ_state, checkpoint_path + '_econ_stat.pth')

    # Save final model
    final_path = os.path.join(save_dir, "ppo_constrained_final")
    agent.save(final_path)
    econ_state = {
        'n': shared_stats['econ'].n,
        'mean': shared_stats['econ'].mean,
        'S': shared_stats['econ'].S
    }
    torch.save(econ_state, final_path + '_econ_stat.pth')
    
    # Save Loss Evolution Chart
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(losses_total, label="Total Loss", color="#1f77b4")
        plt.plot(losses_actor, label="Actor Loss", color="#2ca02c", alpha=0.7)
        plt.plot(losses_critic, label="Critic Loss", color="#d62728", alpha=0.7)
        plt.title(f"Evolution of Training Losses (Constrained PPO)")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plot_path = os.path.join(save_dir, "losses_plot.png")
        plt.savefig(plot_path, dpi=200)
        plt.close()
    except Exception as e:
        yield f"[WARNING] Could not create loss chart: {e}"
        
    yield "[SUCCESS] Single-Core Training completed successfully!"

# --- MULTI CORE TRAINING GENERATOR ---
def train_multi_core_generator(seed, excel_path, train_split, max_capacity, lr_actor, lr_critic, gamma, k_epochs, eps_clip, batch_size, max_episodes_total, num_envs, num_workers, horizon=90, save_dir="modelos_producao_constrained",
                               holding_cost=0.70, transport_cost=10.0, fixed_transport_cost=10.0,
                               stockout_penalty=0.25, waste_penalty=1.0, zero_stock_penalty=5.0):
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass # Método de início já definido
        
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    yield f"[INFO] Initializing training in MULTI-CORE mode with {num_workers} workers..."
    
    # Determinar dinamicamente o limite de encomenda
    df_temp = pd.read_excel(excel_path)
    split_idx = int(len(df_temp) * train_split)
    MAX_ORDER_LIMIT = float(df_temp.iloc[:split_idx]['real_value'].max())
    
    yield f"[INFO] Max Daily Order Limit (MAX_ORDER_LIMIT) = {MAX_ORDER_LIMIT} units (Warehouse Capacity = {max_capacity} units)"
    
    shared_stats = {
        'econ': EnvRunningStat(), 'eco': EnvRunningStat(), 'risk': EnvRunningStat()
    }
    
    agent = ParallelPPOAgent(state_dim=17, action_dim=1, max_action=MAX_ORDER_LIMIT, 
                             lr_actor=lr_actor, lr_critic=lr_critic, gamma=gamma, K_epochs=k_epochs, eps_clip=eps_clip, batch_size=batch_size)
    
    os.makedirs(save_dir, exist_ok=True)
    
    envs_per_worker = num_envs // num_workers
    weights_queues = [mp.Queue() for _ in range(num_workers)]
    results_queue = mp.Queue()
    
    processes = []
    for i in range(num_workers):
        p = mp.Process(target=ppo_worker, args=(i, excel_path, train_split, envs_per_worker, max_capacity, weights_queues[i], results_queue, shared_stats, horizon,
                                               holding_cost, transport_cost, fixed_transport_cost,
                                               stockout_penalty, waste_penalty, zero_stock_penalty))
        p.start()
        processes.append(p)
        
    episodes_played = 0
    iteration = 0
    
    losses_total = []
    losses_actor = []
    losses_critic = []
    
    yield "[START] Multi-core training started."
    
    try:
        while episodes_played < max_episodes_total:
            iteration += 1
            current_weights = {
                'actor': {k: v.cpu() for k, v in agent.policy_old_actor.state_dict().items()}
            }
            for q in weights_queues:
                q.put(current_weights)
            
            all_worker_data = []
            for _ in range(num_workers):
                all_worker_data.append(results_queue.get())
            
            T = len(all_worker_data[0]['memory']['rewards'])
            
            for t in range(T):
                step_states = torch.cat([res['memory']['states'][t] for res in all_worker_data], dim=0).to(agent.device)
                step_actions = torch.cat([res['memory']['actions'][t] for res in all_worker_data], dim=0).to(agent.device)
                step_logprobs = torch.cat([res['memory']['logprobs'][t] for res in all_worker_data], dim=0).to(agent.device)
                step_rewards = []
                step_dones = []
                for res in all_worker_data:
                    step_rewards.extend(res['memory']['rewards'][t])
                    step_dones.extend(res['memory']['dones'][t])
                
                agent.buffer.states.append(step_states)
                agent.buffer.actions.append(step_actions)
                agent.buffer.logprobs.append(step_logprobs)
                agent.buffer.rewards.append(step_rewards)
                agent.buffer.is_terminals.append(step_dones)
                
            if len(all_worker_data[0]['memory']['states']) > T:
                final_states = torch.cat([res['memory']['states'][T] for res in all_worker_data], dim=0).to(agent.device)
                agent.buffer.states.append(final_states)
            
            loss_t, loss_a, loss_c = agent.update()
            losses_total.append(loss_t)
            losses_actor.append(loss_a)
            losses_critic.append(loss_c)
            
            episodes_played += num_envs
            avg_profit = np.mean([res['total_profit'] / envs_per_worker for res in all_worker_data])
            
            yield f"Episodes: {episodes_played}/{max_episodes_total} | Batch Profit Avg: {avg_profit:.2f}€ | Loss: {loss_t:.4f}"
            
            if iteration % 5 == 0:
                checkpoint_path = os.path.join(save_dir, "ppo_constrained_checkpoint")
                agent.save(checkpoint_path)
                econ_state = {
                    'n': shared_stats['econ'].n,
                    'mean': shared_stats['econ'].mean,
                    'S': shared_stats['econ'].S
                }
                torch.save(econ_state, checkpoint_path + '_econ_stat.pth')
                
    except Exception as e:
        yield f"[ERROR] Failure in training loop: {e}"
    finally:
        yield "[CLEANUP] Terminating subprocesses..."
        for q in weights_queues: 
            q.put(None)
        for p in processes: 
            p.join()
            
        final_path = os.path.join(save_dir, "ppo_constrained_final")
        agent.save(final_path)
        
        econ_state = {
            'n': shared_stats['econ'].n,
            'mean': shared_stats['econ'].mean,
            'S': shared_stats['econ'].S
        }
        torch.save(econ_state, final_path + '_econ_stat.pth')
        
        # Save Loss Evolution Chart
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(losses_total, label="Total Loss", color="#1f77b4")
            plt.plot(losses_actor, label="Actor Loss", color="#2ca02c", alpha=0.7)
            plt.plot(losses_critic, label="Critic Loss", color="#d62728", alpha=0.7)
            plt.title(f"Evolution of Training Losses (Constrained PPO) - Multi-Core")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)
            plot_path = os.path.join(save_dir, "losses_plot.png")
            plt.savefig(plot_path, dpi=200)
            plt.close()
        except Exception as e:
            yield f"[WARNING] Could not create loss chart: {e}"
            
        yield "[SUCCESS] Multi-Core Training finished successfully!"

# --- TEST AND FINE-TUNING SIMULATOR STEP ---
def continual_training_step(agent, new_experiences, env_train, max_action_val, lr_actor=1e-5, lr_critic=5e-5, batch_size=32):
    """ Executes 1 cycle of online update (Fine-Tuning) using a Mixed Buffer """
    for param_group in agent.optimizer_actor.param_groups:
        param_group['lr'] = lr_actor
    for param_group in agent.optimizer_critic.param_groups:
        param_group['lr'] = lr_critic
        
    num_new_days = len(new_experiences)
    num_old_needed = num_new_days * 4 # 80/20 proportion
    
    # 1. Generate old experiences using training environment
    state = env_train.reset()
    dias_gerados = 0
    agent.policy_old_actor.eval()
    
    while dias_gerados < num_old_needed:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            action_mean, log_std = agent.policy_old_actor(state_tensor)
            dist = torch.distributions.Normal(action_mean, torch.exp(torch.clamp(log_std, -2.3, 1.5)))
            action_percent = dist.sample()
            action_logprob = dist.log_prob(action_percent)
            physical_action = torch.round(torch.clamp(action_percent * max_action_val, 0, max_action_val)).cpu().numpy().flatten()[0]
            
        next_state, reward, done, _ = env_train.step(physical_action)
        
        agent.buffer.states.append(state_tensor) 
        agent.buffer.actions.append(action_percent) 
        agent.buffer.logprobs.append(action_logprob) 
        agent.buffer.rewards.append([reward]) 
        agent.buffer.is_terminals.append([done]) 
        
        state = next_state
        dias_gerados += 1
        
        if done:
            state = env_train.reset()
            
    # 2. Inject new experiences
    for exp in new_experiences:
        agent.buffer.states.append(torch.FloatTensor(exp['state']).unsqueeze(0).to(agent.device)) 
        agent.buffer.actions.append(torch.FloatTensor(exp['action']).unsqueeze(0).to(agent.device)) 
        agent.buffer.logprobs.append(torch.FloatTensor(exp['logprob']).unsqueeze(0).to(agent.device)) 
        agent.buffer.rewards.append([exp['reward']]) 
        agent.buffer.is_terminals.append([exp['is_terminal']]) 
        
    # 3. Update network weights
    agent.update()



# --- SIMULATION RUNNER ---
def run_testing_simulation(excel_path, train_split, max_capacity, initial_model_base_path, s_min, S_max, update_interval_days=15, online_lr_actor=1e-5, online_lr_critic=5e-5, online_batch_size=32, save_dir="modelos_producao_constrained",
                           holding_cost=0.70, transport_cost=10.0, fixed_transport_cost=10.0,
                           stockout_penalty=0.25, waste_penalty=1.0, zero_stock_penalty=5.0):
    """
    Runs evaluation simulation, comparing RL Agent, Min-Max Baseline, and Oracle
    Yields data dictionary daily for Streamlit live charting and logging.
    """
    yield {"status": "init", "msg": "[INIT] Preparando ambientes de teste..."}
    
    # 1. Initialize Environments
    env_test = StockEnvironment(excel_path=excel_path, is_training=False, train_split=train_split, max_capacity=max_capacity,
                                holding_cost=holding_cost, transport_cost=transport_cost, fixed_transport_cost=fixed_transport_cost,
                                stockout_penalty=stockout_penalty, waste_penalty=waste_penalty, zero_stock_penalty=zero_stock_penalty)
    env_minmax = StockEnvironment(excel_path=excel_path, is_training=False, train_split=train_split, max_capacity=max_capacity,
                                  holding_cost=holding_cost, transport_cost=transport_cost, fixed_transport_cost=fixed_transport_cost,
                                  stockout_penalty=stockout_penalty, waste_penalty=waste_penalty, zero_stock_penalty=zero_stock_penalty)
    env_minmax.max_order_limit = float('inf') # Sem limite de encomenda para o Baseline
    env_oracle = StockEnvironment(excel_path=excel_path, is_training=False, train_split=train_split, max_capacity=max_capacity,
                                  holding_cost=holding_cost, transport_cost=transport_cost, fixed_transport_cost=fixed_transport_cost,
                                  stockout_penalty=stockout_penalty, waste_penalty=waste_penalty, zero_stock_penalty=zero_stock_penalty)
    env_oracle.max_order_limit = float('inf') # Sem limite de encomenda para o Oráculo
    
    env_train = StockEnvironment(excel_path=excel_path, is_training=True, train_split=train_split, max_capacity=max_capacity,
                                 holding_cost=holding_cost, transport_cost=transport_cost, fixed_transport_cost=fixed_transport_cost,
                                 stockout_penalty=stockout_penalty, waste_penalty=waste_penalty, zero_stock_penalty=zero_stock_penalty)
    
    state_dim = 17
    action_dim = 1
    max_order_limit = env_test.max_order_limit
    
    yield {"status": "init", "msg": f"[INIT] Max Daily Order Limit (Action Cap): {max_order_limit} units"}
    
    # 2. Instantiate and Load Agent
    agent = ParallelPPOAgent(state_dim=state_dim, action_dim=action_dim, max_action=max_order_limit, batch_size=online_batch_size)
    
    try:
        agent.load(initial_model_base_path)
        yield {"status": "init", "msg": f"[OK] Base Model successfully loaded!"}
    except Exception as e:
        yield {"status": "error", "msg": f"[CRITICAL ERROR] Failed to load base model: {e}"}
        return
        
    # Load running stats for Z-score if exists
    econ_stat_path = initial_model_base_path + '_econ_stat.pth'
    if os.path.exists(econ_stat_path):
        try:
            econ_state = torch.load(econ_stat_path, weights_only=False, map_location='cpu')
            yield {"status": "init", "msg": "[OK] Environment Z-Score stats loaded!"}
            for env in [env_test, env_minmax, env_oracle, env_train]:
                env.stat_profit.n = econ_state['n']
                env.stat_profit.mean = econ_state['mean']
                env.stat_profit.S = econ_state['S']
        except Exception as e:
            yield {"status": "init", "msg": f"[WARNING] Failed to load {econ_stat_path}: {e}. Starting from scratch."}
    else:
        yield {"status": "init", "msg": "[WARNING] _econ_stat.pth file not found. Dynamic scaler will start from zero."}
        
    agent.policy_old_actor.to(agent.device)
    agent.policy_old_actor.eval()
    
    state = env_test.reset()
    state_minmax = env_minmax.reset()
    state_oracle = env_oracle.reset()
    
    done = False
    
    # Tracking variables
    rewards_agent = []
    profits_agent = []
    actions_agent = []
    vendas_reais = []
    
    profits_minmax = []
    profits_oracle = []
    
    update_days = []
    
    cum_profit_agent = 0
    cum_profit_minmax = 0
    cum_profit_oracle = 0
    
    dias_simulados = 0
    
    # Log variables for Excel exporter
    log_dias = []
    log_procura_real = []
    log_preco_venda = []
    
    log_acoes_agente = []
    log_acoes_minmax = []
    log_acoes_oracle = []
    
    log_stock_inicial_agente = []
    log_stock_final_agente = []
    log_vendas_agente = []
    log_vendas_perdidas_agente = []
    log_apodrecimento_agente = []
    log_excesso_agente = []
    
    log_lucro_diario_agente = []
    log_lucro_acumulado_agente = []
    log_lucro_acumulado_minmax = []
    log_lucro_acumulado_oracle = []

    # Min-Max detailed logs
    log_stock_inicial_minmax = []
    log_stock_final_minmax = []
    log_vendas_minmax = []
    log_vendas_perdidas_minmax = []
    log_apodrecimento_minmax = []
    log_excesso_minmax = []
    log_lucro_diario_minmax = []
    
    # Oracle detailed logs
    log_stock_inicial_oracle = []
    log_stock_final_oracle = []
    log_vendas_oracle = []
    log_vendas_perdidas_oracle = []
    log_apodrecimento_oracle = []
    log_excesso_oracle = []
    log_lucro_diario_oracle = []
    
    flag_stockout = []
    flag_clientes_perdidos = []
    flag_excesso_armazem = []
    flag_apodrecimento = []
    
    current_15d_buffer = []
    
    yield {"status": "start", "msg": "[PRODUCTION] Continuous market simulation started."}
    
    while not done:
        day = env_test.current_step
        
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
        
        # --- ORACLE BASELINE ---
        stock_hoje_oracle = sum(env_oracle.stock_profile) + env_oracle.in_transit.get(env_oracle.current_step, 0)
        vendas_hoje_oracle = env_oracle.data.iloc[env_oracle.current_step]['real_value']
        stock_amanha_oracle = max(0, stock_hoje_oracle - vendas_hoje_oracle) + env_oracle.in_transit.get(env_oracle.current_step + 1, 0)
        
        action_oracle = 0
        demand_tomorrow = env_oracle.data.iloc[env_oracle.current_step + 1]['real_value'] if env_oracle.current_step + 1 <= env_oracle.max_steps else 0
        
        if stock_amanha_oracle < demand_tomorrow:
            future_demand = 0
            for i in range(1, 5): # T+1 a T+4
                idx = env_oracle.current_step + i
                if idx <= env_oracle.max_steps:
                    future_demand += env_oracle.data.iloc[idx]['real_value']
            action_oracle = max(0, future_demand - stock_amanha_oracle)
            action_oracle = min(action_oracle, max_capacity)
            
        stock_inicial_hoje_oracle = sum(env_oracle.stock_profile)
        _, _, _, info_oracle = env_oracle.step(action_oracle)
        cum_profit_oracle += info_oracle['profit']
        profits_oracle.append(cum_profit_oracle)
        
        stock_final_hoje_oracle = sum(env_oracle.stock_profile)
        sales_or = info_oracle['sales']
        spoilage_or = info_oracle['spoilage']
        lost_sales_or = max(0, info_oracle['real_demand'] - sales_or)
        overcapacity_waste_or = max(0, info_oracle['overflow_waste'] - spoilage_or)
        
        log_stock_inicial_oracle.append(stock_inicial_hoje_oracle)
        log_stock_final_oracle.append(stock_final_hoje_oracle)
        log_vendas_oracle.append(sales_or)
        log_vendas_perdidas_oracle.append(lost_sales_or)
        log_apodrecimento_oracle.append(spoilage_or)
        log_excesso_oracle.append(overcapacity_waste_or)
        log_lucro_diario_oracle.append(info_oracle['profit'])
        
        # --- RL AGENT ---
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            action_mean, log_std = agent.policy_old_actor(state_tensor)
            dist = torch.distributions.Normal(action_mean, torch.exp(torch.clamp(log_std, -2.3, 1.5)))
            action_percent = dist.sample()
            action_logprob = dist.log_prob(action_percent)
            physical_action = torch.round(torch.clamp(action_percent * max_order_limit, 0, max_order_limit)).cpu().numpy().flatten()[0]
            
        stock_inicial_hoje = sum(env_test.stock_profile)
        next_state, reward, done_env, info = env_test.step(physical_action)
        done = done_env
        
        log_msg = f"[Dia {day:03d}] Encomendas -> Agente: {info['order_placed']:03d} | MinMax: {info_minmax['order_placed']:03d} | Oráculo: {info_oracle['order_placed']:03d} || Lucro Agente: {info['profit']:.1f}€ | MinMax: {info_minmax['profit']:.1f}€"
        
        # Store experience
        current_15d_buffer.append({
            'state': state_tensor.squeeze(0).cpu().numpy(),
            'action': action_percent.squeeze(0).cpu().numpy(),
            'logprob': action_logprob.squeeze(0).cpu().numpy(),
            'reward': reward,
            'is_terminal': done
        })
        
        # Global stats tracking
        rewards_agent.append(reward)
        cum_profit_agent += info['profit']
        profits_agent.append(cum_profit_agent)
        actions_agent.append(physical_action)
        vendas_reais.append(env_test.data.iloc[day]['real_value'])
        
        # Detail stats
        stock_final_hoje = sum(env_test.stock_profile)
        sales = info['sales']
        real_demand = info['real_demand']
        spoilage = info['spoilage']
        lost_sales = max(0, real_demand - sales)
        overcapacity_waste = max(0, info['overflow_waste'] - spoilage)
        
        log_dias.append(day)
        log_procura_real.append(real_demand)
        log_preco_venda.append(info['price_today'])
        log_acoes_agente.append(info['order_placed'])
        log_acoes_minmax.append(info_minmax['order_placed'])
        log_acoes_oracle.append(info_oracle['order_placed'])
        log_stock_inicial_agente.append(stock_inicial_hoje)
        log_stock_final_agente.append(stock_final_hoje)
        log_vendas_agente.append(sales)
        log_vendas_perdidas_agente.append(lost_sales)
        log_apodrecimento_agente.append(spoilage)
        log_excesso_agente.append(overcapacity_waste)
        log_lucro_diario_agente.append(info['profit'])
        log_lucro_acumulado_agente.append(cum_profit_agent)
        log_lucro_acumulado_minmax.append(cum_profit_minmax)
        log_lucro_acumulado_oracle.append(cum_profit_oracle)
        
        flag_stockout.append(1 if stock_final_hoje <= 0 else 0)
        flag_clientes_perdidos.append(1 if lost_sales > 0 else 0)
        flag_excesso_armazem.append(1 if overcapacity_waste > 0 else 0)
        flag_apodrecimento.append(1 if spoilage > 0 else 0)
        
        state = next_state
        dias_simulados += 1
        
        update_triggered = False
        # --- TRIGGER FINE-TUNING ---
        if dias_simulados % update_interval_days == 0 and not done:
            update_triggered = True
            update_days.append(dias_simulados)
            
            # Execute fine-tuning step
            continual_training_step(agent, current_15d_buffer, env_train, max_order_limit,
                                    lr_actor=online_lr_actor, lr_critic=online_lr_critic, batch_size=online_batch_size)
            
            current_15d_buffer = []
            agent.policy_old_actor.eval()
            log_msg += f" -> [FINE-TUNING ACTIVE: Model v{len(update_days)}]"
            
        # Yield daily stats to display in UI dynamically
        yield {
            "status": "running",
            "day": day,
            "msg": log_msg,
            "agent_profit_cum": cum_profit_agent,
            "minmax_profit_cum": cum_profit_minmax,
            "oracle_profit_cum": cum_profit_oracle,
            "agent_action": info['order_placed'],
            "minmax_action": info_minmax['order_placed'],
            "oracle_action": info_oracle['order_placed'],
            "real_demand": real_demand,
            "stock_level": stock_final_hoje,
            "order_placed": info['order_placed'],
            "spoilage": spoilage,
            "overflow_waste": overcapacity_waste,
            "minmax_stock_level": stock_final_hoje_minmax,
            "minmax_spoilage": spoilage_mm,
            "update_triggered": update_triggered,
            "version_count": len(update_days)
        }
        
    # Done - Save final model
    os.makedirs(save_dir, exist_ok=True)
    final_model_path = os.path.join(save_dir, "ppo_constrained_online_final")
    agent.save(final_model_path)
    
    final_econ_state = {
        'n': env_test.stat_profit.n,
        'mean': env_test.stat_profit.mean,
        'S': env_test.stat_profit.S
    }
    torch.save(final_econ_state, final_model_path + '_econ_stat.pth')
    
    # Generate static report Excel
    excel_report_path = os.path.join(save_dir, "Relatorio_Simulacao.xlsx")
    try:
        df_excel = pd.DataFrame({
            'Dia': log_dias,
            'Procura_Real': log_procura_real,
            'Preco_Venda_Dia': log_preco_venda,
            
            # AGENTE PPO
            'Acao_Agente_PPO': log_acoes_agente,
            'Stock_Inicial_Agente': log_stock_inicial_agente,
            'Stock_Final_Agente': log_stock_final_agente,
            'Vendas_Agente': log_vendas_agente,
            'Vendas_Perdidas_Agente': log_vendas_perdidas_agente,
            'Apodrecimento_Agente': log_apodrecimento_agente,
            'Excesso_Armazem_Agente': log_excesso_agente,
            'Lucro_Diario_Agente': log_lucro_diario_agente,
            'Lucro_Acumulado_Agente': log_lucro_acumulado_agente,
            
            # BASELINE MIN-MAX
            'Acao_MinMax': log_acoes_minmax,
            'Stock_Inicial_MinMax': log_stock_inicial_minmax,
            'Stock_Final_MinMax': log_stock_final_minmax,
            'Vendas_MinMax': log_vendas_minmax,
            'Vendas_Perdidas_MinMax': log_vendas_perdidas_minmax,
            'Apodrecimento_MinMax': log_apodrecimento_minmax,
            'Excesso_Armazem_MinMax': log_excesso_minmax,
            'Lucro_Diario_MinMax': log_lucro_diario_minmax,
            'Lucro_Acumulado_MinMax': log_lucro_acumulado_minmax,
            
            # ORÁCULO
            'Acao_Oraculo': log_acoes_oracle,
            'Stock_Inicial_Oracle': log_stock_inicial_oracle,
            'Stock_Final_Oracle': log_stock_final_oracle,
            'Vendas_Oracle': log_vendas_oracle,
            'Vendas_Perdidas_Oracle': log_vendas_perdidas_oracle,
            'Apodrecimento_Oracle': log_apodrecimento_oracle,
            'Excesso_Armazem_Oracle': log_excesso_oracle,
            'Lucro_Diario_Oracle': log_lucro_diario_oracle,
            'Lucro_Acumulado_Oraculo': log_lucro_acumulado_oracle,
            
            'Flag_Stockout': flag_stockout,
            'Flag_Clientes_Perdidos': flag_clientes_perdidos,
            'Flag_Excesso_Armazem': flag_excesso_armazem,
            'Flag_Apodrecimento': flag_apodrecimento
        })
        df_excel.to_excel(excel_report_path, index=False)
    except Exception as e:
        yield {"status": "warning", "msg": f"[WARNING] Failed to save Excel report: {e}"}
        
    yield {
        "status": "complete",
        "msg": f"[COMPLETED] Simulation finished after {dias_simulados} days.",
        "cum_profit_agent": cum_profit_agent,
        "cum_profit_minmax": cum_profit_minmax,
        "cum_profit_oracle": cum_profit_oracle,
        "stockout_days": sum(flag_stockout),
        "spoilage_total": sum(log_apodrecimento_agente),
        "lost_sales_total": sum(log_vendas_perdidas_agente),
        "overflow_waste_total": sum(log_excesso_agente),
        "excel_report_path": excel_report_path,
        "final_model_path": final_model_path,
        "update_days": update_days
    }
