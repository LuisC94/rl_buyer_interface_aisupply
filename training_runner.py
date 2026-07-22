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
               stockout_penalty=0.25, waste_penalty=1.0, zero_stock_penalty=5.0,
               max_shelf_life=15.0):
    """ Worker that manages a block of PPO environments in synchronization """
    # Set seed for this worker to ensure diversity
    worker_seed = 42 + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    
    envs = [StockEnvironment(excel_path=excel_path, is_training=True, train_split=train_split, 
                             max_capacity=capacity, shared_stats=shared_stats,
                             holding_cost=holding_cost, transport_cost=transport_cost, fixed_transport_cost=fixed_transport_cost,
                             stockout_penalty=stockout_penalty, waste_penalty=waste_penalty, zero_stock_penalty=zero_stock_penalty,
                             max_shelf_life=max_shelf_life) for _ in range(num_envs)]
    
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
                                stockout_penalty=0.25, waste_penalty=1.0, zero_stock_penalty=5.0,
                                max_shelf_life=15.0):
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
                             stockout_penalty=stockout_penalty, waste_penalty=waste_penalty, zero_stock_penalty=zero_stock_penalty,
                             max_shelf_life=max_shelf_life) for _ in range(num_envs)]
    
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
                               stockout_penalty=0.25, waste_penalty=1.0, zero_stock_penalty=5.0,
                               max_shelf_life=15.0):
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
                                               stockout_penalty, waste_penalty, zero_stock_penalty, max_shelf_life))
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
                           stockout_penalty=0.25, waste_penalty=1.0, zero_stock_penalty=5.0,
                           max_shelf_life=15.0, disruptions=None):
    """
    Runs evaluation simulation, comparing RL Agent, Min-Max Baseline, and Oracle
    Yields data dictionary daily for Streamlit live charting and logging.
    """
    yield {"status": "init", "msg": "[INIT] Preparando ambientes de teste..."}
    
    # 1. Initialize Environments
    env_test = StockEnvironment(excel_path=excel_path, is_training=False, train_split=train_split, max_capacity=max_capacity,
                                holding_cost=holding_cost, transport_cost=transport_cost, fixed_transport_cost=fixed_transport_cost,
                                stockout_penalty=stockout_penalty, waste_penalty=waste_penalty, zero_stock_penalty=zero_stock_penalty,
                                max_shelf_life=max_shelf_life)
    env_minmax = StockEnvironment(excel_path=excel_path, is_training=False, train_split=train_split, max_capacity=max_capacity,
                                  holding_cost=holding_cost, transport_cost=transport_cost, fixed_transport_cost=fixed_transport_cost,
                                  stockout_penalty=stockout_penalty, waste_penalty=waste_penalty, zero_stock_penalty=zero_stock_penalty,
                                  max_shelf_life=max_shelf_life)
    env_minmax.max_order_limit = float('inf') # Sem limite de encomenda para o Baseline
    env_timesupply = StockEnvironment(excel_path=excel_path, is_training=False, train_split=train_split, max_capacity=max_capacity,
                                      holding_cost=holding_cost, transport_cost=transport_cost, fixed_transport_cost=fixed_transport_cost,
                                      stockout_penalty=stockout_penalty, waste_penalty=waste_penalty, zero_stock_penalty=zero_stock_penalty,
                                      max_shelf_life=max_shelf_life)
    env_timesupply.max_order_limit = float('inf') # Sem limite de encomenda para o Baseline
    
    env_floatingpoint = StockEnvironment(excel_path=excel_path, is_training=False, train_split=train_split, max_capacity=max_capacity,
                                        holding_cost=holding_cost, transport_cost=transport_cost, fixed_transport_cost=fixed_transport_cost,
                                        stockout_penalty=stockout_penalty, waste_penalty=waste_penalty, zero_stock_penalty=zero_stock_penalty,
                                        max_shelf_life=max_shelf_life)
    env_floatingpoint.max_order_limit = float('inf') # Sem limite de encomenda para o Baseline
    
    env_train = StockEnvironment(excel_path=excel_path, is_training=True, train_split=train_split, max_capacity=max_capacity,
                                 holding_cost=holding_cost, transport_cost=transport_cost, fixed_transport_cost=fixed_transport_cost,
                                 stockout_penalty=stockout_penalty, waste_penalty=waste_penalty, zero_stock_penalty=zero_stock_penalty,
                                 max_shelf_life=max_shelf_life)
    
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
            for env in [env_test, env_minmax, env_timesupply, env_floatingpoint, env_train]:
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
    state_timesupply = env_timesupply.reset()
    state_floatingpoint = env_floatingpoint.reset()
    
    done = False
    
    # Tracking variables
    rewards_agent = []
    profits_agent = []
    actions_agent = []
    vendas_reais = []
    
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
        orig_max_shelf_life = {}
        orig_real_demands = {}
        for env in [env_test, env_minmax, env_timesupply, env_floatingpoint]:
            orig_max_shelf_life[env] = env.max_shelf_life
            orig_real_demands[env] = env.data.at[day, 'real_value']
            
        # Processar disrupções para o dia atual (1-indexed na UI)
        day_idx = day + 1
        day_disruptions = [d for d in disruptions if d["day"] == day_idx]
        for dis in day_disruptions:
            dis_type = dis.get("type")
            dis_param = dis.get("param")
            dis_val = dis.get("value")
            
            # 1. FORÇAR STOCK
            if dis_type == "stock_override":
                val = float(dis_val)
                for env in [env_test, env_minmax, env_timesupply, env_floatingpoint]:
                    if val == 0.0:
                        env.stock_profile = [0.0, 0.0, 0.0, 0.0]
                        env.active_batches = []
                    else:
                        env.stock_profile = [val, 0.0, 0.0, 0.0]
                        env.active_batches = [{
                            'quantity': val,
                            'age': 0.0,
                            'quality': 100.0,
                            'max_shelf_life': env.max_shelf_life
                        }]
                        
            # 2. PERDA DE CARGA EM TRÂNSITO (Afeta a mercadoria que chega HOJE)
            elif dis_type == "in_transit_loss":
                val = float(dis_val)
                for env in [env_test, env_minmax, env_timesupply, env_floatingpoint]:
                    if day in env.in_transit:
                        env.in_transit[day] = env.in_transit[day] * (1.0 - val)
                        
            # 3. QUALIDADE INFERIOR (Reduz a validade do lote que chega HOJE)
            elif dis_type == "shelf_life_drop":
                val = float(dis_val)
                for env in [env_test, env_minmax, env_timesupply, env_floatingpoint]:
                    env.max_shelf_life = val
                    
            # 4. FORÇAR PROCURA REAL
            elif dis_type == "demand_override":
                val = float(dis_val)
                for env in [env_test, env_minmax, env_timesupply, env_floatingpoint]:
                    env.data.at[day, 'real_value'] = val
                    
            # 5. REDUÇÃO DE CAPACIDADE DO ARMAZÉM (Persistente)
            elif dis_type == "capacity_drop":
                val = float(dis_val)
                for env in [env_test, env_minmax, env_timesupply, env_floatingpoint]:
                    env.max_capacity = val
                    
            # 6. ALTERAÇÃO DE CUSTO DA FUNÇÃO OBJETIVO (Persistente)
            elif dis_type == "cost_change":
                val = float(dis_val)
                p_name = dis_param
                for env in [env_test, env_minmax, env_timesupply, env_floatingpoint]:
                    if p_name == "holding_cost":
                        env.holding_cost = val
                    elif p_name == "transport_cost":
                        env.transport_cost = val
                    elif p_name == "fixed_transport_cost":
                        env.fixed_transport_cost = val
                    elif p_name == "stockout_penalty":
                        env.stockout_penalty = val
                    elif p_name == "waste_penalty":
                        env.waste_penalty = val
                    elif p_name == "zero_stock_penalty":
                        env.zero_stock_penalty = val
        
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
        cum_profit_timesupply += info_timesupply['profit']
        profits_timesupply.append(cum_profit_timesupply)
        log_stock_inicial_timesupply.append(stock_inicial_hoje_ts)
        log_stock_final_timesupply.append(sum(env_timesupply.stock_profile))
        log_vendas_timesupply.append(info_timesupply['sales'])
        log_lucro_diario_timesupply.append(info_timesupply['profit'])
        log_acoes_timesupply.append(action_timesupply)
        log_lucro_acumulado_timesupply.append(cum_profit_timesupply)
        
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
        cum_profit_floatingpoint += info_floatingpoint['profit']
        profits_floatingpoint.append(cum_profit_floatingpoint)
        log_stock_inicial_floatingpoint.append(stock_inicial_hoje_fp)
        log_stock_final_floatingpoint.append(sum(env_floatingpoint.stock_profile))
        log_vendas_floatingpoint.append(info_floatingpoint['sales'])
        log_lucro_diario_floatingpoint.append(info_floatingpoint['profit'])
        log_acoes_floatingpoint.append(action_floatingpoint)
        log_lucro_acumulado_floatingpoint.append(cum_profit_floatingpoint)
        
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
        
        log_msg = f"[Dia {day:03d}] Encomendas -> Agente: {info['order_placed']:03d} | MinMax: {info_minmax['order_placed']:03d} | TimeSupply: {action_timesupply:03d} | FloatingPoint: {action_floatingpoint:03d} || Lucro Agente: {info['profit']:.1f}€ | MinMax: {info_minmax['profit']:.1f}€"
        
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
        log_stock_inicial_agente.append(stock_inicial_hoje)
        log_stock_final_agente.append(stock_final_hoje)
        log_vendas_agente.append(sales)
        log_vendas_perdidas_agente.append(lost_sales)
        log_apodrecimento_agente.append(spoilage)
        log_excesso_agente.append(overcapacity_waste)
        log_lucro_diario_agente.append(info['profit'])
        log_lucro_acumulado_agente.append(cum_profit_agent)
        log_lucro_acumulado_minmax.append(cum_profit_minmax)
        
        flag_stockout.append(1 if stock_final_hoje <= 0 else 0)
        flag_clientes_perdidos.append(1 if lost_sales > 0 else 0)
        flag_excesso_armazem.append(1 if overcapacity_waste > 0 else 0)
        flag_apodrecimento.append(1 if spoilage > 0 else 0)
        
        # Aplicar atrasos de entrega à encomenda efetuada hoje (que chegaria no dia + 1)
        delay_dis = [d for d in day_disruptions if d["type"] == "delay"]
        if delay_dis:
            delay_days = int(delay_dis[0]["value"])
            for env in [env_test, env_minmax, env_timesupply, env_floatingpoint]:
                if (day + 1) in env.in_transit:
                    qty = env.in_transit.pop(day + 1)
                    target_day = day + 1 + delay_days
                    env.in_transit[target_day] = env.in_transit.get(target_day, 0.0) + qty
                    
        # Restauro de atributos temporários modificados
        for env in [env_test, env_minmax, env_timesupply, env_floatingpoint]:
            env.max_shelf_life = orig_max_shelf_life[env]
            env.data.at[day, 'real_value'] = orig_real_demands[env]
            
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
            'Desperdicio_Validade_Agente': log_apodrecimento_agente,
            'Excesso_Armazem_Agente': log_excesso_agente,
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
        "lost_sales_total": sum(log_vendas_perdidas_agente),
        "overflow_waste_total": sum(log_excesso_agente),
        "excel_report_path": excel_report_path,
        "final_model_path": final_model_path,
        "update_days": update_days,
        "log_dias": log_dias,
        "log_lucro_acumulado_agente": log_lucro_acumulado_agente,
        "log_lucro_acumulado_minmax": log_lucro_acumulado_minmax,
        "log_lucro_acumulado_timesupply": log_lucro_acumulado_timesupply,
        "log_lucro_acumulado_floatingpoint": log_lucro_acumulado_floatingpoint,
        "log_acoes_agente": log_acoes_agente,
        "log_vendas_agente": log_vendas_agente,
        "log_vendas_perdidas_agente": log_vendas_perdidas_agente,
        "log_apodrecimento_agente": log_apodrecimento_agente,
        "log_stock_final_agente": log_stock_final_agente
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

