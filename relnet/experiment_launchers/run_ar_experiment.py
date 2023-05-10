import sys
from copy import deepcopy
from pathlib import Path

sys.path.append('/relnet')
from relnet.agent.mcts.mcts_agent import AR40RandMCTSAgent
from relnet.environment.graph_edge_env import GraphEdgeEnv
from relnet.evaluation.file_paths import FilePaths
from relnet.objective_functions.objective_functions import LargestComponentSizeTargeted, GlobalEfficiency
from relnet.state.network_generators import NetworkGenerator
from relnet.state.geometric_network_generators import KHNetworkGenerator

from billiard.pool import Pool
import pandas as pd

edge_percentage = 10
obj_fun_kwargs = {"random_seed": 42, "mc_sims_multiplier": 0.25}

def get_options(file_paths, seed):
    mcts_opts = {}
    mcts_opts['random_seed'] = 42 * seed
    mcts_opts['draw_trees'] = False
    mcts_opts['log_progress'] = True
    mcts_opts['log_filename'] = file_paths.construct_log_filepath()
    mcts_opts['parallel_eval'] = False
    mcts_opts['num_simulations'] = 1
    return mcts_opts


def get_file_paths(exp_id):
    parent_dir = '/experiment_data'
    file_paths = FilePaths(parent_dir, exp_id, setup_directories=True)
    return file_paths

def get_R_data(obj_fun, g, g_size, eval_seed):
    local_data = []
    options = get_options(file_paths, eval_seed)
    agent = create_agent(obj_fun, options)
    avg_perf = agent.eval([g])

    data_item = {}
    data_item['obj_fun'] = obj_fun.name
    data_item['g_size'] = g_size
    data_item['seed'] = eval_seed
    data_item['R'] = avg_perf
    local_data.append(data_item)
    agent.finalize()
    return local_data


def create_agent(obj_fun, options):
    env = GraphEdgeEnv(obj_fun, obj_fun_kwargs, edge_percentage)
    agent = AR40RandMCTSAgent(env)
    hyperparams = agent.get_default_hyperparameters()
    hyperparams['expansion_budget_modifier'] = 20
    opts_copy = deepcopy(options)
    agent.setup(opts_copy, hyperparams)
    return agent

if __name__ == '__main__':
    all_gen_params = [NetworkGenerator.get_default_generator_params(),
                      NetworkGenerator.get_default_generator_params_med(),
                      NetworkGenerator.get_default_generator_params_large(),
                      ]

    obj_funs = [ GlobalEfficiency(), LargestComponentSizeTargeted()]
    exp_id = 'ar_test'

    num_seeds = 1000

    print(f"running with {num_seeds} seeds.")

    storage_root = Path('/experiment_data/stored_graphs')
    kwargs = {'store_graphs': True, 'graph_storage_root': storage_root}
    fp = get_file_paths(exp_id)
    csv_path = fp.figures_dir / f"R_data_{num_seeds}.csv"

    gen = KHNetworkGenerator(**kwargs)

    file_paths = get_file_paths(exp_id)
    graph_seeds = NetworkGenerator.construct_network_seeds(0, 0, 1)
    train_graph_seeds, validation_graph_seeds, test_graph_seeds = graph_seeds


    pool_size = 16
    worker_pool = Pool(processes=pool_size)
    eval_tasks = []

    for gp in all_gen_params:
        n = gp['n']
        train_graphs = gen.generate_many(gp, train_graph_seeds)
        validation_graphs = gen.generate_many(gp, validation_graph_seeds)
        test_graphs = gen.generate_many(gp, test_graph_seeds)

        for obj_fun in obj_funs:
            for eval_seed in range(num_seeds):
                g = test_graphs[0]
                eval_tasks.append((obj_fun, g, n, eval_seed))

    data = []
    for local_data in worker_pool.starmap(get_R_data, eval_tasks):
        data.extend(local_data)
    pd.DataFrame(data).to_csv(csv_path, header=True)



