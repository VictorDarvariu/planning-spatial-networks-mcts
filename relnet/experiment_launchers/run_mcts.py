import sys
from copy import deepcopy
from pathlib import Path

from relnet.agent.baseline.baseline_agent import GreedyAgent, CostSensitiveGreedyAgent

sys.path.append('/relnet')
from relnet.agent.mcts.mcts_agent import AR40RandMCTSAgent, SGUCTAgent, StandardMCTSAgent
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

if __name__ == '__main__':
    # gen_params = NetworkGenerator.get_default_generator_params_tiny()
    # gen_params = NetworkGenerator.get_default_generator_params()
    # gen_params = NetworkGenerator.get_default_generator_params_med()
    # gen_params = NetworkGenerator.get_default_generator_params_large()
    # gen_params = NetworkGenerator.get_default_generator_params_xlarge()
    # gen_params = NetworkGenerator.get_default_generator_params_xxlarge()
    # gen_params = NetworkGenerator.get_default_generator_params_xxxlarge()
    # gen_params = NetworkGenerator.get_default_generator_params_4xlarge()
    gen_params = NetworkGenerator.get_default_generator_params_5xlarge()

    obj_fun = GlobalEfficiency()
    exp_id = 'psn_development'

    storage_root = Path('/experiment_data/stored_graphs')
    kwargs = {'store_graphs': True, 'graph_storage_root': storage_root}
    fp = get_file_paths(exp_id)
    gen = KHNetworkGenerator(**kwargs)

    file_paths = get_file_paths(exp_id)

    n_graphs = 1

    graph_seeds = NetworkGenerator.construct_network_seeds(0, 0, n_graphs)
    train_graph_seeds, validation_graph_seeds, test_graph_seeds = graph_seeds

    n = gen_params['n']
    train_graphs = gen.generate_many(gen_params, train_graph_seeds)
    validation_graphs = gen.generate_many(gen_params, validation_graph_seeds)
    test_graphs = gen.generate_many(gen_params, test_graph_seeds)

    #agent_class = AR40RandMCTSAgent
    agent_class = StandardMCTSAgent

    seed = 42
    options = get_options(fp, seed)
    options['log_timings'] = True
    options['timings_path'] = fp.timings_dir

    perfs = []

    for g in test_graphs:
        env = GraphEdgeEnv(obj_fun, obj_fun_kwargs, edge_percentage)
        agent = agent_class(env)
        hyperparams = agent.get_default_hyperparameters()
        # hyperparams['expansion_budget_modifier'] = 20
        hyperparams['expansion_budget_modifier'] = 5
        # hyperparams['sim_policy'] = 'min_cost'
        hyperparams['sim_policy'] = 'random'
        # hyperparams['sim_policy_bias'] = 25
        opts_copy = deepcopy(options)

        if agent_class in [GreedyAgent, CostSensitiveGreedyAgent]:
            opts_copy['model_identifier_prefix'] = f'{agent_class.algorithm_name}_default_{n}'
        else:
            opts_copy['model_identifier_prefix'] = f"{agent_class.algorithm_name}_default_{n}_{hyperparams['sim_policy']}"
        agent.setup(opts_copy, hyperparams)

        perf = agent.eval([g])
        perfs.append(perf)

    print(f"test set perfs: {perfs}")







