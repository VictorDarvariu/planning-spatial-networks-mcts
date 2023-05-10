from relnet.agent.baseline.baseline_agent import *
from relnet.agent.mcts.mcts_agent import *
from relnet.objective_functions.objective_functions import *
from relnet.state.geometric_network_generators import KHNetworkGenerator, GeometricMetroNetworkGenerator, \
    GeometricInternetTopologyNetworkGenerator
from relnet.state.network_generators import NetworkGenerator


class ExperimentConditions(object):
    def __init__(self, base_n, possible_edge_percentage, train_individually, restriction_mechanism):
        self.gen_params = {}
        self.base_n = base_n
        self.possible_edge_percentage = possible_edge_percentage
        self.train_individually = train_individually
        self.restriction_mechanism = restriction_mechanism

        self.gen_params['n'] = self.base_n
        self.gen_params['m_percentage_er'] = 20
        self.gen_params['m_ba'] = 2
        self.gen_params['m_ws'] = 2
        self.gen_params['p_ws'] = 0.1
        self.gen_params['d_reg'] = 2
        self.gen_params['alpha_kh'] = 10
        self.gen_params['beta_kh'] = 0.001

        self.gen_params['m'] = NetworkGenerator.compute_number_edges(self.gen_params['n'], self.gen_params['m_percentage_er'])

        #self.num_mc_sims_multiplier = 1
        self.num_mc_sims_multiplier = 0.25

        self.obj_fun_seed = 42

        #size_multipliers = [1]
        self.size_multipliers = [1]
        #self.size_multipliers = [1, 1.5, 2]

        self.greedy_size_threshold = 2.5

        self.agents_models = []

        self.objective_functions = [
            LargestComponentSizeTargeted,
            GlobalEfficiency,
        ]

        self.agent_budgets = {
            LargestComponentSizeTargeted.name: {
            },
            GlobalEfficiency.name: {
            }
        }

        self.network_generators = [
            KHNetworkGenerator,
        ]

        self.experiment_params = {'train_graphs': 0,
                                  'validation_graphs': 50,
                                  'test_graphs': 50,
                                  'num_runs': 10
                                  }

        self.use_geometric_features = True


    def get_model_seed(self, run_number):
        return run_number * 42

    def update_size_dependant_params(self, multiplier):
        self.gen_params['n'] = int(self.base_n * multiplier)
        self.gen_params['m'] = NetworkGenerator.compute_number_edges(self.gen_params['n'], self.gen_params['m_percentage_er'])

        self.gen_params['size_multiplier'] = multiplier

    def set_generator_seeds(self):
        self.train_seeds, self.validation_seeds, self.test_seeds = NetworkGenerator.construct_network_seeds(
            self.experiment_params['train_graphs'],
            self.experiment_params['validation_graphs'],
            self.experiment_params['test_graphs'])

    def set_generator_seeds_individually(self, g_num, num_graphs, one_graph=False):
        self.validation_seeds = [g_num]
        self.test_seeds = [g_num]

        if not one_graph:
            self.train_seeds = [g_num + (i * num_graphs) for i in range(1, num_graphs)]
        else:
            self.train_seeds = [g_num]

    def update_relevant_agents(self, algorithm_class):
        if algorithm_class == "model":
            relevant_agents = deepcopy(self.agents_models)
        else:
            relevant_agents = deepcopy(self.agents_planning)
        self.relevant_agents = relevant_agents

    def extend_seeds_to_skip(self, run_num_start, run_num_end):
        for net in self.network_generators:
            for obj in self.objective_functions:
                for agent in self.relevant_agents:
                    setting = (net.name, obj.name, agent.algorithm_name)
                    if setting not in self.model_seeds_to_skip:
                        self.model_seeds_to_skip[setting] = []

                    for run_num_before in range(0, run_num_start):
                        self.model_seeds_to_skip[setting].append(self.get_model_seed(run_num_before))

                    for run_num_after in range(run_num_end + 1, self.experiment_params['num_runs']):
                        self.model_seeds_to_skip[setting].append(self.get_model_seed(run_num_after))

    def __str__(self):
        as_dict = deepcopy(self.__dict__)
        del as_dict["agents_models"]
        del as_dict["agents_planning"]
        del as_dict["agents_baseline"]
        del as_dict["objective_functions"]
        del as_dict["network_generators"]
        return str(as_dict)

    def __repr__(self):
        return self.__str__()


class PreliminaryExperimentConditions(ExperimentConditions):
    def __init__(self, base_n, possible_edge_percentage, train_individually, restriction_mechanism):
        super().__init__(base_n, possible_edge_percentage, train_individually, restriction_mechanism)

        self.agents_baseline = {
            LargestComponentSizeTargeted.name: [
                MinCostAgent,
                RandomAgent,
                GreedyAgent,
                CostSensitiveGreedyAgent,
                LowestDegreeProductAgent,
                FiedlerVectorAgent,
                EffectiveResistanceAgent
            ],
            GlobalEfficiency.name: [
                MinCostAgent,
                RandomAgent,
                GreedyAgent,
                CostSensitiveGreedyAgent,
                LBHBAgent,
            ],
        }

        self.agents_planning = [
            StandardMCTSAgent
        ]

        self.experiment_params['model_seeds'] = [self.get_model_seed(run_num) for run_num in
                                                 range(self.experiment_params['num_runs'])]

        self.hyperparam_grids = self.create_hyperparam_grids()

        # Can be used to skip some parameter combinations by their int id
        self.parameter_combs_to_skip = {
        }

        self.model_seeds_to_skip = {
        }

    def create_hyperparam_grids(self):
        hyperparam_grid_base =  {
            StandardMCTSAgent.algorithm_name: {
                "C_p": [0.05, 0.1, 0.25, 0.5, 1],
                "adjust_C_p": [True],
                "expansion_budget_modifier": [1], #20
                "rollout_depth": [1]
            },
        }
        hyperparam_grids = {}
        for f in self.objective_functions:
            hyperparam_grids[f.name] = deepcopy(hyperparam_grid_base)
        return hyperparam_grids


class BTMExperimentConditions(ExperimentConditions):
    def __init__(self, base_n, possible_edge_percentage, train_individually, restriction_mechanism):
        super().__init__(base_n, possible_edge_percentage, train_individually, restriction_mechanism)

        self.agents_baseline = {
            LargestComponentSizeTargeted.name: [],
            GlobalEfficiency.name: [],
        }

        self.agents_planning = [
            BTMMCTSAgent
        ]

        self.experiment_params['model_seeds'] = [self.get_model_seed(run_num) for run_num in
                                                 range(self.experiment_params['num_runs'])]

        self.hyperparam_grids = self.create_hyperparam_grids()

        # Can be used to skip some parameter combinations by their int id
        self.parameter_combs_to_skip = {
        }

        self.model_seeds_to_skip = {
        }

    def create_hyperparam_grids(self):
        hyperparam_grid_base = {}
        hyperparam_grids = {}
        for f in self.objective_functions:
            hyperparam_grids[f.name] = deepcopy(hyperparam_grid_base)
        return hyperparam_grids


class MinCostExperimentConditions(ExperimentConditions):
    def __init__(self, base_n, possible_edge_percentage, train_individually, restriction_mechanism):
        super().__init__(base_n, possible_edge_percentage, train_individually, restriction_mechanism)

        self.agents_baseline = {
            LargestComponentSizeTargeted.name: [],
            GlobalEfficiency.name: [],
        }

        self.agents_planning = [
            MinCostMCTSAgent
        ]

        self.experiment_params['model_seeds'] = [self.get_model_seed(run_num) for run_num in
                                                 range(self.experiment_params['num_runs'])]

        self.hyperparam_grids = self.create_hyperparam_grids()

        # Can be used to skip some parameter combinations by their int id
        self.parameter_combs_to_skip = {
        }

        self.model_seeds_to_skip = {
        }

    def create_hyperparam_grids(self):
        hyperparam_grid_base =  {
            MinCostMCTSAgent.algorithm_name: {
                "sim_policy_bias": [0.1, 0.25, 0.5, 1, 2.5, 5, 10, 25]
            },
        }
        hyperparam_grids = {}
        for f in self.objective_functions:
            hyperparam_grids[f.name] = deepcopy(hyperparam_grid_base)
        return hyperparam_grids

class ReductionExperimentConditions(ExperimentConditions):
    def __init__(self, base_n, possible_edge_percentage, train_individually, restriction_mechanism):
        super().__init__(base_n, possible_edge_percentage, train_individually, restriction_mechanism)

        self.agents_baseline = {
            LargestComponentSizeTargeted.name: [],
            GlobalEfficiency.name: [],
        }

        self.agents_planning = [
            AR80RandMCTSAgent,
            AR60RandMCTSAgent,
            AR40RandMCTSAgent,
            AR40DegreeMCTSAgent,
            AR40InvDegreeMCTSAgent,
            AR40LBHBMCTSAgent,
            AR40NumConnsMCTSAgent,
            AR40BestEdgeMCTSAgent,
            AR40BestEdgeCSMCTSAgent,
            AR40AvgEdgeMCTSAgent,
            AR40AvgEdgeCSMCTSAgent
        ]

        self.experiment_params['model_seeds'] = [self.get_model_seed(run_num) for run_num in
                                                 range(self.experiment_params['num_runs'])]

        self.hyperparam_grids = self.create_hyperparam_grids()

        # Can be used to skip some parameter combinations by their int id
        self.parameter_combs_to_skip = {
        }

        self.model_seeds_to_skip = {
        }

    def create_hyperparam_grids(self):
        hyperparam_grid_base =  {}
        hyperparam_grids = {}
        for f in self.objective_functions:
            hyperparam_grids[f.name] = deepcopy(hyperparam_grid_base)
        return hyperparam_grids

class SGUCTSynthExperimentConditions(ExperimentConditions):
    def __init__(self, base_n, possible_edge_percentage, train_individually,  restriction_mechanism):
        super().__init__(base_n, possible_edge_percentage, train_individually, restriction_mechanism)

        self.agents_baseline = {
            LargestComponentSizeTargeted.name: [],
            GlobalEfficiency.name: [],
        }

        self.agents_planning = [
            SGUCTAgent
        ]

        self.experiment_params['model_seeds'] = [self.get_model_seed(run_num) for run_num in
                                                 range(self.experiment_params['num_runs'])]

        self.hyperparam_grids = self.create_hyperparam_grids()

        # Can be used to skip some parameter combinations by their int id
        self.parameter_combs_to_skip = {
        }

        self.model_seeds_to_skip = {
        }

    def create_hyperparam_grids(self):
        hyperparam_grid_base =  {
            SGUCTAgent.algorithm_name: {
                "C_p": [0.05, 0.1, 0.25],
                "adjust_C_p": [True],
                "expansion_budget_modifier": [20],
                "rollout_depth": [1],
                "reduction_policy": [AvgEdgeCSReductionPolicy.policy_name]
            },
        }
        hyperparam_grids = {}
        for f in self.objective_functions:
            hyperparam_grids[f.name] = deepcopy(hyperparam_grid_base)
        return hyperparam_grids

class SGUCTRWExperimentConditions(ExperimentConditions):
    def __init__(self, base_n, possible_edge_percentage, train_individually, restriction_mechanism):
        super().__init__(base_n, possible_edge_percentage, train_individually, restriction_mechanism)

        self.agents_baseline = {
            LargestComponentSizeTargeted.name: [
                MinCostAgent,
                RandomAgent,
                GreedyAgent,
                CostSensitiveGreedyAgent,
                LowestDegreeProductAgent,
                FiedlerVectorAgent,
                EffectiveResistanceAgent
            ],
            GlobalEfficiency.name: [
                MinCostAgent,
                RandomAgent,
                LBHBAgent,
            ],
        }

        self.agents_planning = [
            StandardMCTSAgent,
            SGUCTAgent
        ]

        self.experiment_params = {'train_graphs': 0,
                                  'validation_graphs': 1,
                                  'test_graphs': 1,
                                  'num_runs': 10}

        self.experiment_params['model_seeds'] = [self.get_model_seed(run_num) for run_num in
                                                 range(self.experiment_params['num_runs'])]

        self.hyperparam_grids = self.create_hyperparam_grids()

        # Can be used to skip some parameter combinations by their int id
        self.parameter_combs_to_skip = {
        }

        self.model_seeds_to_skip = {
        }

        self.network_generators = [
            GeometricInternetTopologyNetworkGenerator,
            GeometricMetroNetworkGenerator
        ]

    def create_hyperparam_grids(self):
        hyperparam_grid_base =  {
            StandardMCTSAgent.algorithm_name: {
                "C_p": [0.05, 0.1, 0.25],
                "adjust_C_p": [True],
                "expansion_budget_modifier": [20],
                "rollout_depth": [1],
            },

            SGUCTAgent.algorithm_name: {
                "C_p": [0.05, 0.1, 0.25],
                "adjust_C_p": [True],
                "expansion_budget_modifier": [20],
                "rollout_depth": [1],
                "reduction_policy":  [AvgEdgeCSReductionPolicy.policy_name]
            },
        }
        hyperparam_grids = {}
        for f in self.objective_functions:
            hyperparam_grids[f.name] = deepcopy(hyperparam_grid_base)
        return hyperparam_grids

class TimingsExperimentConditions(ExperimentConditions):
    def __init__(self, base_n, possible_edge_percentage, train_individually, restriction_mechanism):
        super().__init__(base_n, possible_edge_percentage, train_individually, restriction_mechanism)

        self.agents_baseline = {
            LargestComponentSizeTargeted.name: [
                GreedyAgent,
                CostSensitiveGreedyAgent,
            ],
            GlobalEfficiency.name: [
                GreedyAgent,
                CostSensitiveGreedyAgent,
            ],
        }

        self.agents_planning = [
            StandardMCTSAgent,
            SGUCTAgent
        ]

        self.experiment_params = {'train_graphs': 0,
                                  'validation_graphs': 25,
                                  'test_graphs': 25,
                                  'num_runs': 1
                                  }

        self.experiment_params['model_seeds'] = [self.get_model_seed(run_num) for run_num in
                                                 range(self.experiment_params['num_runs'])]

        self.hyperparam_grids = self.create_hyperparam_grids()



        # Can be used to skip some parameter combinations by their int id
        self.parameter_combs_to_skip = {
        }

        self.model_seeds_to_skip = {
        }

    def create_hyperparam_grids(self):
        hyperparam_grid_base =  {
            StandardMCTSAgent.algorithm_name: {
                "C_p": [0.05],
                "adjust_C_p": [True],
                "expansion_budget_modifier": [20],
                "rollout_depth": [1]
            },

            SGUCTAgent.algorithm_name: {
                "C_p": [0.05],
                "adjust_C_p": [True],
                "expansion_budget_modifier": [20],
                "rollout_depth": [1],
            },
        }
        hyperparam_grids = {}
        for f in self.objective_functions:
            hyperparam_grids[f.name] = deepcopy(hyperparam_grid_base)
        return hyperparam_grids

class TimingsRWExperimentConditions(ExperimentConditions):
    def __init__(self, base_n, possible_edge_percentage, train_individually, restriction_mechanism):
        super().__init__(base_n, possible_edge_percentage, train_individually, restriction_mechanism)

        self.agents_baseline = {
            LargestComponentSizeTargeted.name: [
                MinCostAgent,
                RandomAgent,
                GreedyAgent,
                CostSensitiveGreedyAgent,
                LowestDegreeProductAgent,
                FiedlerVectorAgent,
                EffectiveResistanceAgent
            ],
            GlobalEfficiency.name: [
                MinCostAgent,
                RandomAgent,
                GreedyAgent,
                CostSensitiveGreedyAgent,
                LBHBAgent,
            ],
        }

        self.agents_planning = [
            StandardMCTSAgent,
            SGUCTAgent
        ]

        self.experiment_params = {'train_graphs': 0,
                                  'validation_graphs': 1,
                                  'test_graphs': 1,
                                  'num_runs': 1
                                  }

        self.experiment_params['model_seeds'] = [self.get_model_seed(run_num) for run_num in
                                                 range(self.experiment_params['num_runs'])]

        self.hyperparam_grids = self.create_hyperparam_grids()



        # Can be used to skip some parameter combinations by their int id
        self.parameter_combs_to_skip = {
        }

        self.model_seeds_to_skip = {
        }

        self.network_generators = [
            GeometricInternetTopologyNetworkGenerator,
            GeometricMetroNetworkGenerator
        ]

    def create_hyperparam_grids(self):
        hyperparam_grid_base =  {
            StandardMCTSAgent.algorithm_name: {
                "C_p": [0.05],
                "adjust_C_p": [True],
                "expansion_budget_modifier": [20],
                "rollout_depth": [1]
            },

            SGUCTAgent.algorithm_name: {
                "C_p": [0.05],
                "adjust_C_p": [True],
                "expansion_budget_modifier": [20],
                "rollout_depth": [1],
            },
        }
        hyperparam_grids = {}
        for f in self.objective_functions:
            hyperparam_grids[f.name] = deepcopy(hyperparam_grid_base)
        return hyperparam_grids



def get_conditions_for_experiment(which, base_n, possible_edge_percentage, train_individually, restriction_mechanism):

    if which == 'prelim':
        cond = PreliminaryExperimentConditions(base_n, possible_edge_percentage, train_individually, restriction_mechanism)
    elif which == 'btm':
        cond = BTMExperimentConditions(base_n, possible_edge_percentage, train_individually, restriction_mechanism)
    elif which == 'mincost':
        cond = MinCostExperimentConditions(base_n, possible_edge_percentage, train_individually, restriction_mechanism)
    elif which == 'reduction':
        cond = ReductionExperimentConditions(base_n,possible_edge_percentage, train_individually, restriction_mechanism)
    elif which == 'sg_uct_synth':
        cond = SGUCTSynthExperimentConditions(base_n, possible_edge_percentage, train_individually, restriction_mechanism)
    elif which == 'sg_uct_rw':
        cond = SGUCTRWExperimentConditions(base_n, possible_edge_percentage, train_individually, restriction_mechanism)
    elif which == 'timings':
        cond = TimingsExperimentConditions(base_n, possible_edge_percentage, train_individually, restriction_mechanism)
    elif which == 'timings_rw':
        cond = TimingsRWExperimentConditions(base_n, possible_edge_percentage, train_individually, restriction_mechanism)
    else:
        raise ValueError(f"experiment {which} not recognised")

    return cond