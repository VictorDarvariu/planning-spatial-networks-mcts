from relnet.objective_functions.objective_functions_ext import *

from relnet.state.graph_state import get_graph_hash, GeometricRelnetGraph


def extract_kwargs(s2v_graph, kwargs):
    num_mc_sims = 20
    random_seed = 42
    if 'mc_sims_multiplier' in kwargs:
        num_mc_sims = int(s2v_graph.num_nodes * kwargs['mc_sims_multiplier'])
    if 'random_seed' in kwargs:
        random_seed = kwargs['random_seed']
    return num_mc_sims, random_seed


class LargestComponentSizeTargeted(object):
    name = "lcs_targeted"
    upper_limit = 0.5

    @staticmethod
    def compute(s2v_graph, **kwargs):
        num_mc_sims, random_seed = extract_kwargs(s2v_graph, kwargs)
        N, M, edges = s2v_graph.num_nodes, s2v_graph.num_edges, s2v_graph.edge_pairs
        graph_hash = get_graph_hash(s2v_graph)
        lcs = size_largest_component_targeted(N, M, edges, num_mc_sims, graph_hash, random_seed)
        return lcs

class GlobalEfficiency(object):
    name = "global_eff"
    upper_limit = 1

    @staticmethod
    def compute(s2v_graph, **kwargs):
        if type(s2v_graph) != GeometricRelnetGraph:
            raise ValueError("cannot compute efficiency for non-geometric graphs!")

        N, M, edges, edge_lengths = s2v_graph.num_nodes, s2v_graph.num_edges, s2v_graph.edge_pairs, s2v_graph.get_edge_lengths_as_arr()
        pairwise_dists = s2v_graph.get_all_pairwise_distances()
        eff = global_efficiency(N, M, edges, edge_lengths, pairwise_dists)

        if eff == -1.:
            eff = 0.
        return eff