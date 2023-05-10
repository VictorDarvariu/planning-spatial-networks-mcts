import math
from abc import ABC

import networkx as nx
import numpy as np

from relnet.state import networkx_extension as nx_ext
from relnet.state.graph_state import GeometricRelnetGraph
from relnet.state.network_generators import NetworkGenerator, InternetTopologyNetworkGenerator, MetroNetworkGenerator
from relnet.utils.config_utils import local_seed


class GeometricNetworkGenerator(NetworkGenerator, ABC):
    def post_generate_instance(self, instance):
        pos_x = nx.get_node_attributes(instance, 'pos_x')
        pos_y = nx.get_node_attributes(instance, 'pos_y')
        pos = {}
        for node, x in pos_x.items():
            pos[node] = (x, pos_y[node])
        node_positions = np.array([p[1] for p in sorted(pos.items(), key=lambda x: x[0])], dtype=np.float32)

        state = GeometricRelnetGraph(instance, node_positions)
        return state

    def rem_edges_if_needed(self, gen_params, nx_graph, random_seed):
        if 'rem_edges_prop' in gen_params:
            edges = list(nx_graph.edges())
            n_to_rem = math.floor(gen_params['rem_edges_prop'] * len(edges))

            with local_seed(random_seed):
                edges_to_rem_idx = np.random.choice(np.arange(len(edges)), n_to_rem, replace=False)
            nx_graph.remove_edges_from([edges[idx] for idx in edges_to_rem_idx])

    def rem_nodes_if_needed(self, gen_params, nx_graph, random_seed):
        if 'rem_nodes_prop' in gen_params:
            nodes = list(nx_graph.nodes())
            n_to_rem = math.floor(gen_params['rem_nodes_prop'] * len(nodes))
            with local_seed(random_seed):
                nodes_to_rem = np.random.choice(nodes, n_to_rem, replace=False)

            edges_to_rem = []
            for node in nodes_to_rem:
                node_edges = list(nx_graph.edges(node))
                edges_to_rem.extend(node_edges)
            nx_graph.remove_edges_from(edges_to_rem)

    def set_position_attributes(self, nx_graph):
        pos = nx.get_node_attributes(nx_graph, 'pos')
        for node, (x, y) in pos.items():
            nx_graph.node[node]['pos_x'] = x
            nx_graph.node[node]['pos_y'] = y
            del nx_graph.node[node]['pos']


class KHNetworkGenerator(GeometricNetworkGenerator):
    name = 'kaiser_hilgetag'
    conn_radius_modifiers = {'range': 1.5,
                            'max_current': 1}

    def generate_instance(self, gen_params, random_seed):
        n = gen_params['n']
        alpha, beta = gen_params['alpha_kh'], gen_params['beta_kh']
        nx_graph = nx_ext.kaiser_hilgetag_graph(n, alpha=alpha, beta=beta, seed=random_seed)
        self.set_position_attributes(nx_graph)
        self.rem_nodes_if_needed(gen_params, nx_graph, random_seed)
        return nx_graph


class GeometricInternetTopologyNetworkGenerator(GeometricNetworkGenerator, InternetTopologyNetworkGenerator):
    conn_radius_modifiers = {'range': 3,
                             'max_current': 2}

class GeometricMetroNetworkGenerator(GeometricNetworkGenerator, MetroNetworkGenerator):
    conn_radius_modifiers = {'range': 3,
                             'max_current': 2}
