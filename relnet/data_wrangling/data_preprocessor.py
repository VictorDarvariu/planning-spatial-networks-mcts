import json
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path

import networkx as nx
import numpy as np
from pyproj import Proj, transform


class DataPreprocessor(ABC):
    RAW_DATA_DIR_NAME = 'raw_data'
    CLEANED_DATA_DIR_NAME = 'cleaned_data'
    PROCESSED_DATA_DIR_NAME = 'processed_data'
    DATASET_METADATA_FILE_NAME = 'dataset_metadata.json'

    MIN_NETWORK_SIZE = 100
    MAX_NETWORK_SIZE = 150


    CANONICAL_LAT_ATTR_NAME = "lat"
    CANONICAL_LON_ATTR_NAME = "lon"

    CANONICAL_X_COORD_ATTR_NAME = "pos_x"
    CANONICAL_Y_COORD_ATTR_NAME = "pos_y"

    ENFORCE_CONNECTEDNESS = False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        web_mercator_proj = Proj(init='epsg:3857')
        wgs84_proj = Proj(init='epsg:4326')

    def __init__(self, root_dir_string):
        self.root_dir = Path(root_dir_string)
        self.raw_data_dir = self.root_dir / self.RAW_DATA_DIR_NAME
        self.cleaned_data_dir = self.root_dir / self.CLEANED_DATA_DIR_NAME
        self.processed_data_dir = self.root_dir / self.PROCESSED_DATA_DIR_NAME

        for d in [self.raw_data_dir, self.cleaned_data_dir, self.processed_data_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.raw_dataset_dir = self.raw_data_dir / self.DS_NAME
        self.cleaned_dataset_dir = self.cleaned_data_dir / self.DS_NAME
        self.processed_dataset_dir = self.processed_data_dir / self.DS_NAME

        for d in [self.raw_dataset_dir, self.cleaned_dataset_dir, self.processed_dataset_dir]:
            d.mkdir(parents=True, exist_ok=True)


    def execute_task(self, task, **kwargs):
        if task == "clean":
            self.clean_data(**kwargs)
        elif task == "process":
            self.process_data(**kwargs)

    def check_connectedness(self, G):
        return nx.is_connected(G)

    def check_sizes(self, G):
        return len(G) >= self.MIN_NETWORK_SIZE and len(G) <= self.MAX_NETWORK_SIZE


    def merge_identical_nodes(self, G):
        lat = nx.get_node_attributes(G, self.CANONICAL_LAT_ATTR_NAME)
        lon = nx.get_node_attributes(G, self.CANONICAL_LON_ATTR_NAME)

        lats = []
        lons = []
        for node in G:
            node_lat = lat[node]
            node_lon = lon[node]

            lats.append(node_lat)
            lons.append(node_lon)

        coords = np.hstack([np.array(lats).reshape(-1, 1), np.array(lons).reshape(-1, 1)])
        unq, count = np.unique(coords, axis=0, return_counts=True)
        repeated_groups = unq[count > 1]

        for repeated_group in repeated_groups:
            repeated_idx = np.argwhere(np.all(coords == repeated_group, axis=1))
            all_rep = repeated_idx.ravel()

            kept_node = all_rep[0]
            for other_node in all_rep[1:]:
                other_node_edges = deepcopy(list(G.edges(other_node)))
                G.remove_node(other_node)

                for edge in other_node_edges:
                    edge_end = edge[1]
                    if kept_node != edge_end:
                        G.add_edge(kept_node, edge_end)
                        print(f"adding edge {kept_node, edge_end}")

        H = nx.relabel.convert_node_labels_to_integers(G)
        return H


    @staticmethod
    def extract_largest_cc(G):
        largest_cc = max(nx.connected_components(G), key=len)
        lcc_G = G.subgraph(largest_cc).copy()
        lcc_G_relabeled = nx.relabel.convert_node_labels_to_integers(lcc_G)
        return lcc_G_relabeled

    def check_graph_criteria(self, G):
        return self.check_sizes(G) and (self.check_connectedness(G) if self.ENFORCE_CONNECTEDNESS else True)

    @abstractmethod
    def clean_data(self, **kwargs):
        pass

    def process_data(self, include_geom_coords=True):
        cleaned_graph_files = sorted(self.cleaned_dataset_dir.glob("*.graphml"))
        dataset_metadata = {}

        num_graphs = 0
        graph_names = []
        graph_props = []

        for i, f in enumerate(cleaned_graph_files):
            cleaned_G = nx.readwrite.read_graphml(f.resolve())
            if include_geom_coords:
                self.project_coords_to_unit_square(cleaned_G)
            else:
                self.remove_location_attrs(cleaned_G)

            processed_filepath = self.processed_dataset_dir / f.name
            nx.readwrite.write_graphml(cleaned_G, processed_filepath.resolve())

            graph_name = f.stem
            num_graphs += 1
            graph_names.append(graph_name)

            props = {}
            props['num_nodes'] = len(cleaned_G.nodes)
            props['num_edges'] = len(cleaned_G.edges)
            graph_props.append(props)

        dataset_metadata['num_graphs'] = num_graphs
        dataset_metadata['graph_names'] = graph_names
        dataset_metadata['graph_props'] = graph_props

        metadata_filename = self.processed_dataset_dir / self.DATASET_METADATA_FILE_NAME

        with open(metadata_filename.resolve(), "w") as fp:
            json.dump(dataset_metadata, fp, indent=4)

    def project_coords_to_unit_square(self, cleaned_G):
        lat = nx.get_node_attributes(cleaned_G, self.CANONICAL_LAT_ATTR_NAME)
        lon = nx.get_node_attributes(cleaned_G, self.CANONICAL_LON_ATTR_NAME)

        xs = []
        ys = []
        for node in cleaned_G:
            node_lat = lat[node]
            node_lon = lon[node]
            pos_x, pos_y = transform(self.wgs84_proj, self.web_mercator_proj, node_lon, node_lat)

            xs.append(pos_x)
            ys.append(pos_y)
        x_attrs, y_attrs = self.normalize_euclidean_coords(cleaned_G, xs, ys)

        for (n, d) in cleaned_G.nodes(data=True):
            d.clear()
        nx.set_node_attributes(cleaned_G, x_attrs, self.CANONICAL_X_COORD_ATTR_NAME)
        nx.set_node_attributes(cleaned_G, y_attrs, self.CANONICAL_Y_COORD_ATTR_NAME)

    def normalize_euclidean_coords(self, G, xs, ys, int_keys=False):
        x_coords = np.array(xs)
        y_coords = np.array(ys)
        min_x, max_x = np.min(x_coords), np.max(x_coords)
        min_y, max_y = np.min(y_coords), np.max(y_coords)
        rescaled_xs = (x_coords - min_x) / (max_x - min_x)
        rescaled_ys = (y_coords - min_y) / (max_y - min_y)
        node_ids = [int(node) for node in G]
        x_attrs = {(str(nid) if not int_keys else nid): rescaled_xs[nid] for nid in node_ids}
        y_attrs = {(str(nid) if not int_keys else nid): rescaled_ys[nid] for nid in node_ids}
        return x_attrs, y_attrs

    def remove_location_attrs(self, cleaned_G):
        self.remove_node_attrs(cleaned_G, self.CANONICAL_LAT_ATTR_NAME)
        self.remove_node_attrs(cleaned_G, self.CANONICAL_LON_ATTR_NAME)

    def partition_graph_by_attribute(self, G, attr, attr_possible_values):
        partitioned_graphs = {}
        for attr_value in attr_possible_values:
            nodes = (
                node
                for node, data
                in G.nodes(data=True)
                if data.get(attr) == attr_value
            )
            attr_subgraph = G.subgraph(nodes).copy()

            self.remove_node_attrs(attr_subgraph, attr)

            H = nx.relabel.convert_node_labels_to_integers(attr_subgraph)
            partitioned_graphs[attr_value] = H
        return partitioned_graphs

    def remove_node_attrs(self, G, attr):
        for (n, d) in G.nodes(data=True):
            del d[attr]

    def remove_edge_attrs(self, G, attr):
        for (t, f, d) in G.edges(data=True):
            del d[attr]

    def check_and_write_subgraph(self, country_code, subgraph):
        cleaned_filepath = self.cleaned_dataset_dir / f"{country_code}.graphml"
        is_connected = self.check_connectedness(subgraph)

        can_write = False
        if is_connected:
            if self.check_sizes(subgraph):
                can_write = True
        else:
            lcc = self.extract_largest_cc(subgraph)
            if self.check_sizes(lcc):
                can_write = True
                subgraph = lcc

        if can_write:
            nx.readwrite.write_graphml(subgraph, cleaned_filepath.resolve())