from xml.etree import ElementTree as ET

import networkx as nx

from relnet.data_wrangling.data_preprocessor import DataPreprocessor


class InternetTopologyDataPreprocessor(DataPreprocessor):
    DS_NAME = "internet_topology"

    def clean_data(self, **kwargs):
        graph_files = sorted(self.raw_dataset_dir.glob("*.graphml"))
        cleaned_files = 0

        for f in graph_files:
            original_G = nx.readwrite.read_graphml(f.resolve())
            G = self.select_node_subset(original_G)
            if G is not None:
                lat = nx.get_node_attributes(G, 'Latitude')
                lon = nx.get_node_attributes(G, 'Longitude')

                for (n, d) in G.nodes(data=True):
                    d.clear()

                for (n1, n2, d) in G.edges(data=True):
                    d.clear()

                nx.set_node_attributes(G, lat, self.CANONICAL_LAT_ATTR_NAME)
                nx.set_node_attributes(G, lon, self.CANONICAL_LON_ATTR_NAME)
                G = self.merge_identical_nodes(G)

                cleaned_filepath = self.cleaned_dataset_dir / f.name
                nx.readwrite.write_graphml(G, cleaned_filepath.resolve())
                self.remove_unnecessary_attributes(cleaned_filepath.resolve())

                cleaned_files += 1

        print(f"cleaned dataset contains {cleaned_files} files.")

    def remove_unnecessary_attributes(self, resolved_path):
        tree = ET.parse(resolved_path)
        ns = 'http://graphml.graphdrawing.org/xmlns'
        ET.register_namespace('', ns)
        root = tree.getroot()

        for key_element in root.findall(f'{{{ns}}}key'):
            if key_element.attrib['attr.name'] not in [self.CANONICAL_LON_ATTR_NAME, self.CANONICAL_LAT_ATTR_NAME]:
                root.remove(key_element)

        graph_element = list(root)[-1]
        for data_element in graph_element.findall(f'{{{ns}}}data'):
            graph_element.remove(data_element)

        tree.write(resolved_path)

    def select_node_subset(self, G):
        # first, find the nodes that don't have any lat/lon information
        lat = nx.get_node_attributes(G, 'Latitude')
        lon = nx.get_node_attributes(G, 'Longitude')

        node_set = set(G.nodes)
        lat_exists_set = set(lat.keys())
        lon_exists_set = set(lon.keys())
        both_exist_set = lat_exists_set.intersection(lon_exists_set)

        missing_coord_nodes = node_set - both_exist_set

        G.remove_nodes_from(missing_coord_nodes)
        H = nx.relabel.convert_node_labels_to_integers(G)
        try:
            if not nx.is_connected(H):
                H = self.extract_largest_cc(H)
        except nx.NetworkXPointlessConcept:
            return None

        if self.check_graph_criteria(H):
            return H
        return None
