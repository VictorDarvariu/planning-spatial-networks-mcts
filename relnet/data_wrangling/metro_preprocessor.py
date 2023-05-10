import networkx as nx

from relnet.data_wrangling.data_preprocessor import DataPreprocessor


class MetroDataPreprocessor(DataPreprocessor):
    DS_NAME = "metro"

    def clean_data(self, **kwargs):
        adjacency_files = self.extract_adjacency_files()

        for city_name, adj_file in adjacency_files.items():
            print(f"doing city name {city_name}")
            G = nx.Graph()

            edges_to_add = []
            with open(adj_file.resolve(), "r", encoding=('utf-8' if city_name != "Madrid" else 'iso-8859-1')) as fh:
                next(fh)
                next(fh)

                hit_edges = False
                for raw_line in fh:
                    line = raw_line.strip()
                    if line == "*Edges":
                        hit_edges = True
                        continue

                    data_elems = line.split(" ")
                    if not hit_edges:
                        nid, lat, lon = int(data_elems[0]), float(data_elems[2]), float(data_elems[3])
                        G.add_node(nid, lat=lat, lon=lon)
                    else:
                        edge_from, edge_to = int(data_elems[0]), int(data_elems[1])
                        if edge_from in G and edge_to in G:
                            edges_to_add.append((edge_from, edge_to))

            G.add_edges_from(edges_to_add)
            H = nx.relabel.convert_node_labels_to_integers(G)
            H = self.merge_identical_nodes(H)

            if not nx.is_connected(H):
                H = self.extract_largest_cc(H)
            if self.check_graph_criteria(H):
                cleaned_filepath = self.cleaned_dataset_dir / f"{city_name}.graphml"
                nx.readwrite.write_graphml(H, cleaned_filepath.resolve())


    def extract_adjacency_files(self):
        latest_adjacency_files = {}
        all_dirs = sorted([dir for dir in self.raw_dataset_dir.iterdir() if not dir.name.startswith(".")])
        for dir in all_dirs:
            city_name = dir.name
            topologies_dirs = list(dir.glob("*-topologies"))
            if len(topologies_dirs) == 0:
                print(f"skipping {city_name} since no topology directory exists.")
                continue
            else:
                topology_dir = topologies_dirs[0]
                adjacency_files = list(topology_dir.glob("*-adjacency.net"))
                latest_adjacency_file = sorted(adjacency_files)[-1]

                latest_adjacency_files[city_name] = latest_adjacency_file

        return latest_adjacency_files



