import argparse
import sys

sys.path.append('/relnet')

from relnet.data_wrangling.internet_topology_preprocessor import InternetTopologyDataPreprocessor
from relnet.data_wrangling.metro_preprocessor import MetroDataPreprocessor

supported_datasets = ["internet_topology", "metro"]

def process_dataset(dataset, task, root_dir, additional_args):
    preprocessor_class = get_preprocessor_for_dataset(dataset)
    preprocessor = preprocessor_class(root_dir)
    preprocessor.execute_task(task, **additional_args)

def get_preprocessor_for_dataset(dataset):
    ds_preprocessors = {InternetTopologyDataPreprocessor.DS_NAME: InternetTopologyDataPreprocessor,
                        MetroDataPreprocessor.DS_NAME: MetroDataPreprocessor,
                        }

    return ds_preprocessors[dataset]


def main():
    parser = argparse.ArgumentParser(description="Script to process raw real-world network data into canonical format.")
    parser.add_argument("--dataset", required=True, type=str,
                        help="Dataset to process.",
                        choices=supported_datasets + ["all"])

    parser.add_argument("--task", required=True, type=str,
                        help="Task to execute.",
                        choices=["clean", "process"])

    parser.add_argument("--root_dir", type=str, help="Root path where dataset is located.")
    parser.set_defaults(root_dir="/experiment_data/real_world_graphs")

    parser.add_argument('--include_geom_coords', dest='include_geom_coords', action='store_true')
    parser.set_defaults(include_geom_coords=False)

    args = parser.parse_args()

    additional_args = {}
    additional_args['include_geom_coords'] = True if args.include_geom_coords else False
    print(additional_args)

    dataset = args.dataset
    if dataset == "all":
        for ds in supported_datasets:
            process_dataset(ds, args.task, args.root_dir, additional_args)
    else:
        process_dataset(dataset,args.task,  args.root_dir, additional_args)

if __name__ == "__main__":
    main()