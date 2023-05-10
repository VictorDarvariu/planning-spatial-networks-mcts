#! /bin/bash

docker exec -d relnet-manager /bin/bash -c "source activate relnet-cenv; python run_experiments.py --experiment_part both --which prelim --base_n 25 --algorithm_class planning  --edge_percentage 10 --experiment_id prelim_kh_25 --force_insert_details"

#docker exec -d relnet-manager /bin/bash -c "source activate relnet-cenv; python run_experiments.py --experiment_part both --which prelim --base_n 50 --algorithm_class planning  --edge_percentage 10 --experiment_id prelim_kh_50 --force_insert_details"
#
#docker exec -d relnet-manager /bin/bash -c "source activate relnet-cenv; python run_experiments.py --experiment_part both --which prelim --base_n 75 --algorithm_class planning  --edge_percentage 10 --experiment_id prelim_kh_75 --force_insert_details"