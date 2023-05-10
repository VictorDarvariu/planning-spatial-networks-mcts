#! /bin/bash

### Size 25
docker exec -d relnet-manager /bin/bash -c "source activate relnet-cenv; python run_experiments.py --experiment_part both --which btm --base_n 25 --algorithm_class planning  --edge_percentage 10 --experiment_id btm_kh_25 --bootstrap_hyps_expid prelim_kh_25 --force_insert_details"

docker exec -d relnet-manager /bin/bash -c "source activate relnet-cenv; python run_experiments.py --experiment_part both --which mincost --base_n 25 --algorithm_class planning  --edge_percentage 10 --experiment_id mincost_kh_25 --bootstrap_hyps_expid prelim_kh_25 --force_insert_details"

docker exec -d relnet-manager /bin/bash -c "source activate relnet-cenv; python run_experiments.py --experiment_part both --which reduction --base_n 25 --algorithm_class planning  --edge_percentage 10 --experiment_id reduction_kh_25 --bootstrap_hyps_expid prelim_kh_25 --force_insert_details"

### Size 50
#docker exec -d relnet-manager /bin/bash -c "source activate relnet-cenv; python run_experiments.py --experiment_part both --which btm --base_n 50 --algorithm_class planning  --edge_percentage 10 --experiment_id btm_kh_50 --bootstrap_hyps_expid prelim_kh_50 --force_insert_details"
#
#docker exec -d relnet-manager /bin/bash -c "source activate relnet-cenv; python run_experiments.py --experiment_part both --which mincost --base_n 50 --algorithm_class planning  --edge_percentage 10 --experiment_id mincost_kh_50 --bootstrap_hyps_expid prelim_kh_50 --force_insert_details"
#
#docker exec -d relnet-manager /bin/bash -c "source activate relnet-cenv; python run_experiments.py --experiment_part both --which reduction --base_n 50 --algorithm_class planning  --edge_percentage 10 --experiment_id reduction_kh_50 --bootstrap_hyps_expid prelim_kh_50 --force_insert_details"

### Size 75
#docker exec -d relnet-manager /bin/bash -c "source activate relnet-cenv; python run_experiments.py --experiment_part both --which btm --base_n 75 --algorithm_class planning  --edge_percentage 10 --experiment_id btm_kh_75 --bootstrap_hyps_expid prelim_kh_75 --force_insert_details"
#
#docker exec -d relnet-manager /bin/bash -c "source activate relnet-cenv; python run_experiments.py --experiment_part both --which mincost --base_n 75 --algorithm_class planning  --edge_percentage 10 --experiment_id mincost_kh_75 --bootstrap_hyps_expid prelim_kh_75 --force_insert_details"
#
#docker exec -d relnet-manager /bin/bash -c "source activate relnet-cenv; python run_experiments.py --experiment_part both --which reduction --base_n 75 --algorithm_class planning  --edge_percentage 10 --experiment_id reduction_kh_75 --bootstrap_hyps_expid prelim_kh_75 --force_insert_details"
