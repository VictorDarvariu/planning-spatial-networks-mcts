#! /bin/bash

docker exec -d relnet-manager /bin/bash -c "source activate relnet-cenv; python run_experiments.py --experiment_part both --which sg_uct_rw --base_n -1 --algorithm_class planning  --edge_percentage 10 --experiment_id sg_uct_rw --force_insert_details --train_individually --parallel_eval"