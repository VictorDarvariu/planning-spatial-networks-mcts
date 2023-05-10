#! /bin/bash

docker exec -d relnet-manager /bin/bash -c "source activate relnet-cenv; python /relnet/run_experiments.py --experiment_part both --which timings_rw --base_n -1 --algorithm_class planning  --edge_percentage 10 --experiment_id timingsv2_rw --force_insert_details --parallel_eval --train_individually"


