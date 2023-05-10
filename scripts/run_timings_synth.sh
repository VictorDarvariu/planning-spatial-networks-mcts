#! /bin/bash

gsize=$1
docker exec -d relnet-manager /bin/bash -c "source activate relnet-cenv; python /relnet/run_experiments.py --experiment_part both --which timings --base_n $gsize --algorithm_class planning  --edge_percentage 10 --experiment_id timingsv2_kh_$gsize --force_insert_details --parallel_eval --train_individually"


