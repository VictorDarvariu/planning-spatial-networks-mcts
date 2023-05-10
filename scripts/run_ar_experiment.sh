#! /bin/bash

docker exec -d relnet-worker-cpu /bin/bash -c "source activate relnet-cenv ; python relnet/experiment_launchers/run_ar_experiment.py"