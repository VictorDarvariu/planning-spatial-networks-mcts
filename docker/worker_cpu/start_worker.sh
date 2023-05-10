#! /bin/bash
source activate relnet-cenv

cd /relnet/relnet/objective_functions && make
cd /relnet

celery -A tasks worker -Ofair --loglevel=debug --without-gossip --without-mingle &
tail -f /dev/null
