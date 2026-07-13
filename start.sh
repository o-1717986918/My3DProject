#!/bin/bash
export OMP_NUM_THREADS=1

host=${1:-localhost}
port=${2:-60000}

for i in {1..11}; do
  python3 run_player.py --host $host --port $port -n $i -t MujocoCodebase &
done
