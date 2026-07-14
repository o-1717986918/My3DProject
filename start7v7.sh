#!/bin/bash
export OMP_NUM_THREADS=1

host=${1:-localhost}
port=${2:-60000}

for i in {1..7}; do
  python3 run_player.py --host $host --port $port -n $i -t MujocoCodebase -f my_field &
done

for i in {1..7}; do
  python3 run_player.py --host $host --port $port -n $i -t MujocoCodebase2 -f my_field &
done

# 等所有agent连上并beam完毕，然后触发开球
sleep 10
python3 -c "
import socket, struct
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('$host', 60001))
msg = '(kickOff)'
s.send(struct.pack('>I', len(msg)) + msg.encode())
s.close()
print('KickOff triggered!')
"
