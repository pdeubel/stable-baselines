import os

os.system('mpirun --use-hwthread-cpus -np 40 python3 -m mpi-algorithms --env="CartPole-v1" --seed=0 --num-timesteps=100000 --algorithm="TRPO"')
