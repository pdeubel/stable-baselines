import os

os.system('mpirun --use-hwthread-cpus -np 40 python3 -m mpi-algorithms --env="Hopper-v2" --seed=0 --num-timesteps=15000 --algorithm="TRPO"')
os.system('mpirun --use-hwthread-cpus -np 40 python3 -m mpi-algorithms --env="Hopper-v2" --seed=0 --num-timesteps=15000 --algorithm="PPO1"')

os.system('mpirun --use-hwthread-cpus -np 40 python3 -m mpi-algorithms --env="HalfCheetah-v2" --seed=0 --num-timesteps=15000 --algorithm="TRPO"')
os.system('mpirun --use-hwthread-cpus -np 40 python3 -m mpi-algorithms --env="HalfCheetah-v2" --seed=0 --num-timesteps=15000 --algorithm="PPO1"')

os.system('mpirun --use-hwthread-cpus -np 40 python3 -m mpi-algorithms --env="Walker2d-v2" --seed=0 --num-timesteps=15000 --algorithm="TRPO"')
os.system('mpirun --use-hwthread-cpus -np 40 python3 -m mpi-algorithms --env="Walker2d-v2" --seed=0 --num-timesteps=15000 --algorithm="PPO1"')
