from mpi4py import MPI

from stable_baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import logger
from stable_baselines.trpo_mpi import TRPO
from stable_baselines.ppo1 import PPO1
import stable_baselines.common.tf_util as tf_util

import os
import time


def train(env_id, num_timesteps, seed, algorithm, model_save_file=None, log_dir=None):

    with tf_util.single_threaded_session():
        logger.configure(folder=log_dir, format_strs=['stdout', 'log', 'csv'])

        workerseed = seed + MPI.COMM_WORLD.Get_rank()
        env = make_mujoco_env(env_id, workerseed)

        if algorithm == "TRPO":
            model = TRPO(MlpPolicy, env, seed=workerseed, verbose=1)
        else:
            # Algorithm is PPO
            model = PPO1(MlpPolicy, env, seed=workerseed, verbose=1)

        model.learn(total_timesteps=num_timesteps)

        if model_save_file is not None:
            model.save(model_save_file)

        env.close()


def main():

    main_dir = "simulation_results"
    try:
        os.mkdir(main_dir)
    except FileExistsError:
        pass

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        save_dir = [os.path.join(main_dir, time.strftime('%Y_%m_%d-%Hh_%Mm_%Ss', time.localtime(time.time())))]
    else:
        save_dir = None

    save_dir = comm.bcast(save_dir, root=0)

    # Unpack list
    save_dir = save_dir[0]

    parser = mujoco_arg_parser()
    parser.add_argument('--algorithm', help="The algorithm which shall be used, TRPO or PPO", type=str, default="TRPO")
    args = parser.parse_args()

    model_file = os.path.join(save_dir, "model")
    log_dir = os.path.join(save_dir, "log")

    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, algorithm=args.algorithm,
          model_save_file=model_file, log_dir=log_dir)


if __name__ == '__main__':
    main()
