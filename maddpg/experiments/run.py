#! coding='utf-8'

import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import copy

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers

agent_states = [[[np.array([0.1, 0.1]), np.array([0., 0.])], [np.array([0.3, 0.3]), np.array([0., 0.])], 
                [np.array([0.5, 0.5]), np.array([0., 0.])], [np.array([0.7, 0.7]), np.array([0., 0.])]],
                [[np.array([0.2, 0.2]), np.array([0., 0.])], [np.array([0.4, 0.4]), np.array([0., 0.])], 
                [np.array([0.6, 0.6]), np.array([0., 0.])], [np.array([0.8, 0.8]), np.array([0., 0.])]]]
landmark_states = [[[np.array([-0.1, -0.1]), np.array([0., 0.])], [np.array([-0.3, -0.3]), np.array([0., 0.])], 
                   [np.array([-0.5, -0.5]), np.array([0., 0.])], [np.array([-0.7, -0.7]), np.array([0., 0.])]],
                   [[np.array([-0.2, -0.2]), np.array([0., 0.])], [np.array([-0.4, -0.4]), np.array([0., 0.])], 
                   [np.array([-0.6, -0.6]), np.array([0., 0.])], [np.array([-0.8, -0.8]), np.array([0., 0.])]]]

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=300000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=128, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()

if __name__ == '__main__':
    group_dim = 2
    from run_group import train
    arglist = parse_args()
    for g in range(group_dim):
        train(arglist, agent_states[g], landmark_states[g])
