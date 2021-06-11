#! coding='utf-8'

import argparse
import numpy as np
from numpy.lib.function_base import append
import tensorflow as tf
import time
import pickle
import copy

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=300000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--group-dim", type=int, default=1, help="number of group")
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

def mlp_model(input, num_outputs, scope, reuse=False, num_units=128, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def make_env(scenario_name, benchmark, agent_states, landmark_states):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world(copy.deepcopy(agent_states), copy.deepcopy(landmark_states))
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation) # shared_viewer=False 各自第一视角
    return env

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers


def train(arglist, agent_states, landmark_states):
    env = list()
    # trainers = list()

    with U.make_session(8):
        for g in range(arglist.group_dim):
            # Create environment
            env.append(make_env(arglist.scenario, arglist.benchmark, agent_states[g], landmark_states[g]))
        
        # Create agent trainers
        obs_shape_n = [env[0].observation_space[i].shape for i in range(env[0].n)]
        num_adversaries = min(env[0].n, arglist.num_adversaries)
        trainers = get_trainers(env[0], num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        obs_n = list()
        for g in range(arglist.group_dim):
            obs_n.append(env[g].reset(copy.deepcopy(agent_states[g]), copy.deepcopy(landmark_states[g])))
        episode_step = 0

        print('Starting iterations...')
        while True:
            episode_step += 1
            entities_group = list()

            for g in range(arglist.group_dim):
                # get action
                action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n[g])]
                print("action_n: {}".format(action_n))
                # environment step
                new_obs_n, rew_n, done_n, info_n = env[g].step(action_n)
                done = all(done_n)
                terminal = (episode_step >= arglist.max_episode_len)
                print(episode_step)
                # collect experience
                for i, agent in enumerate(trainers):
                    agent.experience(obs_n[g][i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
                obs_n[g] = new_obs_n

                entities_group.extend(env[g].world.entities)

                # 重复训练，运行时可以关掉
                if done or terminal or episode_step == 0:
                    obs_n[g] = env[g].reset(copy.deepcopy(agent_states[g]), copy.deepcopy(landmark_states[g]))
                    episode_step = 0

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env[0].render(entities_group)

if __name__ == '__main__':

    agent_states = [[[np.array([0.1, 0.1]), np.array([0., 0.])], [np.array([0.3, 0.3]), np.array([0., 0.])], 
                    [np.array([0.5, 0.5]), np.array([0., 0.])], [np.array([0.7, 0.7]), np.array([0., 0.])]],
                    [[np.array([0.2, 0.2]), np.array([0., 0.])], [np.array([0.4, 0.4]), np.array([0., 0.])], 
                    [np.array([0.6, 0.6]), np.array([0., 0.])], [np.array([0.8, 0.8]), np.array([0., 0.])]]]
    landmark_states = [[[np.array([-0.1, -0.1]), np.array([0., 0.])], [np.array([-0.3, -0.3]), np.array([0., 0.])], 
                    [np.array([-0.5, -0.5]), np.array([0., 0.])], [np.array([-0.7, -0.7]), np.array([0., 0.])]],
                    [[np.array([-0.2, -0.2]), np.array([0., 0.])], [np.array([-0.4, -0.4]), np.array([0., 0.])], 
                    [np.array([-0.6, -0.6]), np.array([0., 0.])], [np.array([-0.8, -0.8]), np.array([0., 0.])]]]

    arglist = parse_args()
    train(arglist, agent_states, landmark_states)
