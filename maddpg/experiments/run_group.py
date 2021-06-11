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
    with U.make_session(8):
        # Create environment
        env = make_env(arglist.scenario, arglist.benchmark, agent_states, landmark_states)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        obs_n = env.reset(copy.deepcopy(agent_states), copy.deepcopy(landmark_states))
        episode_step = 0

        print('Starting iterations...')
        while True:
            episode_step += 1
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
            print("action_n: {}".format(action_n))
            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            print(episode_step)
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n

            # 重复训练，运行时可以关掉
            if done or terminal:
                obs_n = env.reset(copy.deepcopy(agent_states), copy.deepcopy(landmark_states))
                episode_step = 0

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render(env.world.entities)
                continue
