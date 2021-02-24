#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_clone.py experts/Humanoid-v1.pkl Humanoid-v1

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow.compat.v1 as tf
import numpy as np
import tf_util
import gym
import load_policy
from keras.models import Sequential
from keras.layers import Dense

tf.disable_v2_behavior()

def env_dims(env):
    return env.observation_space.shape[0], env.action_space.shape[0]

class Network:
    def __init__(self, env):
        input_len, ouput_len = env_dims(env)

        self.model = Sequential()
        self.model.add(Dense(64, input_dim=input_len, activation='relu'))
        self.model.add(Dense(ouput_len))

        self.model.compile(loss='mse', optimizer='sgd')

    def train(self, observations, actions, epochs, batch_size, verbose):
        self.model.fit(observations, actions,
                        batch_size = batch_size,
                        epochs = epochs,
                        verbose=verbose)

    def act(self, obs):
        obs_batch = np.expand_dims(obs,0)
        act_batch = self.model.predict_on_batch(obs_batch)
        return np.ndarray.flatten(act_batch)

def generate_rollouts(env, policy, max_steps, num_rollouts, render, verbose):

    returns = []
    observations = []
    actions = []
    for i in range(num_rollouts):
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = policy.act(obs)
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if render:
                env.render()
            if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

    print('Return summary: mean=%f, std=%f' % (np.mean(returns), np.std(returns)))
    return np.array(observations), np.array(actions)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=50,
                        help='Number of expert roll outs')
    parser.add_argument('--verbose', type=int, choices=[0,1,2], default=1,
                        help='Verbose')
    args = parser.parse_args()
    env = gym.make(args.envname)

    max_steps = args.max_timesteps or env.spec.timestep_limit

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()
        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            # print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(action[0])
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                # if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}

    student = Network(env)
 
    print("Behavior Cloning....")
    student.train(expert_data['observations'], expert_data['actions'], 1000, 128, args.verbose)

    print("Generating rollouts from new model..")
    generate_rollouts(env, student, max_steps, args.num_rollouts, args.render, args.verbose)


if __name__ == '__main__':
    main()
