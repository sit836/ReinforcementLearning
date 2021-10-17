import numpy as np
from collections import defaultdict

from discrete_limit_env import StudentEnv

"""
Reference: https://github.com/frangipane/reinforcement-learning/blob/master/02-dynamic-programming/student_MDP.ipynb 
"""


class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()[observation]


def sample_mdp(env, agent, episode_count, gamma=1.0):
    episodes = []  # list of lists, trajectories per episode
    rewards_by_state = defaultdict(list)  # total rewards per starting state

    for i in range(episode_count):
        episode = []
        cum_reward = 0

        ob = env.reset()
        start_state = ob
        reward = 0
        step = 0
        done = False
        episode.append(start_state)

        if start_state == 4:
            # started in terminal state
            rewards_by_state[start_state].append(cum_reward)
            episodes.append(episode)
            continue
        while True:
            step += 1
            action = agent.act(ob, reward, done)
            episode.append(action)
            ob, reward, done, _ = env.step(action)
            cum_reward += reward * (gamma ** step)
            episode.append(ob)
            if done:
                rewards_by_state[start_state].append(cum_reward)
                break
        episodes.append(episode)
    return episodes, rewards_by_state


def print_trajectory(obs, actions_for_obs, ep):
    for idx, i in enumerate(ep):
        if idx % 2 == 0:
            # state
            print(f'({obs[i]})', end="")
        else:
            # action
            print(f'--[{actions_for_obs[ep[idx - 1]][i]}]-->', end="")
    print('\n')


if __name__ == '__main__':
    env = StudentEnv()
    agent = RandomAgent(env.action_space)
    print(env.action_space)
    print(env.observation_space)

    episodes, rewards = sample_mdp(env, agent, episode_count=10000)
    # print(episodes)
    # print(rewards)

    obs = {0: 'FACEBOOK', 1: 'CLASS1', 2: 'CLASS2', 3: 'CLASS3', 4: 'SLEEP'}

    # actions_for_obs = {
    #     0: {0: 'facebook', 1: 'quit'},
    #     1: {0: 'facebook', 1: 'study'},
    #     2: {0: 'sleep', 1: 'study'},
    #     3: {0: 'pub', 1: 'study'},
    #     4: {0: 'sleep'}
    # }
    # one_episode = episodes[0]
    # print_trajectory(obs, actions_for_obs, one_episode)

    for start_state in range(len(obs)):
        avg_reward = np.mean(rewards[start_state])
        print(obs[start_state], round(avg_reward, 2))
