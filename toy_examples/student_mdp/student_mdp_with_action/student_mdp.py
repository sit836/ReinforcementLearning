import numpy as np
from collections import defaultdict

from discrete_limit_env import StudentEnv

from constants import actions_for_obs, obs, theta, discount_factor
from helper import print_trajectory, create_random_policy
from dynamic_programming import value_iteration, policy_eval, policy_improvement


class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()[observation]


def sample_mdp(env, agent, episode_count, discount_factor=1.0):
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
            cum_reward += reward * (discount_factor ** step)
            episode.append(ob)
            if done:
                rewards_by_state[start_state].append(cum_reward)
                break
        episodes.append(episode)
    return episodes, rewards_by_state


def print_result(opt_policy, opt_value_fun, method):
    print(f"###{method}###")
    for s, actions in enumerate(opt_policy):
        print(f"In state {obs[s]}")
        print(f"optimal state-value: ", opt_value_fun[s])
        print(f"action for optimal policy: ", actions_for_obs[s][np.argmax(actions)], '\n')


if __name__ == '__main__':
    env = StudentEnv()
    agent = RandomAgent(env.action_space)
    print(env.action_space)
    print(env.observation_space)

    episodes, rewards = sample_mdp(env, agent, episode_count=1000)
    # print(episodes)
    # print(rewards)
    #
    # one_episode = episodes[0]
    # print_trajectory(obs, actions_for_obs, one_episode)

    # # State values evaluation
    # for start_state in range(len(obs)):
    #     avg_reward = np.mean(rewards[start_state])
    #     print(obs[start_state], round(avg_reward, 2))

    # Policy evaluation
    random_policy = create_random_policy()
    value_fun = policy_eval(random_policy, env, discount_factor=discount_factor, theta=theta)
    for s, value in enumerate(value_fun):
        print(f"optimal state-value in state {obs[s]}: ", round(value, 2))

    opt_policy_pi, opt_value_fun_pi = policy_improvement(env, theta, discount_factor)
    print_result(opt_policy_pi, opt_value_fun_pi, method="Policy Improvement")

    opt_policy_vi, opt_value_fun_vi = value_iteration(env, theta=theta, discount_factor=discount_factor)
    print_result(opt_policy_vi, opt_value_fun_vi, method="Value Iteration")
