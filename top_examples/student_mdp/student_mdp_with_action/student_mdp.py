import numpy as np
from collections import defaultdict

from discrete_limit_env import StudentEnv

from constants import actions_for_obs, obs
from helper import print_trajectory, create_random_policy

"""
Reference: https://github.com/frangipane/reinforcement-learning/blob/master/02-dynamic-programming/student_MDP.ipynb 
"""


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


def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.vA is a vector of the number of actions per state in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)
    while True:
        delta = 0
        # For each state, perform a "full backup"
        for s in range(env.nS):
            v = 0
            # Look at the possible next actions
            for a, action_prob in enumerate(policy[s]):
                # For each action, look at the possible next states...
                for prob, next_state, reward, done in env.P[s][a]:
                    # Calculate the expected value. Ref: Sutton book eq. 4.6.
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            # How much our value function changed (across any states)
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        # Stop evaluating once our value function change is below a threshold
        if delta < theta:
            break
    return np.array(V)


if __name__ == '__main__':
    env = StudentEnv()
    agent = RandomAgent(env.action_space)
    print(env.action_space)
    print(env.observation_space)

    episodes, rewards = sample_mdp(env, agent, episode_count=10000)
    # print(episodes)
    # print(rewards)

    # one_episode = episodes[0]
    # print_trajectory(obs, actions_for_obs, one_episode)

    # State values evaluation
    for start_state in range(len(obs)):
        avg_reward = np.mean(rewards[start_state])
        print(obs[start_state], round(avg_reward, 2))

    # Policy evaluation
    random_policy = create_random_policy()
    value_fun = policy_eval(random_policy, env)
    for s, value in enumerate(value_fun):
        print(f"optimal state-value in state {obs[s]}: ", round(value, 2))
