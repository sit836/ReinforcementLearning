import gym
import numpy as np
from tqdm import tqdm

from utils import make_plot

class SARSA:
    def __init__(self, env):
        self.env = env
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n

    def epsilon_greedy(self, Q, epsilon, n_actions, s):
        """
        Q: Q Table
        epsilon: exploration parameter
        n_actions: number of actions
        s: state
        """
        if np.random.random() <= epsilon:
            return np.random.randint(n_actions)
        else:
            return np.argmax(Q[s, :])

    def train(self, alpha, gamma, epsilon, n_episodes):
        """
        alpha: learning rate
        gamma: exploration parameter
        n_episodes: number of episodes
        """
        Q = np.zeros((self.n_states, self.n_actions))
        reward_array = np.zeros(n_episodes)

        for i in tqdm(range(n_episodes)):
            s = self.env.reset()  # initial state
            a = self.epsilon_greedy(Q, epsilon, self.n_actions, s)
            done = False

            while not done:
                s_, reward, done, _ = self.env.step(a)
                a_ = self.epsilon_greedy(Q, epsilon, self.n_actions, s_)
                Q[s, a] += alpha * (reward + (gamma * Q[s_, a_]) - Q[s, a])

                if done:
                    reward_array[i] = reward
                    break
                s, a = s_, a_
        self.env.close()
        return Q, reward_array

    def evaluation(self, Q, n_test_episodes, enable_visual):
        """
        Q: trained Q table
        n_episodes: number of episodes
        enable_visual: whether enable visualization
        """
        reward_array = np.zeros(n_test_episodes)
        for i in tqdm(range(n_test_episodes)):
            s = self.env.reset()
            a = np.argmax(Q[s])
            done = False
            while not done:
                if enable_visual:
                    self.env.render()
                s_, reward, done, _ = self.env.step(a)
                # pick an action according the state and trained Q table
                a_ = np.argmax(Q[s_])
                if done:
                    reward_array[i] = reward
                    break
                s, a = s_, a_
        self.env.close()
        return reward_array


if __name__ == '__main__':
    alpha = 0.1  # learning rate
    gamma = 0.9  # discount factor
    epsilon = 0.5  # exploration parameter
    n_train_episodes = 10000
    n_test_episodes = 100
    enable_visual = True

    is_slippery = True
    env = gym.make("FrozenLake-v1", is_slippery=is_slippery)

    sarsa = SARSA(env)
    Q, _ = sarsa.train(alpha, gamma, epsilon, n_train_episodes)

    test_reward_array = sarsa.evaluation(Q, n_test_episodes=n_test_episodes, enable_visual=enable_visual)
    make_plot(test_reward_array, n_test_episodes, method='SARSA')
