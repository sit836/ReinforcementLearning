import gym
import numpy as np
from tqdm import tqdm

from utils import make_plot


class QLearning:
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
        gamma: discount factor
        epsilon: exploration parameter
        n_episodes: number of episodes
        """
        Q = np.zeros((self.n_states, self.n_actions))
        reward_array = np.zeros(n_episodes)

        for i in tqdm(range(n_episodes)):
            s = self.env.reset()  # initial state
            done = False

            while not done:
                a = self.epsilon_greedy(Q, epsilon, self.n_actions, s)
                s_, reward, done, _ = self.env.step(a)
                Q[s, a] += alpha * (reward + (gamma * np.max(Q[s_, :])) - Q[s, a])

                if done:
                    reward_array[i] = reward
                    break
                s = s_
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
    n_train_episodes = 500
    n_test_episodes = 1
    enable_visual = True

    is_slippery = False
    env = gym.make("FrozenLake-v1", is_slippery=is_slippery)

    qlearning = QLearning(env)
    Q, _ = qlearning.train(alpha, gamma, epsilon, n_train_episodes)
    print(Q)

    test_reward_array = qlearning.evaluation(Q, n_test_episodes=n_test_episodes, enable_visual=enable_visual)
    make_plot(test_reward_array, n_test_episodes, method='Q-learning')
