import gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


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
        gamma: exploration parameter
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

    q_learning = QLearning(env)
    Q, _ = q_learning.train(alpha, gamma, epsilon, n_train_episodes)

    test_reward_array = q_learning.evaluation(Q, n_test_episodes=n_test_episodes, enable_visual=enable_visual)
    avg_test_reward = np.mean(test_reward_array)

    plt.subplots(figsize=(6, 6), dpi=100)
    plt.hist(test_reward_array)
    plt.ylabel('Reward', fontsize=12)
    plt.xlabel('Episode', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('Q_learning Agent\nReward Per Episode for {} Episodes - Average: {:.2f}'.format(n_test_episodes, avg_test_reward),
              fontsize=12)
    plt.show()
