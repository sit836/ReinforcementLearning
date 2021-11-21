import matplotlib.pyplot as plt
import numpy as np


def make_plot(test_reward_array, n_test_episodes, method):
    avg_test_reward = np.mean(test_reward_array)

    plt.subplots(figsize=(6, 6), dpi=100)
    plt.hist(test_reward_array)
    plt.ylabel('Reward', fontsize=12)
    plt.xlabel('Episode', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(
        f'{method} Agent\nReward Per Episode for {n_test_episodes} Episodes - Average: {round(avg_test_reward, 2)}',
        fontsize=12)
    plt.show()
