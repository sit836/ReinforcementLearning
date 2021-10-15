import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def query(action, rewards_mean):
    return np.random.normal(loc=rewards_mean[action], scale=0.05, size=1)


def take_action(expected_rewards, walker, is_greedy, epsilon):
    def greedy_action():
        if walker == 0:
            return np.random.choice(range(num_arms), size=1)
        else:
            return np.argmax(expected_rewards)

    def epsilon_greedy_action(epsilon):
        if np.random.binomial(1, p=1 - epsilon, size=1) > 0:
            return greedy_action()
        else:
            return np.random.choice(range(num_arms), size=1)

    if is_greedy:
        return greedy_action()
    else:
        return epsilon_greedy_action(epsilon)


if __name__ == '__main__':
    is_greedy = False
    epsilon = 0  # if epsilon = 0, then greedy; if epsilon = 1, then random choice
    num_arms = 5
    rewards_mean = np.array([0.2, -0.3, 0.3, 0.6, 0.0])

    num_iter = 500
    expected_rewards = np.zeros(num_arms)
    counts = np.zeros(num_arms)

    for walker in range(num_iter):
        action = take_action(expected_rewards, walker, is_greedy, epsilon)
        counts[action] += 1
        expected_rewards[action] = expected_rewards[action] + (query(action, rewards_mean) - expected_rewards[action]) / \
                                   counts[action]

    print("expected_rewards: ", expected_rewards)

    sns.barplot(x=['Reward ' + str(r) for r in rewards_mean], y=counts, alpha=0.6)
    plt.ylabel("Number of Action Taken")
    plt.show()
