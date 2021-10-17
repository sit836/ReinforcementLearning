import numpy as np

from constants import actions_for_obs


def print_trajectory(obs, actions_for_obs, ep):
    for idx, i in enumerate(ep):
        if idx % 2 == 0:
            # state
            print(f'({obs[i]})', end="")
        else:
            # action
            print(f'--[{actions_for_obs[ep[idx - 1]][i]}]-->', end="")
    print('\n')


def create_random_policy():
    """
    Create a random policy which is equivalent to an agent chooses action randomly in each state
    """
    random_policy = dict()
    for s, actions in actions_for_obs.items():
        n_actions = len(actions)
        random_policy[s] = np.ones(n_actions) / n_actions
    return random_policy
