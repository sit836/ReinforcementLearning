FACEBOOK = 0
CLASS1 = 1
CLASS2 = 2
CLASS3 = 3
SLEEP = 4  # terminal state

obs = {0: 'FACEBOOK', 1: 'CLASS1', 2: 'CLASS2', 3: 'CLASS3', 4: 'SLEEP'}
actions_for_obs = {
    0: {0: 'facebook', 1: 'quit'},
    1: {0: 'facebook', 1: 'study'},
    2: {0: 'sleep', 1: 'study'},
    3: {0: 'pub', 1: 'study'},
    4: {0: 'sleep'}
}

discount_factor = 0.9
theta = 0.0001  # stopping threshold in solving Bellman's equation
