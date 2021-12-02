import gym

"""
    The surface is described using a grid like the following
        SFFF
        FHFH
        FFFH
        HFFG
    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    G : goal, where the frisbee is located
    
    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, and zero otherwise.
"""
MAX_ITERATIONS = 10

env = gym.make("FrozenLake-v1", is_slippery=False)
env.reset()
env.render()

for i in range(MAX_ITERATIONS):
    random_action = env.dim_action_space.sample()
    new_state, reward, done, info = env.step(random_action)
    env.render()
    if done:
        break
