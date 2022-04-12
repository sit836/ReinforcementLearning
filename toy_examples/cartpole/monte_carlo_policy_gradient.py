import gym
import matplotlib.pyplot as plt
import numpy as np
from gym.wrappers.monitor import Monitor
from tqdm import tqdm

"""
    Based on the work of Janis Klaise: https://www.janisklaise.com/post/rl-policy-gradients/
"""
GLOBAL_SEED = 0
DIM_STATE_SPACE = 4
MAX_EPISODES = 3000


class LogisticPolicy:
    def __init__(self, theta, alpha, gamma):
        self.theta = theta
        self.alpha = alpha
        self.gamma = gamma

    def logistic(self, y):
        return 1 / (1 + np.exp(-y))

    def probs(self, x):
        y = x @ self.theta
        prob0 = self.logistic(y)

        return np.array([prob0, 1 - prob0])

    def act(self, x):
        # sample an action in proportion to probabilities

        probs = self.probs(x)
        action = np.random.choice([0, 1], p=probs)

        return action, probs[action]

    def grad_log_p(self, x):
        # calculate grad-log-probs

        y = x @ self.theta
        grad_log_p0 = x - x * self.logistic(y)
        grad_log_p1 = - x * self.logistic(y)

        return grad_log_p0, grad_log_p1

    def grad_log_p_dot_rewards(self, grad_log_p, discounted_rewards):
        # dot grads with future rewards for each action in episode

        return grad_log_p.T @ discounted_rewards

    def discount_rewards(self, rewards):
        # calculate temporally adjusted, discounted rewards

        discounted_rewards = np.zeros(len(rewards))
        cumulative_rewards = 0
        for i in reversed(range(0, len(rewards))):
            cumulative_rewards = cumulative_rewards * self.gamma + rewards[i]
            discounted_rewards[i] = cumulative_rewards

        return discounted_rewards

    def update(self, rewards, obs, actions):
        # calculate gradients for each action over all observations
        grad_log_p = np.array([self.grad_log_p(ob)[action] for ob, action in zip(obs, actions)])

        assert grad_log_p.shape == (len(obs), DIM_STATE_SPACE)

        # calculate temporaly adjusted, discounted rewards
        discounted_rewards = self.discount_rewards(rewards)

        # gradients times rewards
        dot = self.grad_log_p_dot_rewards(grad_log_p, discounted_rewards)

        # gradient ascent on parameters
        self.theta += self.alpha * dot


def run_episode(env, policy, render=False):
    observation = env.reset()
    totalreward = 0

    observations = []
    actions = []
    rewards = []
    probs = []

    done = False

    while not done:
        if render:
            env.render()

        observations.append(observation)

        action, prob = policy.act(observation)
        observation, reward, done, info = env.step(action)

        totalreward += reward
        rewards.append(reward)
        actions.append(action)
        probs.append(prob)

    return totalreward, np.array(rewards), np.array(observations), np.array(actions), np.array(probs)


def train(theta, alpha, gamma, Policy, max_epsidoes=1000, seed=None, evaluate=False):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson

    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf

    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right
        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.

        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """
    # initialize environment and policy
    env = gym.make('CartPole-v0')
    if seed is not None:
        env.seed(seed)

    episode_rewards = []
    policy = Policy(theta, alpha, gamma)

    for i in tqdm(range(max_epsidoes)):
        env.render()

        total_reward, rewards, observations, actions, probs = run_episode(env, policy)
        episode_rewards.append(total_reward)
        policy.update(rewards, observations, actions)
        print("EP: " + str(i) + " Score: " + str(total_reward) + " ", end="\r", flush=False)

    # evaluation call after training is finished - evaluate last trained policy on 100 episodes
    if evaluate:
        consecutive_target = 100
        env = Monitor(env, 'pg_cartpole/', video_callable=False, force=True)
        for _ in range(consecutive_target):
            run_episode(env, policy, render=False)
        env.env.close()

    return episode_rewards, policy


def plot_rewards(episode_rewards):
    fontsize = 14
    success_threshold = 195.0

    plt.plot(episode_rewards)
    plt.axhline(y=success_threshold, color='r', linestyle='--', label='Success Threshold')
    plt.ylabel("Total Reward", fontsize=fontsize)
    plt.xlabel("Iteration", fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.show()


if __name__ == "__main__":
    np.random.seed(GLOBAL_SEED)

    episode_rewards, policy = train(theta=np.random.rand(DIM_STATE_SPACE),
                                    alpha=0.002,
                                    gamma=0.99,
                                    Policy=LogisticPolicy,
                                    max_epsidoes=MAX_EPISODES,
                                    seed=GLOBAL_SEED,
                                    evaluate=False)
    plot_rewards(episode_rewards)
