import numpy as np


def policy_eval(policy, env, discount_factor, theta):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    Copied from https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Value%20Iteration%20Solution.ipynb
    note: no modifications needed for use with the DiscreteLimitActionsEnv.

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


def one_step_lookahead(state, V, env, discount_factor):
    """
    Helper function to calculate the value for all action in a given state.

    Args:
        state: The state to consider (int)
        V: The value to use as an estimator, Vector of length env.nS
        env: The OpenAI environment.
        discount_factor: gamma discount factor.

    Returns:
        A vector of length env.vA[state] containing the expected value of each action.
    """
    A = np.zeros(env.vA[state])
    for a in range(env.vA[state]):
        for prob, next_state, reward, _ in env.P[state][a]:
            A[a] += prob * (reward + discount_factor * V[next_state])
    return A


def policy_improvement(env, theta, discount_factor):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.

    Args:
        env: The OpenAI environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: gamma discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

    """
    # Start with a random policy
    policy = [np.ones(nA) / nA for nA in env.vA]

    while True:
        # Evaluate the current policy
        V = policy_eval(policy, env, discount_factor, theta)

        # Will be set to false if we make any changes to the policy
        policy_stable = True

        # For each state...
        for s in range(env.nS):
            # The best action we would take under the current policy
            chosen_a = np.argmax(policy[s])

            # Find the best action by one-step lookahead
            # Ties are resolved arbitarily
            action_values = one_step_lookahead(s, V, env, discount_factor)
            best_a = np.argmax(action_values)

            # Greedily update the policy
            if chosen_a != best_a:
                policy_stable = False
            policy[s] = np.eye(env.vA[s])[best_a]

        # If the policy is stable we've found an optimal policy. Return it
        if policy_stable:
            return policy, V


def value_iteration(env, theta, discount_factor):
    """
    Value Iteration Algorithm.
    Copied from https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Value%20Iteration%20Solution.ipynb
    with slight alterations for compatibility with the action_space in DiscreteLimitActionsEnv.

    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.vA is a vector of the number of actions per state in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """
    V = np.zeros(env.nS)
    while True:
        # Stopping condition
        delta = 0
        # Update each state...
        for s in range(env.nS):
            # Do a one-step lookahead to find the best action
            A = one_step_lookahead(s, V, env, discount_factor)
            best_action_value = np.max(A)
            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(best_action_value - V[s]))
            # Update the value function. Ref: Sutton book eq. 4.10.
            V[s] = best_action_value
            # Check if we can stop
        if delta < theta:
            break

    # Create a deterministic policy using the optimal value function
    policy = [np.zeros(nA) for nA in env.vA]
    for s in range(env.nS):
        # One step lookahead to find the best action for this state
        A = one_step_lookahead(s, V, env, discount_factor)
        best_action = np.argmax(A)
        # Always take the best action
        policy[s][best_action] = 1.0
    return policy, V
