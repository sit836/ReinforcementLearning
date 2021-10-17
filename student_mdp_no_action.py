import numpy as np


class RewardChain:
    def __init__(self, transition_matrix, rewards, state_names, discount_factor, terminal_index=6):
        self.transition_matrix = transition_matrix
        self.state_names = state_names
        self.terminal_index = terminal_index
        self.rewards = rewards
        self.discount_factor = discount_factor

        assert (len(set(
            [len(row) for row in transition_matrix])) == 1), "transition_matrix rows must have equal lengths!"
        assert len(transition_matrix[0]) == len(transition_matrix), "transition_matrix must be square!"
        assert (len(self.rewards) == len(transition_matrix)), "rewards must be same length as transition_matrix"

    def generate_episode(self, start_state):
        episode = []
        if isinstance(start_state, str):
            start_indx = self.state_names.index(start_state)
        else:
            start_indx = start_state

        state = start_indx

        while state != self.terminal_index:
            episode.append(state)
            transition_prob = self.transition_matrix[state]
            next_state = np.random.choice(range(len(self.transition_matrix)), p=transition_prob)
            state = next_state

        episode.append(self.terminal_index)

        if isinstance(start_state, str):
            return self.pretty(episode)
        else:
            return episode

    def pretty(self, episode):
        return [self.state_names[i] for i in episode]

    def compute_total_discounted_reward(self, episode):
        str_check = [isinstance(x, str) for x in episode]
        if any(str_check):
            assert all(str_check), "Path must be all int or all string"
            episode = [self.state_names.index(x) for x in episode]

        counter = 0
        reward = 0
        for state in episode:
            reward += (self.rewards[state] * (self.discount_factor ** counter))
            counter += 1

        return reward

    def evaluate_state_value_function(self, num_iteration=1000):
        states = np.zeros(len(self.transition_matrix))

        for s in range(len(self.state_names)):
            rewards = np.zeros(num_iteration)
            for i in range(num_iteration):
                episode = chain.generate_episode(s)
                rewards[i] = chain.compute_total_discounted_reward(episode)

            states[s] = np.mean(rewards)
        return {state_names[i]: round(states[i], 2) for i in range(len(self.transition_matrix))}

    @property
    def compute_theoretical_state_value_function(self):
        R = self.rewards
        P = np.matrix(self.transition_matrix)
        I = np.identity(len(self.transition_matrix))
        solution = (np.linalg.inv((I - self.discount_factor * P)) @ R).tolist()[0]

        return {state_names[i]: round(solution[i], 2) for i in range(len(self.transition_matrix))}


if __name__ == '__main__':
    state_names = ["Class_1", "Class_2", "Class_3", "Pass", "Pub", "Facebook", "Sleep"]
    transition_matrix = [[0, .5, 0, 0, 0, .5, 0],
                         [0, 0, .8, 0, 0, 0, .2],
                         [0, 0, 0, .6, .4, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1],
                         [.2, .4, .4, 0, 0, 0, 0],
                         [.1, 0, 0, 0, 0, .9, 0],
                         [0, 0, 0, 0, 0, 0, 1]]
    rewards = [-2, -2, -2, 10, 1, -1, 0]
    discount_factor = 0.9

    chain = RewardChain(transition_matrix, rewards, state_names, discount_factor)

    # episode = chain.generate_path("Class_1")
    # print("episode: ", episode)
    # print("Return: ", chain.compute_total_discounted_reward(episode))

    estimated_state_value_function = chain.evaluate_state_value_function(num_iteration=1000)
    theoretical_state_value_function = chain.compute_theoretical_state_value_function
    print("estimated_state_value_function: ", estimated_state_value_function)
    print("theoretical_state_value_function: ", theoretical_state_value_function)
