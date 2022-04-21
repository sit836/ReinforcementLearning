# Reference: https://python.quantecon.org/short_path.html
import numpy as np
from numpy import inf

Q = np.array([[inf, 1, 5, 3, inf, inf, inf],
              [inf, inf, inf, 9, 6, inf, inf],
              [inf, inf, inf, inf, inf, 2, inf],
              [inf, inf, inf, inf, inf, 4, 8],
              [inf, inf, inf, inf, inf, inf, 4],
              [inf, inf, inf, inf, inf, inf, 1],
              [inf, inf, inf, inf, inf, inf, 0]])

nodes = range(7)
J = np.zeros_like(nodes, dtype=int)  # Initial guess
next_J = np.empty_like(nodes, dtype=int)  # Stores updated guess

max_iter = 10
i = 0

while i < max_iter:
    for v in nodes:
        # minimize Q[v, w] + J[w] over all choices of w
        lowest_cost = inf
        for w in nodes:
            print(f'i={i}, Q[{v}, {w}]={Q[v, w]}, J[{w}]={J[w]}')
            cost = Q[v, w] + J[w]
            if cost < lowest_cost:
                lowest_cost = cost
        next_J[v] = lowest_cost

    if np.equal(next_J, J).all():
        break
    else:
        J[:] = next_J  # Copy contents of next_J to J
        i += 1

    print(f"i={i}, J={J}")
    """
    i=1, J=[1 6 2 4 4 1 0]
        J[0] = 1: Start from node A, take 1 step, the min cost is 1;
        J[1] = 6: Start from node B, take 1 step, the min cost is 6;
        ...
    i=2, J=[7 10  3  5  4  1  0]
        J[0] = 7: Start from node A, take 2 steps, the min cost is 7; 
        ...     
    """


print("The cost-to-go function is", J)
