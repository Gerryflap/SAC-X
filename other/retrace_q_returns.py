import random
from collections import defaultdict

q_values = defaultdict(lambda : 0)

alpha = 0.01
gamma = 0.5

# c_k is omitted for simplicity, eg. same policy
policy = [0.1, 0.9]
states = [0, 1] # 1 is terminal
action_space = [0, 1]
rewards = {(0,0): -1, (0,1): 1}


replay_buffer = [
    [(0, 0), (1,0)],
    [(0, 0), (1,1)],
    [(0, 1), (1,0)],
    [(0, 1), (1,1)],
]

def get_q_return(trajectory):
    expected_q_s0 = sum([policy[a] * q_values[(0, a)] for a in action_space])
    q_return = 0
    for j, (s, a) in enumerate(trajectory[:-1]):
        q_delta = expected_q_s0 - q_values[(s,a)]
        q_return += rewards[(s, a)] + q_delta
    return q_return

def get_q_delta(trajectory):
    q_delta = 0
    cum_reward = 0
    for j, (s, a) in enumerate(trajectory[:-1]):
        cum_reward += gamma**j * rewards[(s,a)]
        expected_q_tp1 = sum([policy[a] * q_values[(1, a)] for a in action_space])
        delta = cum_reward + gamma**(j+1) * expected_q_tp1 - q_values[(s,a)]
        print(delta)
        q_delta += delta

    return q_delta


for i in range(100000):
    trajectory = replay_buffer[random.randint(0, len(replay_buffer)-1)]
    trajectory = trajectory[random.randint(0, len(trajectory)-1):]

    s, a = trajectory[0]
    #q_return = get_q_return(trajectory)
    #q_values[(s,a)] -= alpha*2*(q_values[(s,a)] - q_return)
    q_values[(s,a)] += alpha*get_q_delta(trajectory)
    print(q_values)