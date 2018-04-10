import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr


def rargmax(vector):
    m = np.amax(vector)
    indices = np.nonzero(vector == m)
    print("indices[0] : ", indices[0])
    print("indices[1] : ", indices[1])
    return pr.choice(indices)


register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4',
            'is_slippery': False}
)

env = gym.make('FrozenLake-v3')

# Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])
# Discount factor
dis = .99
num_episodes = 2000

rList = []

aaa = np.random.randn(1, env.action_space.n)
a = rargmax(aaa)
print(aaa)
print(a)

'''
for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False
    # The Q-Table learning algorithm
    while not done:
        # Choose an action by greedily (with noise) picking from Q table
        noise = np.random.randn(1, env.action_space.n)
        actionList = Q[state, :] + noise / (i+1)
        action = rargmax(actionList)
        print("action : ", action)
        print("noise : ", noise)
        print(actionList)
        # Get new state and reward from environment
        new_state, reward, done, _ = env.step(action)

        # Update Q-Table with new knowledge using decay rate
        Q[state, action] = reward + dis * np.max(Q[new_state, :])

        rAll += reward
        state = new_state
        print("state : ", state)
        print(Q)
    rList.append(rAll)

print("success rate: " + str(sum(rList) / num_episodes))
print("final Q table values")
print(Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.show()
'''