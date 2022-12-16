#!/usr/bin/python
#===============================================================================
# code initially from: 
# https://github.com/bhattacharyya/reach_circle
#
# Training script. Creates qtable.csv.
#===============================================================================
import random
import time
import numpy as np
import sys
from enum import Enum, auto

# Set global values
alpha = 0.9 # learning rate
gamma = 1.0 # discount rate
epsilon = 0.0 # exploration threshold
n_showers = 50 # Number of showers to be performed
mov_list = [-1,0,1]  # Change shower temp
initial_temp = 38 + random.randint(-3,3)
shower_length = 60
shower_target_max = 40
shower_target_min = 36
shower_target = 38

Modes = Enum('Modes', ['TRAIN', 'TEST'], start=0)

#fout = open("shower_logfile.txt", "w")

# state space:
# {0: (0, 0), 1: (0, 10), 2: (0, 20), ..., 1799: (490, 350)}
# action space:
# {0: (-20, -20), 1: (-10, -20), 2: (10, -20), ..., 15: (20, 20)}

# trivial state_space dict
# {0: (0), 1: (1), 2: (2), ..., 74:(74)}
state_space = {}
for i in range(75):
    state_space[i] = (i)

# slightly non-trivial action_space dict
# {0:(-1), 1:(0), 2:(1)}
action_space = {}
for i in range(len(mov_list)):
    action_space[i] = (mov_list[i])

# Initialize Q table
q_table = np.zeros(
    [len(state_space), len(action_space)]
)
#[
#  [0. 0. 0.]
#  [0. 0. 0.]
#    ...
#  [0. 0. 0.]
#]

#q_table = np.loadtxt("qtable.csv",delimiter=',') # Uncomment this to read from an existing q table

# Update the q-table after every shower
def shower(mode = Modes.TRAIN.value):
    try:
        mode = Modes(int(mode))
    except ValueError:
        sys.exit("Invalid mode. Must be either train (0) or test (1).")
    print(mode.name, "mode")
    # Set local variables
    shower_time = 0
    temp = initial_temp # initialize starting temp for i_shower = 0
    temp_change = None # initialize temp change per step
    i_shower = 0
    total_reward = 0 # net reward of all n_showers
    shower_reward = 0 # net reward for a single shower

    # Loop showers
    while True:
        step_reward = 0 # reward after a single step
        state = temp

        # i_shower over due to timeout. Go to next shower, reset variables.
        if shower_time == shower_length:
            # print status of the shower that just finished
            print(i_shower, "th i_shower finished. Final temp:", temp, ". Reward:", shower_reward)

            # increment total reward
            total_reward += shower_reward

            # reset shower reward
            shower_reward = 0

            # increment i_shower number
            i_shower += 1

            # reset initial temp, state
            temp = shower_target + random.randint(-3,3)
            state = temp

            # reset shower time
            shower_time = 1

        # quit if this is the last i_shower
        if i_shower == n_showers:
            np.savetxt("qtable.csv", q_table, delimiter=",")
            print("average reward:", total_reward/n_showers)
            break

        # set action
        if i_shower < 20000000:  # arbitrarily chosen high value to prevent exploitation ;
            # change above if exploitation is desired afterwards certain number of i_showers
            temp_change = random.choice(mov_list)
            action = temp_change+1
        else:
            # explore
            if random.random() < epsilon:
                temp_change = random.choice(mov_list)
                action = temp_change+1
            # use q-table ("exploit")
            else:
                # get the /index/ of the max q-value of the state-th array
                action = np.argmax(q_table[state])
                temp_change = action-1

        # change the action so we don't go out of bounds
        if temp > 74:
            temp_change = -1
        if temp <= 1:
            temp_change = 1

        # Update temperature and state
        temp += temp_change

        # assign step reward
        if shower_target_min <= temp <= shower_target_max:
            step_reward = 1
        else:
            #step_reward = -1*(abs(shower_target-temp))/shower_target
            step_reward = -1

        # Update the Q table based on this step's reward
        try:
            # note about 2d np array access:
            # q_table[state] accesses the state-th row, which is an array with
            # length equal to number of actions.
            # q_table[state, action] access the action-th element of the stateth
            # array.
            old_q_value = q_table[state, action]
        except IndexError:
            print("ERROR with q_table")
            print(q_table)
            print(state, action)
            sys.exit()

        next_state = temp
        # max q value given the state after this temp change
        next_max = np.max(q_table[next_state])
        q_target = step_reward + gamma * next_max
        q_delta = q_target - old_q_value
        q_table[state, action] = old_q_value + alpha * q_delta

        # increment shower's reward and time
        shower_reward += step_reward
        shower_time += 1

if __name__ == "__main__":
    if len(sys.argv) == 2:
        shower(sys.argv[1])
    else:
         shower()
