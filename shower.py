#!/usr/bin/python

# ===============================================================================
# Implementation of the common shower temperature control beginner RL example.
#
# This script implements the Bellman equation to create a Q table and test it.
#
# The shower environment is not currently separated from the model
# training/testing, and the "agent" is a one-liner:
# action = np.argmax(q_table[state])
#
# run like:
# python shower.py <n_episodes> <train(0)/test(1)>
# ===============================================================================
import random
import time
import numpy as np
import sys
from enum import Enum, auto

# Set global values
alpha = 0.9  # learning rate
gamma = 1.0  # discount rate
epsilon = 0.0  # exploration threshold
mov_list = [-1, 0, 1]  # Change shower temp
shower_length = 60
temp_target_max = 40
temp_target_min = 36
temp_target = 38
initial_temp_variation = 20
initial_temp = temp_target + random.randint(
    -1 * initial_temp_variation, initial_temp_variation
)

Modes = Enum("Modes", ["TRAIN", "TEST"], start=0)

# fout = open("shower_logfile.txt", "w")

# state space:
# {0: (0, 0), 1: (0, 10), 2: (0, 20), ..., 1799: (490, 350)}
# action space:
# {0: (-20, -20), 1: (-10, -20), 2: (10, -20), ..., 15: (20, 20)}

# trivial state_space dict
# {0: (0), 1: (1), 2: (2), ..., 74:(74)}
state_space = {}
for i in range(75):
    state_space[i] = i

# slightly non-trivial action_space dict
# {0:(-1), 1:(0), 2:(1)}
action_space = {}
for i in range(len(mov_list)):
    action_space[i] = mov_list[i]


# starting state, ending state, action that took you from start to end
def update_q_table(q_table, state_i, state_f, action, step_reward):
    try:
        # note about 2d np array access:
        # q_table[state_i] accesses the state_i-th row, which is an array with
        # length equal to number of actions.
        # q_table[state_i, action] access the action-th element of the state_ith
        # array.
        old_q_value = q_table[state_i, action]
    except IndexError:
        print("ERROR with q_table")
        print(q_table)
        print(state_i, action)
        sys.exit()

    # max q value given the state after this temp change
    next_max = np.max(q_table[state_f])
    q_target = step_reward + gamma * next_max
    q_delta = q_target - old_q_value
    q_table[state_i, action] = old_q_value + alpha * q_delta

    return q_table


# Update the q-table after every second, not just after every shower
def shower(n_showers=1000, mode=Modes.TRAIN.value):
    try:
        mode = Modes(int(mode))
    except ValueError:
        sys.exit("Invalid mode. Must be either train (0) or test (1).")
    print(mode.name, "mode. N showers:", n_showers)

    # Initialize Q table
    q_table = None
    if mode == Modes.TRAIN:
        q_table = np.zeros([len(state_space), len(action_space)])
        # [
        #  [0. 0. 0.]
        #  [0. 0. 0.]
        #    ...
        #  [0. 0. 0.]
        # ]
    elif mode == Modes.TEST:
        q_table = np.loadtxt("qtable.csv", delimiter=",")

    # Set local variables
    shower_time = 0
    temp = initial_temp  # initialize starting temp for i_shower = 0
    temp_change = None  # initialize temp change per step
    i_shower = 0
    total_reward = 0  # net reward of all n_showers
    shower_reward = 0  # net reward for a single shower

    # Loop showers
    while True:
        step_reward = 0  # reward after a single step
        state = temp

        # i_shower over due to timeout. Go to next shower, reset variables.
        if shower_time == shower_length:
            # print status of the shower that just finished
            print(
                i_shower,
                "th i_shower finished. Final temp:",
                temp,
                ". Reward:",
                shower_reward,
            )

            # increment total reward
            total_reward += shower_reward

            # reset shower reward
            shower_reward = 0

            # increment i_shower number
            i_shower += 1

            # reset initial temp, state
            temp = temp_target + random.randint(
                -1 * initial_temp_variation, initial_temp_variation
            )
            state = temp

            # reset shower time
            shower_time = 1

        # quit if this is the last i_shower
        if i_shower == n_showers:
            np.savetxt("qtable.csv", q_table, delimiter=",")
            print("average reward:", total_reward / n_showers)
            break

        # set action
        if mode == Modes.TRAIN:
            if (
                i_shower < 20000000
            ):  # arbitrarily chosen high value to prevent exploitation ;
                # change above if exploitation is desired afterwards certain number of i_showers
                temp_change = random.choice(mov_list)
                action = temp_change + 1
            else:
                # explore
                if random.random() < epsilon:
                    temp_change = random.choice(mov_list)
                    action = temp_change + 1
                # use q-table ("exploit")
                else:
                    # get the /index/ of the max q-value of the state-th array
                    action = np.argmax(q_table[state])
                    temp_change = action - 1
        elif mode == Modes.TEST:
            action = np.argmax(q_table[state])
            temp_change = action - 1

        # change the action so we don't go out of bounds
        if temp > 74:
            temp_change = -1
        if temp <= 1:
            temp_change = 1

        # Update temperature and state
        temp += temp_change

        # assign step reward
        if temp_target_min <= temp <= temp_target_max:
            step_reward = 1
        else:
            # step_reward = -1*(abs(temp_target-temp))
            step_reward = -1

        # print("  ", state, temp, temp_change, step_reward)

        # Update the Q table based on this step's reward
        if mode == Modes.TRAIN:
            q_table = update_q_table(q_table, state, temp, action, step_reward)

        # increment shower's reward and time
        shower_reward += step_reward
        shower_time += 1


if __name__ == "__main__":
    if len(sys.argv) == 2:
        shower(sys.argv[1])
    elif len(sys.argv) == 3:
        shower(int(sys.argv[1]), int(sys.argv[2]))
    else:
        shower()
