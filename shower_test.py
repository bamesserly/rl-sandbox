#!/usr/bin/python
import random
import time
import numpy as np
import sys

# Set global values
alpha = 0.9
gamma = 1.0
epsilon = 0.0
episodes = 1000 # Number of games to be played
mov_list = [-1,0,1]  # Change shower temp
initial_temp = 38 + random.randint(-3,3)

fout = open("shower_logfile.txt", "w")

# trivial state_space dict
# {0, 1, 2, ..., 74}
state_space = {}
for i in range(75):
    state_space[i] = i

# slightly non-trivial action_space dict
# {0:-1, 1:0, 2:1}
action_space = {}
for i in range(len(mov_list)):
    action_space[i] = mov_list[i]

#q_table = np.zeros(
#    [len(state_space), len(action_space)]
#)  # Use this for generating new Q table
q_table = np.loadtxt("qtable.csv",delimiter=',') # Uncomment this to read from an existing q table


def shower():

    # Set local variables
    steps = 0
    temp = initial_temp
    x1 = 1  # movement size in x direction
    game = 1
    reward = 0
    game_reward = 0
    total_reward = 0

    # Loop 1000 games
    end = 1
    while end == 1:
        reward = 0
        state = temp
        steps += 1

        # quit if this is the last game
        if game == episodes:
            end = 0
            np.savetxt("qtable.csv", q_table, delimiter=",")
            fout.write("qtable saved at " + str(steps) + " steps\n")
            print("saved qtable")
            fout.flush()
            print("average reward:", total_reward/episodes)

        # game over due to time out
        if steps == 60:
            print(game, "th game finished. Final temp", temp, ". Reward:", game_reward)
            total_reward += game_reward
            game_reward = 0
            game += 1

            # Reset to initial temp 
            temp = 38 + random.randint(-3,3)

            steps = 1

        ## choose a random action
        #if game < 20000000:  # arbitrarily chosen high value to prevent exploitation ;
        #    # change above if exploitation is desired afterwards certain number of games
        #    x1 = random.choice(mov_list)
        #    action = x1+1
        #else:
        #    # Exploit based on epsilon probability
        #    if random.random() < epsilon:
        #        x1 = random.choice(mov_list)
        #        action = x1+1
        #    else:
        #        action = np.argmax(q_table[state])
        #        x1 = action-1

        action = np.argmax(q_table[state])
        x1 = action-1

        # Not allowed to go out of bounds
        if temp > 74:
            x1 = -1
        if temp <= 1:
            x1 = 1

        # Update temperature
        temp += x1

        # assign reward
        if 37 < temp < 40:
            reward = 1
        else:
            reward = -1*abs(39-temp)

        game_reward += reward

        ## Update the Q table
        #try:
        #    old_q_value = q_table[state, action]
        #except IndexError:
        #    print("ERROR with q_table")
        #    print(q_table)
        #    print(state, action)
        #    sys.exit()
        #next_state = state
        #next_max = np.max(q_table[next_state])
        #q_target = reward + gamma * next_max
        #q_delta = q_target - old_q_value
        #q_table[state, action] = old_q_value + alpha * q_delta

shower()
