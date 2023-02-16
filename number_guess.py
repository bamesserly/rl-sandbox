################################################################################
# Dumb game to be played by machines
# Guess the correct order of the numbers 1-5 which are shuffled.
# Keep guessing until you get the whole sequence.
# Penalized -1 for every wrong guess.
################################################################################

import gymnasium as gym
from gym import Env
from gym.spaces import Discrete, Box
import random
import numpy as np
import math


class NumberGuess(Env):
    def __init__(self, be_verbose=False):
        self.n_numbers = 5
        self.answer = list(range(self.n_numbers))
        random.shuffle(self.answer)
        self.state = 0  # which index in the answer list are you now?
        self.action_space = Discrete(self.n_numbers)  # guess any number in the range
        self.observation_space = Discrete(
            self.n_numbers
        )  # states range from 0th index up to last one
        self.be_verbose = be_verbose
        self.n_guesses = 0
        # self.observation_space = Box(low=np.array([0],dtype=np.uintc), high=np.array([2],dtype=np.uintc))
        if self.be_verbose:
            print("NumberGuess::__init__: hidden answer list:", self.answer)

    def step(self, action):
        if self.be_verbose:
            print(
                "current position:",
                self.state,
                "/",
                self.n_numbers - 1,
                "guess:",
                action,
                "correct answer:",
                self.answer[self.state],
            )
        reward = 0
        if action == self.answer[self.state]:
            self.state += 1
            reward = 1
        else:
            reward = -1

        done = self.state == self.n_numbers or self.state < -50

        info = {}

        # Return step information
        return self.state, reward, done, info

    def render(self):
        pass

    def reset(self):
        random.shuffle(self.answer)
        self.state = 0
        self.n_guesses = 0
        if self.be_verbose:
            print("NumberGuess::reset: hidden answer list:", self.answer)
        return self.state


class NumberGuess2(Env):
    def __init__(self, be_verbose=False):
        self.n_numbers = 5

        # knowledge of the hidden answer: -1 if unknown otherwise contains its value
        # [[-1,0,1,2,3]...[4,3,2,1,0]]
        # size 720
        answer_state = np.array(list(itertools.permutations(np.arange(-1,self.n_numbers,1), self.n_numbers)))

        # each number has been guessed or it hasn't
        # [[0,0,0,0,0]...[1,1,1,1,1]]
        # size 160
        current_guess_state = np.array([a for a in itertools.product(np.arange(0, 2), repeat=self.n_numbers)])

        # concatenate these two state spaces
        # size 115200
        observation_space = np.array([np.concatenate(a) for a in itertools.product(answer_state, current_guess_state)])

        #any([all([my_list[l] == x for x in my_list[l:]]) and my_list[l] == -1 for l in range(len(my_list))])

        self.answer = list(range(self.n_numbers))
        random.shuffle(self.answer)
        self.state = 0  # which index in the answer list are you now?
        self.action_space = Discrete(self.n_numbers)  # guess any number in the range

        # permutations of n numbers = n!
        # current guess record = 2*n
        state_dim = math.factorial(self.n_numbers) * 2 * self.n_numbers
        self.observation_space = Discrete(state_dim)

        self.be_verbose = be_verbose
        self.n_guesses = 0
        # self.observation_space = Box(low=np.array([0],dtype=np.uintc), high=np.array([2],dtype=np.uintc))
        if self.be_verbose:
            print("NumberGuess::__init__: hidden answer list:", self.answer)

    def step(self, action):
        if self.be_verbose:
            print(
                "current position:",
                self.state,
                "/",
                self.n_numbers - 1,
                "guess:",
                action,
                "correct answer:",
                self.answer[self.state],
            )
        reward = 0
        if action == self.answer[self.state]:
            self.state += 1
            reward = 1
        else:
            reward = -1

        done = self.state == self.n_numbers or self.state < -50

        info = {}

        # Return step information
        return self.state, reward, done, info

    def render(self):
        pass

    def reset(self):
        random.shuffle(self.answer)
        self.state = 0
        self.n_guesses = 0
        if self.be_verbose:
            print("NumberGuess::reset: hidden answer list:", self.answer)
        return self.state


if __name__ == "__main__":
    env = NumberGuess()
    # print(env.observation_space.sample()) # 0-1
    # print(env.action_space.sample()) # 0-4
    episodes = 10
    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        score = 0
        n_guesses = 0
        while not done:
            n_guesses += 1
            action = env.action_space.sample()
            n_state, reward, done, info = env.step(action)
            score += reward
        print(f"Episode:{episode} Score:{score} NGuesses:{n_guesses}")
