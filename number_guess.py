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
import itertools


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
        # [[-1,-1,-1,-1,-1]...[4,3,2,1,0]]
        answer_state = np.array([a for a in itertools.product(np.arange(-1,self.n_numbers), repeat=self.n_numbers)])

        # future answers are -1s -- -1's can't come before real numbers
        #
        # l := np.where(a==-1)[0]           --> indices of the -1s and set it to l
        # if (l := np.where(a==-1)[0]).size --> true if list has a -1 else false
        # a[l[0]:]                          --> from the first -1 until the end ...
        # all(a[l[0]:] == -1)               --> are all -1?
        def has_early_ones(a):
            return all(a[l[0]:] == -1) if (l := np.where(a==-1)[0]).size else False

        # numbers in the answer key are unique
        #
        # u               --> list of unique values in a that are greater than 0
        # c               --> corresponding counts of each value in u
        # u[c>1]          --> list of unique values that appear more than once
        # not u[c>1].size --> true when there are no such duplicates
        def answers_are_unique(a):
            u,c = np.unique(a[a >= 0], return_counts=True)
            return not u[c>1].size

        # final size of answer space: 1030
        answer_state = np.array([x for x in answer_state if has_early_ones(x)])
        answer_state = np.array([x for x in answer_state if answers_are_unique(x)])


        # each number has been guessed or it hasn't
        # [[0,0,0,0,0]...[1,1,1,1,1]]
        # size 160
        current_guess_state = np.array([a for a in itertools.product(np.arange(0, 2), repeat=self.n_numbers)])

        # concatenate these two state spaces
        # size 65920
        self.observation_states = np.array([np.concatenate(a) for a in itertools.product(answer_state, current_guess_state)])

        self.answer = list(range(self.n_numbers))
        random.shuffle(self.answer)
        self.state = 0  # which index in the answer list are you now?
        self.action_space = Discrete(self.n_numbers)  # guess any number in the range

        self.observation_space = Discrete(self.observation_states.size)

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
