################################################################################
# Dumb game to be played by machines
# Guess the correct order of the numbers 1-3 which are shuffled.
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


# future answers are -1s -- -1's can't come before real numbers
#
# l := np.where(a==-1)[0]           --> indices of the -1s and set it to l
# if (l := np.where(a==-1)[0]).size --> true if list has a -1 else false
# a[l[0]:]                          --> from the first -1 until the end ...
# all(a[l[0]:] == -1)               --> are all -1?
# IF the list doesn't have -1s then return it cuz it's good
def has_early_ones(a):
    return all(a[l[0] :] == -1) if (l := np.where(a == -1)[0]).size else True


# numbers in the answer key are unique
#
# u               --> list of unique values in a that are greater than 0
# c               --> corresponding counts of each value in u
# u[c>1]          --> list of unique values that appear more than once
# not u[c>1].size --> true when there are no such duplicates
def answers_are_unique(a):
    u, c = np.unique(a[a >= 0], return_counts=True)
    return not u[c > 1].size


class NumberGuess(Env):
    def __init__(self, be_verbose=False):
        self.n_numbers = 3

        # current knowledge of the hidden answer sequence:
        # we get all the info we need from just an unordered list of the
        # previous correctly-guessed numbers.
        # [0,-1,-1],[1,-1,-1],[2,-1,-1],...,
        # [0,1,-1],[0,2,-1],..., [0,1,2]
        # This brings us down to just 32 entries.
        answer_state = []
        for i in range(self.n_numbers + 1):
            for x in list(itertools.combinations(range(self.n_numbers), i)):
                answer_state.extend([list(x) + [-1] * (self.n_numbers - len(x))])
        answer_state = np.array(answer_state)
        # [list(x) +[-1]*(4-len(x)) for x in list(itertools.combinations(range(5),2))]

        # each number has been guessed or it hasn't
        # [[0,0,0]...[1,1,1]]
        # size 8
        current_guess_state = np.array(
            [a for a in itertools.product(np.arange(0, 2), repeat=self.n_numbers)]
        )

        # concatenate these two state spaces
        #
        # example: [-1,1,-1,0,1,1] means: the first number of the hidden answer
        # sequence is 1 and it has been correctly guessed. Now we're onto round
        # two and we have guessed 1 (again, foolishly), and 2.
        #
        # The get_new_state function updates the state separately from the
        # checking of the game over condition.
        self.observation_states = np.array(
            [
                np.concatenate(a)
                for a in itertools.product(answer_state, current_guess_state)
            ]
        )

        # the environment spaces
        self.action_space = Discrete(self.n_numbers)
        self.observation_space = Discrete(self.observation_states.shape[0])
        self.state = 0  # [-1,-1,-1,0,0,0] no answers, no guesses yet

        # the answer hidden number sequence
        self.answer = list(range(self.n_numbers))
        random.shuffle(self.answer)

        self.be_verbose = be_verbose

        self.n_guesses = 0

        if self.be_verbose:
            print("NumberGuess::__init__: hidden answer list:", self.answer)
            print("answer_state", answer_state)
            print("current guess state", current_guess_state)
            print("obs states", self.observation_states)

    # in: array element of self.observation_states
    # out: int element of self.observation_space
    def get_state_from_array(self, arr):
        try:
            return np.where((self.observation_states == arr).all(axis=1))[0][0]
        except IndexError as e:
            print(
                "get_state_from_array Error: array",
                arr,
                "not found in observation space.",
            )
            return -1

    # A user could in principle provide a custom state (say with the first
    # triplet unordered) that is not in the slimmed down obs states.
    #
    # So this function puts the first n_numbers of elements in the provided
    # array into the correct order (which has the minus signs at the end).
    def reorder_array(self, arr):
        unordered_chunk = arr[: self.n_numbers]
        ordered_chunk = [None] * self.n_numbers
        i = 0
        j = self.n_numbers - 1
        for k in range(self.n_numbers):
            if unordered_chunk[k] >= 0:
                ordered_chunk[i] = unordered_chunk[k]
                i += 1
            else:
                ordered_chunk[j] = unordered_chunk[k]
                j -= 1
            # print("  ", k, ordered_chunk)
        # [self.n_numbers-1:i-1:-1] means start at idx n_numbers-1, end at idx
        # i-1, and do it backwards
        #print(
        #    ordered_chunk,
        #    "start idx:",
        #    self.n_numbers - 1,
        #    "stop idx:",
        #    i - 1,
        #    "and step backwards.",
        #)
        # ordered_chunk[i:] = ordered_chunk[self.n_numbers - 1 : i - 1 : -1]
        # ordered_chunk = np.array(ordered_chunk)
        result = arr
        #print(ordered_chunk)
        result[: self.n_numbers] = ordered_chunk
        #print(result)
        return result

    # in: int element of self.observation_space
    # out: array element of self.observation_states
    def get_array_from_state(self, state=None):
        state = state if state != None else self.state
        try:
            return self.observation_states[state]
        except IndexError as e:
            print("get_array_from_state Error:", e)
            return np.empty_like(self.observation_states[0])

    # in: int element of self.observation_space
    # out: human-friendly dict
    def get_dict_from_state(self, state=None):
        keys = ["answer0", "answer1", "answer2", "guess0", "guess1", "guess2"]
        state = state if state != None else self.state
        arr = self.get_array_from_state(state)
        ret = dict(zip(keys, list(arr)))
        try:
            ret["current_position"] = list(arr).index(
                next((x for x in arr if x == -1), None)
            )
        # we have already correctly guessed the final answer
        except ValueError:
            ret["current_position"] = 2
        return ret

    # in: human-friendly dict
    # out: int element of self.observation_space
    def get_state_from_dict(self, d):
        ret = np.array([val for key, val in d.items])
        assert self.get_state_from_array(ret)
        return ret

    # Process a guessed number
    def get_new_state(self, action):
        initial_position = self.get_dict_from_state()["current_position"]
        final_state_array = np.copy(self.get_array_from_state())

        # if guess is correct
        if action == self.answer[initial_position]:
            # update knowledge of answer
            final_state_array[initial_position] = action
            final_state_array[: initial_position + 1].sort()
            # reset current guesses all to -1
            final_state_array[self.n_numbers :].fill(0)
        # if guess is incorrect
        else:
            # update guess
            final_state_array[self.n_numbers + action] = 1

        return self.get_state_from_array(final_state_array)

    def is_done(self):
        ret = -1 not in self.get_array_from_state()
        return ret

    def step(self, action):
        self.n_guesses += 1
        initial_position = self.get_dict_from_state()["current_position"]
        self.state = self.get_new_state(action)
        final_position = self.get_dict_from_state()["current_position"]
        self.n_guesses += 1
        if self.be_verbose:
            print(
                f"current position: {initial_position}/{self.n_numbers - 1} | "
                f"guess:{action} | correct answer:{self.answer[initial_position]}"
            )
        reward = 1 if final_position > initial_position else -1

        if reward == 1:
            self.n_guesses = 0

        done = self.is_done()

        info = self.get_dict_from_state()

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
    env = NumberGuess(True)
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
