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
        # final size of answer space: 1030
        answer_state = np.array(
            [
                a
                for a in itertools.product(
                    np.arange(-1, self.n_numbers), repeat=self.n_numbers
                )
            ]
        )
        answer_state = np.array([x for x in answer_state if has_early_ones(x)])
        answer_state = np.array([x for x in answer_state if answers_are_unique(x)])

        # each number has been guessed or it hasn't
        # [[0,0,0,0,0]...[1,1,1,1,1]]
        # size 160
        current_guess_state = np.array(
            [a for a in itertools.product(np.arange(0, 2), repeat=self.n_numbers)]
        )

        # concatenate these two state spaces
        # size 65920
        self.observation_states = np.array(
            [
                np.concatenate(a)
                for a in itertools.product(answer_state, current_guess_state)
            ]
        )

        # the environment spaces
        self.action_space = Discrete(self.n_numbers)
        self.observation_space = Discrete(self.observation_states.size)
        self.state = 0  # [-1,-1,-1,-1,-1,0,0,0,0,0] no answers, no guesses yet

        # the answer hidden number sequence
        self.answer = list(range(self.n_numbers))
        random.shuffle(self.answer)

        self.be_verbose = be_verbose

        self.n_guesses = 0

        if self.be_verbose:
            print("NumberGuess::__init__: hidden answer list:", self.answer)

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

    # in: int element of self.observation_space
    # out: array element of self.observation_states
    def get_array_from_state(self, state = None):
        state = state if state != None else self.state
        try:
            return self.observation_states[state]
        except IndexError as e:
            print("get_array_from_state Error:", e)
            return np.empty_like(self.observation_states[0])

    # in: int element of self.observation_space
    # out: human-friendly dict
    def get_dict_from_state(self, state = None):
        keys = ["answer0", "answer1", "answer2", "answer3", "answer4",
                "guess0", "guess1", "guess2", "guess3", "guess4"]
        state = state if state != None else self.state
        arr = self.get_array_from_state(state)
        ret = dict(zip(keys, list(arr)))
        ret["current_position"] = l[0] if (l := np.where(arr == -1)[0]).size else self.n_numbers
        return ret

    # in: human-friendly dict
    # out: int element of self.observation_space
    def get_state_from_dict(self, d):
        ret = np.array([val for key, val in d.items])
        assert self.get_state_from_array(ret)
        return ret


    def update_state(self, action):
        current_position = self.get_dict_from_state()["current_position"]
        state_arr = np.copy(self.get_array_from_state())

        # if guess is correct
        if action == self.answer[current_position]:
            # update knowledge of answer
            state_arr[current_position] = action
            # reset current guesses all to -1
            state_arr[self.n_numbers+1 :].fill(0)
        # if guess is incorrect
        else:
            # update guess
            state_arr[self.n_numbers + action] = 1

        # update the state itself
        self.state = self.get_state_from_array(state_arr)

        return self.state


    def is_done(self):
        ret = -1 not in self.get_array_from_state()
        check = self.get_dict_from_state()["current_position"] == self.n_numbers
        try:
            assert ret == check
        except AssertionError:
            print("is_done Error")
            print("self.get_array_from_state()", self.get_array_from_state())
            print("self.get_dict_from_state()[\"current_position\"]", self.get_dict_from_state()["current_position"])
            raise
        return ret


    def step(self, action):
        self.n_guesses += 1
        prev_position = self.get_dict_from_state()["current_position"]
        self.update_state(action)
        self.n_guesses += 1
        if self.be_verbose:
            print(
                f"current position: {prev_position}/{self.n_numbers - 1} | "
                f"guess:{action} | correct answer:{self.answer[prev_position]}"
            )
        reward = 1 if self.get_dict_from_state()["current_position"] > prev_position else - 1

        if reward == 1: self.n_guesses = 0

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


class NumberGuess3(Env):
    def __init__(self, be_verbose=False):
        self.n_numbers = 5

        # current knowledge of the hidden answer sequence:
        # we get all the info we need from just an unordered list of the
        # previous correctly-guessed numbers.
        # [0,-1,-1,-1,-1],[1,-1,-1,-1,-1],[2,-1,-1,-1,-1],...,
        # [0,1,-1,-1,-1],[0,2,-1,-1,-1],..., [0,1,2,3,4]
        # This brings us down to just 32 entries.
        answer_state = []
        for i in range(self.n_numbers+1):
            for x in list(itertools.combinations(range(self.n_numbers),i)):
                answer_state.extend([list(x) +[-1]*(self.n_numbers-len(x))])
        answer_state = np.array(answer_state)
        # [list(x) +[-1]*(4-len(x)) for x in list(itertools.combinations(range(5),2))]

        # each number has been guessed or it hasn't
        # [[0,0,0,0,0]...[1,1,1,1,1]]
        # size 32
        current_guess_state = np.array(
            [a for a in itertools.product(np.arange(0, 2), repeat=self.n_numbers)]
        )

        # concatenate these two state spaces
        # size 1024
        self.observation_states = np.array(
            [
                np.concatenate(a)
                for a in itertools.product(answer_state, current_guess_state)
            ]
        )

        # the environment spaces
        self.action_space = Discrete(self.n_numbers)
        self.observation_space = Discrete(self.observation_states.shape[0])
        self.state = 0  # [-1,-1,-1,-1,-1,0,0,0,0,0] no answers, no guesses yet

        # the answer hidden number sequence
        self.answer = list(range(self.n_numbers))
        random.shuffle(self.answer)

        self.be_verbose = be_verbose

        self.n_guesses = 0

        if self.be_verbose:
            print("NumberGuess::__init__: hidden answer list:", self.answer)

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

    # in: int element of self.observation_space
    # out: array element of self.observation_states
    def get_array_from_state(self, state = None):
        state = state if state != None else self.state
        try:
            return self.observation_states[state]
        except IndexError as e:
            print("get_array_from_state Error:", e)
            return np.empty_like(self.observation_states[0])

    # in: int element of self.observation_space
    # out: human-friendly dict
    def get_dict_from_state(self, state = None):
        keys = ["answer0", "answer1", "answer2", "answer3","answer4",
                "guess0", "guess1", "guess2", "guess3", "guess4"]
        state = state if state != None else self.state
        arr = self.get_array_from_state(state)
        ret = dict(zip(keys, list(arr)))
        try:
            ret["current_position"] = list(arr).index(next((x for x in arr if x == -1), None))
        # we have already correctly guessed the final answer
        except ValueError:
            ret["current_position"] = 4
        return ret

    # in: human-friendly dict
    # out: int element of self.observation_space
    def get_state_from_dict(self, d):
        ret = np.array([val for key, val in d.items])
        assert self.get_state_from_array(ret)
        return ret


    def get_new_state(self, action):
        initial_position = self.get_dict_from_state()["current_position"]
        final_state_array = np.copy(self.get_array_from_state())

        # if guess is correct
        if action == self.answer[initial_position]:
            # update knowledge of answer
            final_state_array[initial_position] = action
            final_state_array[:initial_position+1].sort()
            # reset current guesses all to -1
            final_state_array[self.n_numbers:].fill(0)
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
        reward = 1 if final_position > initial_position else - 1

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
