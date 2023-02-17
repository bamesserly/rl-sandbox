import numpy as np

def update_q_table(params, q_table, state_i, state_f, action, step_reward):
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
        return None

    # max q value given the state after this temp change
    next_max = np.max(q_table[state_f])
    q_target = step_reward + params["gamma"] * next_max
    q_delta = q_target - old_q_value
    q_table[state_i, action] = old_q_value + params["alpha"] * q_delta

    return q_table


# alpha - learning rate
# gamma - discount rate
# epsilon - exploration threshold (not currently used)
default_params = {"alpha": 0.9, "gamma": 1.0, "epsilon": 0.0}


# loop number_guess games
def train_test(env, in_q_table, n_episodes=5, do_train=True, params=default_params):
    q_table = in_q_table.copy()
    total_reward = 0
    for i_game in range(n_episodes):
        done = False
        env.reset()
        state_i = env.state
        game_reward = 0
        # print(i_game)
        while not done:
            # choose action
            action = env.action_space.sample()
            if not do_train and env.n_guesses < 5:
                action = np.argmax(q_table[state_i])

            # take a step
            state_f, reward, done, info = env.step(action)

            if done:
                continue

            # print("  ", env.shower_time, state, reward, done)
            try:
                assert state_f in env.observation_space
            except AssertionError:
                print("Invalid state obtained", state_f, done, i_game, action)
                break

            # update q table
            if do_train:
                q_table = update_q_table(
                    params, q_table, state_i, state_f, action, reward
                )

            # increment reward
            game_reward += reward

            state_i = state_f

        # print("  Shower reward:", shower_reward)
        total_reward += game_reward

    # np.savetxt("qtable.csv", q_table, delimiter=",")
    avg_reward = total_reward / n_episodes
    return q_table, avg_reward
