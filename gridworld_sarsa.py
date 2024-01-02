import numpy as np

class GridWorld:
    def __init__(self):
        self.grid_size = 5
        self.grid = np.zeros((self.grid_size, self.grid_size))  # 5x5 grid
        self.state = np.array([0, 0])  # initial state
        self.actions = np.array([0, 1, 2, 3])  # 0: up, 1: down, 2: left, 3: right
        self.obstacles = np.array([[2, 2], [3, 2]])  # define obstacle states as 2,2 , 3,2
        self.boundary_obstacles = np.array([[-1, i] for i in range(self.grid_size)] +
                                          [[self.grid_size, i] for i in range(self.grid_size)] +
                                          [[i, -1] for i in range(self.grid_size)] +
                                          [[i, self.grid_size] for i in range(self.grid_size)])

        self.water = np.array([[4, 2]])  # water
        self.gold = np.array([[]])
        self.goal = np.array([[4, 4]])  # goal state
        self.discount = 0.9  # discount factor

    def getPossibleTransitions(self, state, action):
        transitions = []

        if action == 0:
            if not np.any(np.all(np.array([state[0]-1, state[1]]) == np.concatenate((self.obstacles, self.boundary_obstacles)), axis=1)):
                new_state = np.array([state[0]-1, state[1]])
            else:
                new_state = state
        elif action == 1:
            if not np.any(np.all(np.array([state[0]+1, state[1]]) == np.concatenate((self.obstacles, self.boundary_obstacles)), axis=1)):
                new_state = np.array([state[0]+1, state[1]])
            else:
                new_state = state
        elif action == 2:
            if not np.any(np.all(np.array([state[0], state[1]-1]) == np.concatenate((self.obstacles, self.boundary_obstacles)), axis=1)):
                new_state = np.array([state[0], state[1]-1])
            else:
                new_state = state
        elif action == 3:
            if not np.any(np.all(np.array([state[0], state[1]+1]) == np.concatenate((self.obstacles, self.boundary_obstacles)), axis=1)):
                new_state = np.array([state[0], state[1]+1])
            else:
                new_state = state

        transitions.append(new_state)  # 80% probability next state

        if action == 0:
            if not np.any(np.all(np.array([state[0], state[1]+1]) == np.concatenate((self.obstacles, self.boundary_obstacles)), axis=1)):
                new_state = np.array([state[0], state[1]+1])
            else:
                new_state = state
        elif action == 1:
            if not np.any(np.all(np.array([state[0], state[1]-1]) == np.concatenate((self.obstacles, self.boundary_obstacles)), axis=1)):
                new_state = np.array([state[0], state[1]-1])
            else:
                new_state = state
        elif action == 2:
            if not np.any(np.all(np.array([state[0]-1, state[1]]) == np.concatenate((self.obstacles, self.boundary_obstacles)), axis=1)):
                new_state = np.array([state[0]-1, state[1]])
            else:
                new_state = state
        elif action == 3:
            if not np.any(np.all(np.array([state[0]+1, state[1]]) == np.concatenate((self.obstacles, self.boundary_obstacles)), axis=1)):
                new_state = np.array([state[0]+1, state[1]])
            else:
                new_state = state

        transitions.append(new_state)  # 5% probability next state

        if action == 0:
            if not np.any(np.all(np.array([state[0], state[1]-1]) == np.concatenate((self.obstacles, self.boundary_obstacles)), axis=1)):
                new_state = np.array([state[0], state[1]-1])
            else:
                new_state = state
        elif action == 1:
            if not np.any(np.all(np.array([state[0], state[1]+1]) == np.concatenate((self.obstacles, self.boundary_obstacles)), axis=1)):
                new_state = np.array([state[0], state[1]+1])
            else:
                new_state = state
        elif action == 2:
            if not np.any(np.all(np.array([state[0]+1, state[1]]) == np.concatenate((self.obstacles, self.boundary_obstacles)), axis=1)):
                new_state = np.array([state[0]+1, state[1]])
            else:
                new_state = state
        elif action == 3:
            if not np.any(np.all(np.array([state[0]-1, state[1]]) == np.concatenate((self.obstacles, self.boundary_obstacles)), axis=1)):
                new_state = np.array([state[0]-1, state[1]])
            else:
                new_state = state

        transitions.append(new_state)  # 5% probability next state

        transitions.append(np.array(state))  # 10% probability next state remains same after breakdown

        return transitions


    def getReward(self, state):
        state = state if isinstance(state, np.ndarray) else state

        if any(np.array_equal(water_state, state) for water_state in self.water):
            # print("WATER!")
            return -10
        elif any(np.array_equal(goal_state, state) for goal_state in self.goal):
            return 10
        # elif state in self.gold:
            # return 5
        else:
            return 0

def epsilon_greedy_policy(Q_values, state, epsilon):
    if np.random.uniform(low=0, high=1) < epsilon:
        return np.random.choice(list(Q_values[tuple(state)].keys()))
    else:
        return max(Q_values[tuple(state)], key=Q_values[tuple(state)].get)
    
def get_best_policy(Q_values):
    policy = {}
    for state, action_values in Q_values.items():
        # print(state, action_values, np.argmax(list(action_values)))
        policy[state] = np.argmax(list(action_values.values()))
    return policy
    
def feature_vector(state, action):
    return np.array([1, state[0], state[1], action])


def predict(env, Q_values, state, action=None):
    if action is None:
        return [Q_values.get(tuple(state), {}).get(a.item(), 0) for a in env.actions]
    else:
        return Q_values.get(tuple(state), {}).get(action.item(), 0)

def q_to_v(q_values):
    v_values = {}

    for state, action_values in q_values.items():
        max_q_value = max(action_values.values())
        v_values[state] = max_q_value

    return v_values

def q_to_v_list(q_function, grid_size=5):
    value_function = [[0.0] * grid_size for _ in range(grid_size)]

    for state, action_values in q_function.items():
        row, col = state
        max_action_value = max(action_values.values())
        value_function[row][col] = max_action_value

    return value_function

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def randomFeasibleState(env):
    feasible_states = []
    for i in range(env.grid_size):
        for j in range(env.grid_size):
            if [i, j] not in np.concatenate((env.obstacles, env.boundary_obstacles, env.goal), axis=0).tolist() and (j in [0, 1, 2] or (i==0 and j==4)):
                feasible_states.append([i, j])
    return feasible_states[np.random.choice(len(feasible_states))]

def convert_to_value_func_list(value_func_dict, grid_size=5):
    true_value_func = np.zeros((grid_size, grid_size))

    for state, value in value_func_dict.items():
        i, j = state
        true_value_func[i][j] = value

    return true_value_func

def n_step_semigradient_sarsa(env, Q_values, n, alpha, epsilon, true_value_func):
    Q_values_check = {(i, j): {a.item(): False for a in env.actions} for i in range(env.grid_size) for j in range(env.grid_size)}
    time_steps = []
    mse = []
    for l in range(NUM_EPISODES):
        # epsilon = 0.9 if NUM_EPISODES < 1000 else 0.5 if NUM_EPISODES < 2000 else 0.3 if NUM_EPISODES < 3000 else 0.1
        state = randomFeasibleState(env)  # initial state
        action = epsilon_greedy_policy(Q_values, state, epsilon)

        states = [state]
        actions = [action]
        rewards = []

        T = float('inf')
        t = 0
        while True:
            if t < T:
                possible_next_states = env.getPossibleTransitions(state, action)
                next_state = possible_next_states[np.random.choice(len(possible_next_states), p=[0.8, 0.05, 0.05, 0.1])]
                reward = env.getReward(next_state)               
                rewards.append(reward)
                # if reward == -10:
                    # print("yep")
                # if reward == 10:
                    # print("YAY")
                    # exit()

                if any(np.array_equal(goal_state, next_state) for goal_state in env.goal):
                    T = t + 1
                    # print("reached the goal!")
                    # print("s, a", states, actions)
                else:
                    states.append(next_state)
                    next_action = epsilon_greedy_policy(Q_values, next_state, epsilon)
                    actions.append(next_action)

            tau = t - n + 1

            if tau >= 0:
                G = sum(env.discount ** (i - tau) * rewards[i] for i in range(tau, min(tau + n, T)))

                if tau + n < T:
                    state_update_n = states[tau + n]
                    action_update_n = actions[tau + n]
                    G += ((env.discount**n) * Q_values[tuple(state_update_n)][action_update_n])

                state_update = states[tau]
                actions_update = actions[tau]
                # print("Tau: ", tau)
                Q_values[tuple(state_update)][actions_update] += alpha * (G - Q_values[tuple(state_update)][actions_update])
                Q_values_check[tuple(state_update)][actions_update] = True
            if t < T:
                state = next_state
                if not any(np.array_equal(goal_state, state) for goal_state in env.goal):
                    action = next_action


                # print("reward was : ", G)
            # print('t: ', t)
            t += 1

            if tau == T - 1:
                break

            
            
        # print("Episode : ", l)
        # print("Rewards received :", rewards)
        # print("Q_Values: ", Q_values)
        # print("Q_values check: ", Q_values_check)
        mse.append(mean_squared_error(np.array(list(true_value_func.values())), np.array(list(q_to_v(Q_values).values()))))
        time_steps.append(t)

    return Q_values, time_steps, mse


NUM_EPISODES = 1000
alpha = 0.1
epsilon = 0.5
n = 8
# initialize value function to zero for all states
value_func =np.zeros((5,5))
env = GridWorld()
Q_values = {(i, j): {a.item(): np.random.uniform(low=0, high=1) for a in env.actions} for i in range(env.grid_size) for j in range(env.grid_size)}
Q_values[(4,4)] = {0:-1, 1:-1, 2:-1, 3:-1}
Q_values[(2,2)] = {0:-1, 1:-1, 2:-1, 3:-1}
Q_values[(3,2)] = {0:-1, 1:-1, 2:-1, 3:-1}

true_value_func = {(0, 0): 4.01868773, (0, 1): 4.55478334, (0, 2): 5.15754366, (0, 3): 5.83363485, (0, 4): 6.45528746, (1, 0): 4.37160397, (1, 1): 5.03235817, (1, 2): 5.8012948, (1, 3): 6.64726448, (1, 4): 7.39070797, (2, 0): 3.86716628, (2, 1): 4.38996363, (2, 2): 0.0, (2, 3): 7.57690382, (2, 4): 8.46366119, (3, 0): 3.41824818, (3, 1): 3.83189559, (3, 2): 0.0, (3, 3): 8.57382965, (3, 4): 9.69459248, (4, 0): 2.99768663, (4, 1): 2.93092585, (4, 2): 6.07330036, (4, 3): 9.69459248, (4, 4): 0.0}
Q_values_list = []
time_steps_list = []
mse_list = []
for i in range(20):
    Q_values = {(i, j): {a.item(): np.random.uniform(low=0, high=1) for a in env.actions} for i in range(env.grid_size) for j in range(env.grid_size)}
    Q_values[(4,4)] = {0:-1, 1:-1, 2:-1, 3:-1}
    Q_values[(2,2)] = {0:-1, 1:-1, 2:-1, 3:-1}
    Q_values[(3,2)] = {0:-1, 1:-1, 2:-1, 3:-1}
    Q_values, time_steps, mse = n_step_semigradient_sarsa(env, Q_values, n, alpha, epsilon, true_value_func)
    Q_values_list.append(Q_values)
    time_steps_list.append(time_steps)
    mse_list.append(mse)\
    
Q_values = {}
for q_values_dict in Q_values_list:
    for key, value in q_values_dict.items():
        if key not in Q_values:
            Q_values[key] = []
        Q_values[key].append(value)
Q_values = {key: {a: np.mean([d[a] for d in values]) for a in env.actions} for key, values in Q_values.items()}

time_steps = np.mean(np.array(time_steps_list), axis=0)
mse = np.mean(np.array(mse_list), axis=0)
# print("value func is ", Q_values)]
value_func = q_to_v_list(Q_values)
value_func[4][4] = 0
value_func[2][2] = 0
value_func[3][2] = 0
best_policy = get_best_policy(Q_values)
best_policy = convert_to_value_func_list(best_policy)
for row_value, row_policy in zip(value_func, best_policy):
    print(" ".join("{:.4f}".format(val) for val in row_value), end=" \t\t")

    # Replace 0 with up arrow, 1 with down arrow, 3 with right arrow, and 2 with left arrow in the policy
    policy_symbols = ["↑" if val == 0 else "↓" if val == 1 else "→" if val == 3 else "←" if val == 2 else " " if val == -1 else 'G' for val in row_policy]
    
    # Print the policy symbols
    print(" ".join(policy_symbols))
print("time steps is ", list(time_steps))
print("MSE is ", list(mse))
