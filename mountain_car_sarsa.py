import ivy
import matplotlib.pyplot as plt
# The following tile coding software is open source under The MIT License (MIT) and is from Sutton and Barto's implementation from their book Reinforcement Learning: An Introduction
# http://incompleteideas.net/tiles/tiles3.html
basehash = hash
class IHT:
    "Structure to handle collisions"
    def __init__(self, sizeval):
        self.size = sizeval
        self.overfullCount = 0
        self.dictionary = {}

    def __str__(self):
        "Prepares a string for printing whenever this object is printed"
        return "Collision table:" + \
               " size:" + str(self.size) + \
               " overfullCount:" + str(self.overfullCount) + \
               " dictionary:" + str(len(self.dictionary)) + " items"

    def count (self):
        return len(self.dictionary)

    def fullp (self):
        return len(self.dictionary) >= self.size

    def getindex (self, obj, readonly=False):
        d = self.dictionary
        if obj in d: return d[obj]
        elif readonly: return None
        size = self.size
        count = self.count()
        if count >= size:
            if self.overfullCount==0: print('IHT full, starting to allow collisions')
            self.overfullCount += 1
            return basehash(obj) % self.size
        else:
            d[obj] = count
            return count

def hashcoords(coordinates, m, readonly=False):
    if type(m)==IHT: return m.getindex(tuple(coordinates), readonly)
    if type(m)==int: return basehash(tuple(coordinates)) % m
    if m==None: return coordinates

from math import floor, log
from itertools import zip_longest

def tiles (ihtORsize, numtilings, floats, ints=[], readonly=False):
    """returns num-tilings tile indices corresponding to the floats and ints"""
    qfloats = [floor(f*numtilings) for f in floats]
    Tiles = []
    for tiling in range(numtilings):
        tilingX2 = tiling*2
        coords = [tiling]
        b = tiling
        for q in qfloats:
            coords.append( (q + b) // numtilings )
            b += tilingX2
        coords.extend(ints)
        Tiles.append(hashcoords(coords, ihtORsize, readonly))
    return Tiles

ivy.set_backend('torch', dynamic=True)
ivy.set_default_device("gpu:0")
ivy.set_soft_device_mode(True)

class mountain_car:
    def __init__(self):
        self.actions = [-1, 0, 1]
        self.x_min = -1.2
        self.x_max = 0.5
        self.v_min = -0.07
        self.v_max = 0.07
        self.x = ivy.random_uniform(low=-0.6, high=-0.4).item()
        self.v = 0
        self.state = (self.x, self.v)

    def get_reward(self, next_state):
        if next_state[0] >= self.x_max:
            return 0
        else:
            return -1
        
    def get_next_state(self, state, action):
        x, v = state
        v_next = v + 0.001 * action - 0.0025 * ivy.cos(3*x)
        v_next = max(min(v_next, self.v_max), self.v_min)

        x_next = x + v_next
        x_next = max(min(x_next, self.x_max), self.x_min)

        if x_next == self.x_min:
            v_next = 0

        return x_next, v_next


def epsilon_greedy_action(env, state, tile):
        rand = ivy.random_uniform(low=0, high=1).item()
        actions = env.actions
        if rand < EPSILON:
            return actions[ivy.multinomial(len(actions), 1).item()]
        else:
            qv = [tile.predict(state, a) for a in actions]
            best_action = ivy.argmax(qv).item() - 1
            return best_action

class Tiles:
    def __init__(self, env, max_size, num_tilings, num_tiles):
        self.iht = IHT(max_size)
        self.scale = ivy.array([num_tiles / abs(env.x_max - env.x_min), num_tiles / abs(env.v_max - env.v_min)])
        self.w = ivy.zeros(max_size)
        self.max_size = max_size
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles
    def predict(self, state, action):
        current_tiles = tiles(self.iht, self.num_tilings, [ivy.dot(ivy.array(state), self.scale)], [action])
        return ivy.sum(self.w[current_tiles])


def n_step_semi_gradient_sarsa(n, gamma, alpha, tile):
    returns = []
    for i in range(NUM_EPISODES):
        t = 0
        T = float('inf')
        env = mountain_car()
        state = env.state
        action = epsilon_greedy_action(env, state, tile)
        states, actions, rewards = [state] + [None] * n, [action] + [None] * n, [None] * (n+1)
        max_p = -2
        while True:
            if t < T:
                state = env.get_next_state(state, action)
                idx = (t+1)%(n+1)
                states[idx] = state
                reward = env.get_reward(state)
                rewards[idx] = reward

                if state[0] >= env.x_max:
                    print(" reached the goal at time t: ", t)
                    returns.append(-1 * t)
                    T = t + 1
                else:
                    action = epsilon_greedy_action(env, state, tile)
                    actions[idx] = action

            tau = t - n + 1
            if tau >= 0:
                mod = n+1
                G = sum([gamma**(i-tau) * rewards[(i+1)%mod] for i in range(tau, min(tau+n, T))])
                if tau + n < T:
                    G += gamma**n * tile.predict(states[(tau+n)%mod], actions[(tau+n)%mod])
                state_tau = states[tau%mod]
                action_tau = actions[tau%mod]

                current_tiles = tiles(tile.iht, tile.num_tilings, [ivy.dot(ivy.array(state_tau), tile.scale)], [action_tau])
                tile.w[current_tiles] += alpha * (G - ivy.sum(tile.w[current_tiles]))

            print("time, position, action: ", t, state[0], action)
            print("device: ", ivy.dev(state[0]))
            max_p = max(max_p, state[0])
            t += 1
            if tau == T - 1 or t>100000:
                break
        # print("max position: ", max_p)
    return returns
    
def plot_return(returns):
    episodes = list(range(1, len(returns) + 1))
    plt.plot(episodes, returns)
    plt.title('Learning Curve - Return vs Episode')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.grid(True)
    plt.show()


def plot_log_return(returns):
    episodes = list(range(1, len(returns) + 1))
    plt.plot(episodes, returns)
    plt.yscale('log')  # Set the y-axis to a logarithmic scale
    plt.title('Learning Curve - Return vs Episode (Log Scale)')
    plt.xlabel('Episode')
    plt.ylabel('Return (log scale)')
    plt.grid(True)
    plt.show()

MAX_SIZE = 4096
NUM_TILINGS = 8
NUM_TILES = 8
N = 8
EPSILON = 0.1
GAMMA = 1
ALPHA = 0.06
NUM_EPISODES = 1000
env = mountain_car()

returns = []
traced_n_step_sarsa = ivy.trace_graph(n_step_semi_gradient_sarsa).__call__ # if this does not work, comment it out 
for i in range(20):
    tile = Tiles(env, MAX_SIZE, NUM_TILINGS, NUM_TILES)
    returns.append(traced_n_step_sarsa(N, GAMMA, ALPHA, tile)) # if this doesn't work, comment it out and uncomment the following line
    # returns.append(n_step_semi_gradient_sarsa(N, GAMMA, ALPHA, tile))

return_mean = ivy.mean(returns, axis=0)
plot_return(return_mean)
plot_log_return(return_mean)
