"""ELE2761 Exercise 3: Deep Q-learning.

Implements the provided functionality to be used in your solution.

CLASSES
    DQN          -- Deep Q Network
    Memory       -- Replay Memory

FUNCTIONS
	get_pendulum -- Returns Pendulum Swing-up environment
	get_lander   -- Returns Lunar Lander environment
"""

import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt

class DQN:
    """Deep learning-based Q approximator.

       METHODS
           train       -- Train network.
           __call__    -- Evaluate network.
           __ilshift__ -- Copy network weights.
    """

    def __init__(self, states, actions=1, hiddens=[25, 25]):
        """Creates a new Q approximator.
        
           DQN(states, actions) creates a Q approximator with `states`
           observation dimensions and `actions` action dimensions. It has
           two hidden layers with 25 neurons each. All layers except
           the last use ReLU activation."
           
           DQN(states, actions, hiddens) additionally specifies the
           number of neurons in the hidden layers.

           EXAMPLE
               >>> dqn = DQN(2, 1, [10, 10])
        """
        
        inputs = tf.keras.Input(shape=(states+actions,))
        layer = inputs
        for h in hiddens:
            layer = tf.keras.layers.Dense(h, activation='relu')(layer)
        outputs = tf.keras.layers.Dense(1, activation='linear')(layer)

        self.__model = tf.keras.Model(inputs, outputs)
        self.__model.compile(loss=tf.keras.losses.MeanSquaredError(),
                           optimizer=tf.keras.optimizers.Adam())

    def train(self, s, a, target):
        """Trains the Q approximator.
        
           DQN.train(s, a, target) trains the Q approximator such that
           it approaches DQN(s, a) = target.
           
           `s` is a matrix specifying a batch of observations, in which
           each row is an observation. `a` is a vector specifying an
           action for every observation in the batch. `target` is a vector
           specifying a target value for each observation-action pair in
           the batch.
           
           EXAMPLE
               >>> dqn = DQN(2, 1)
               >>> dqn.train([[0.1, 2], [0.4, 3], [0.2, 5]], [-1, 1, 0], [12, 16, 19])
        """
           
        #self.__model.train_on_batch(self.__combine(s, a), np.atleast_1d(target))
        inp, reshape = self.__combine(s, a)
        self.__model.train_on_batch(inp, np.atleast_1d(target))


    def __call__(self, s, a):
        """Evaluates the Q approximator.
        
           DQN(s, a) returns the value of the approximator at observation
           `s` and action `a`.
           
           `s` is either a vector specifying a single observation, or a
           matrix in which each row specifies one observation in a batch.
           If `a` is the same size as the number of rows in `s`, it specifies
           the action at which to evaluate each observation in the batch.
           Otherwise, it specifies the action(s) at which the evaluate ALL
           observations in the batch.
           
           EXAMPLE
               >>> dqn = DQN(2, 1)
               >>> # single observation and action
               >>> print(dqn([0.1, 2], -1))
               [[ 12 ]]
               >>> # batch of observations and actions
               >>> print(dqn([[0.1, 2], [0.4, 3]], [-1, 1]))
               [[12]
                [16]]
               >>> # evaluate single observation at multiple actions
               >>> print(dqn([0.1, 2], [-1, 1]))
               [[12  -12]]
        """

        inp, reshape = self.__combine(s, a)
        out = np.asarray(self.__model(inp))
        if reshape:
            out = np.reshape(out, reshape)
        return out

    def __ilshift__(self, other):
        """Copies network weights.
        
           dqn2 <<= dqn1 copies the weights from `dqn1` into `dqn2`. The
           networks must have the same structure.
        """

        self.__model.set_weights(other.__model.get_weights())

        return self

    def __combine(self, s, a):
        # Massage s into a 2d array of type float32
        s = np.atleast_2d(np.asarray(s, dtype=np.float32))

        # Massage a into 2d "row-array" of type float32
        a = np.atleast_2d(np.asarray(a, dtype=np.float32))
        if a.shape[1] > 1:
            a = a.transpose()

        # Replicate s and a if necessary
        reshape = None
        if s.shape[0] == 1 and a.shape[0] > 1:
            reshape = [1, a.shape[0]]
            s = np.tile(s, [a.shape[0], 1])
        elif s.shape[0] > 1 and a.shape[0] > 1 and s.shape[0] != a.shape[0]:
            reshape = [s.shape[0], a.shape[0]]
            s = np.repeat(s, np.repeat(reshape[1], reshape[0]), axis=0)
            a = np.tile(a, [reshape[0], 1])

        inp = np.hstack((s, a)).astype(np.float32)

        return inp, reshape

class Memory:
    """Replay memory
	   
       METHODS
           add    -- Add transition to memory.
           sample -- Sample minibatch from memory.
	"""
    def __init__(self, states, actions, size=1000000):
        """Creates a new replay memory.
        
           Memory(states, action) creates a new replay memory for storing
           transitions with `states` observation dimensions and `actions`
           action dimensions. It can store 1000000 transitions.
           
           Memory(states, actions, size) additionally specifies how many
           transitions can be stored.
        """

        self.s = np.ndarray([size, states])
        self.a = np.ndarray([size, actions])
        self.r = np.ndarray([size, 1])
        self.sp = np.ndarray([size, states])
        self.done = np.ndarray([size, 1])
        self.n = 0
    
    def __len__(self):
        """Returns the number of transitions currently stored in the memory."""

        return self.n
    
    def add(self, s, a, r, sp, done):
        """Adds a transition to the replay memory.
        
           Memory.add(s, a, r, sp, done) adds a new transition to the
           replay memory starting in state `s`, taking action `a`,
           receiving reward `r` and ending up in state `sp`. `done`
           specifies whether the episode finished at state `sp`.
        """

        self.s[self.n, :] = s
        self.a[self.n, :] = a
        self.r[self.n, :] = r
        self.sp[self.n, :] = sp
        self.done[self.n, :] = done
        self.n += 1
    
    def sample(self, size):
        """Get random minibatch from memory.
        
        s, a, r, sp, done = Memory.sample(batch) samples a random
        minibatch of `size` transitions from the replay memory. All
        returned variables are vectors of length `size`.
        """

        idx = np.random.randint(0, self.n, size)

        return self.s[idx], self.a[idx], self.r[idx], self.sp[idx], self.done[idx]

"""OpenAI Gym Environment wrapper.

   METHODS
       reset   -- Reset environment
       step    -- Step environment
       render  -- Visualize environment
       close   -- Close visualization
       
   MEMBERS
       states  -- Number of state dimensions
       actions -- Vector of possible actions       
"""
class Environment():
    def reset(self):
        """Reset environment to start state.
        
           obs = env.reset() returns the start state observation.
        """
        return self.env.reset()
    
    def step(self, u):
        """Step environment.
        
           obs, r, done, info = env.step(u) takes action u and
           returns the next state observation, reward, whether
           the episode terminated, and extra information.
        """
        return self.env.step(u)
    
    def render(self):
        """Render environment.
        
           env.render() renders the current state of the
           environment in a separate window.
           
           NOTE
               You must call env.close() to close the window,
               before creating a new environment; otherwise
               the kernel may hang.
        """
        return self.env.render()
    
    def close(self):
        """Closes the rendering window."""
        return self.env.close()    

"""OpenAI Gym Pendulum-v0 environment."""
class Pendulum(Environment):
    """Creates a new Pendulum environment."""
    def __init__(self):
        """Creates a new Pendulum environment.
        
           EXAMPLE
               >>> env = Pendulum()
               >>> print(env.states)
               3
               >>> print(env.actions)
               [-2.  0.  2.]
        """
        self.env = gym.make("Pendulum-v0")        
        self.states = self.env.observation_space.shape[0]        
        self.actions = np.linspace(self.env.action_space.low[0], self.env.action_space.high[0], 3)

    def step(self, u):
        return self.env.step(np.atleast_1d(u))

    def plot(self, network):
        """Plot value function.
		
		   plot(dqn) plots the value function of network `dqn`.
		"""
        xx, yy = np.meshgrid(np.linspace(-np.pi,np.pi, 256), np.linspace(-10, 10, 256))
        obs = np.hstack((np.reshape(np.cos(xx), (xx.size, 1)),
                         np.reshape(np.sin(xx), (xx.size, 1)),
                         np.reshape(       yy , (xx.size, 1))))
        zz = np.reshape(np.amax(network(obs, self.actions), axis=1), xx.shape)

        plt.contourf(xx, yy, zz, 256)
        plt.colorbar()

"""OpenAI Gym LunarLander-v2 environment."""
class Lander(Environment):
    def __init__(self):
        """Creates a new Lander environment.
                
           EXAMPLE
               >>> env = Lander()
               >>> print(env.states)
               8
               >>> print(env.actions)
               [0 1 2 3]
        """
        self.env = gym.make("LunarLander-v2")
        self.states = self.env.observation_space.shape[0]
        self.actions = np.arange(self.env.action_space.n)
