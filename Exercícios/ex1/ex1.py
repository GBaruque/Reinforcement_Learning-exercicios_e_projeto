"""ELE2761 Exercise 1: Dynamic Programming.

Implements the provided functionality to be used in your solution.

FUNCTIONS
    gridworld    -- Grid world environment.
    observations -- Efficiency analysis.
    plotv        -- Value function plotting.
"""

import numpy as np
import matplotlib.pyplot as plt

__obs = 0

def gridworld(s, a):
    """Grid world from Sutton & Barto, example 3.8.
    
    r, sp = gridworld(s, a) returns the reward r and next state sp when
    starting in state s and taking action a. 0 <= s, sp < 25, 0 <= a < 4
    
    ACTIONS
        0 -- up
        1 -- down
        2 -- right
        3 -- left
    """
    
    global __obs
    __obs += 1

    y = int(s/5)
    x = int(s-y*5)
    a = int(a)
    
    if y == 0:
        if x == 1:
            r = 10
            sp = 4*5+1
            return r, sp
        elif x == 3:
            r = 5
            sp = 2*5+3
            return r, sp

    x += -(a==3) + (a==2);
    y += -(a==0) + (a==1);

    r = 0;

    if x < 0:
        r = -1
        x = 0
    elif x > 4:
        r = -1
        x = 4
    elif y < 0:
        r = -1
        y = 0
    elif y > 4:
        r = -1
        y = 4

    sp = y*5+x
    
    return r, sp

def observations():
    """Returns number of gridworld observations since last call.

    NOTE
        Make sure to reset the count by calling this function before
        calling the function you want to analyse.
    """
    global __obs
    obs = __obs
    __obs = 0
    return obs

def plotv(v):
    """Plot grid world value function and induced policy.
    
    plotv(v) plots the grid world value function v and its induced policy.

    NOTE
        If multiple actions are optimal, only one is plotted.
    """
    
    v = np.reshape(v, 25)

    pi = np.zeros(25)
    for s in range(25):
        actions = np.zeros(4)
        for a in range(4):
            _, sp = gridworld(s, a)
            actions[a] = v[sp]
        pi[s] = np.argmax(actions)

    v = np.reshape(v, (5, 5))
    pi = np.reshape(pi, (5, 5))
    
    ax = -(pi==3).astype(int) + (pi==2).astype(int);
    ay = -(pi==0).astype(int) + (pi==1).astype(int);
    
    x, y = np.meshgrid(np.arange(0, 5, 1), np.arange(0, 5, 1))
    
    plt.quiver(x, y, ax, -ay)
    
    for xx in range(5):
      for yy in range(5):
        plt.text(xx, yy, '%5.1f' % (v[yy, xx]), horizontalalignment='center')
    
    plt.xticks(np.arange(0, 5, 1), np.arange(1, 6, 1))
    plt.yticks(np.arange(0, 5, 1), np.arange(1, 6, 1))
    plt.gca().invert_yaxis()
    
    plt.title('Gridworld $V(s)$ and $\pi(s)$')
    plt.xlabel('Column')
    plt.ylabel('Row')
    
    plt.show()
