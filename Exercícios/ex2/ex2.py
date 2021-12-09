"""ELE2761 Exercise 2: SARSA.

Implements the provided functionality to be used in your solution.

FUNCTIONS
    check      -- Check SARSA class for errors.
    discretize -- Discretize and clip a continuous value.
    drawip     -- Draw inverted pendulum system.
    pendulum   -- Inverted pendulum system.
    test       -- Test trained SARSA object.
    train      -- Train SARSA object.
"""

import copy
from math import pi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
from IPython import display

plt.rcParams["animation.html"] = "jshtml"

__EPS = 1e-10

def discretize(v, minv, maxv, numstates):
    """Discretize and clip a continuous value.

       s = discretize(v, minv, maxv, numstates) discretizes v,
       from domain [minv, maxv] into a discrete state s in range
       [0, numstates-1].
    """

    v = np.asarray(v)
    minv = np.asarray(minv)
    maxv = np.asarray(maxv)
    numstates = np.asarray(numstates)

    myx = np.maximum(np.minimum(v, maxv), minv)
    myx = (myx - minv)/(maxv-minv+__EPS)
    s = np.fix(myx * numstates)

    return np.asarray(s, 'int')

def drawip(x, u=[]):
    """Draw inverted pendulum system.

       drawip(x) draws the inverted pendulum system at state x.

       drawip(x, u) additionally draws the actuation u.

       NOTE
           Rows of x and u are plotted as an animation.
    """

    x = np.asarray(x)
    if x.ndim == 1:
        x = np.expand_dims(x, axis=0)
    u = np.asarray(u)
    if u.ndim == 0:
        u = np.expand_dims(u, axis=0)

    frames = x.shape[0]

    if np.size(u) > 0 and np.size(u) != frames:
        print("State and actuation sizes do not match")
    
    # Initialize figure
    fig, ax = plt.subplots()
    ax.axis([-1,1,-1,1])
    pole, = ax.plot([],[])
    torque, = ax.plot([],[])

    def animate(i):
        """Draw one frame of animation."""
        pole.set_data([0, np.cos(x[i,0]+pi/2)], [0, np.sin(x[i, 0]+pi/2)])

        if np.size(u) > 0:
            ang = 2*pi*u[i]/12
            segment = np.linspace(0, ang, 10)
            xc = 0.25*np.cos(segment + x[i,0] + pi/2)
            yc = 0.25*np.sin(segment + x[i,0] + pi/2)

            torque.set_data(xc, yc)

    if frames > 1:
        ani = matplotlib.animation.FuncAnimation(fig, animate, frames=frames, repeat=True, repeat_delay=1000, interval=50)
        plt.close(fig)
        return ani
    else:
        animate(0)

def pendulum(x, u):
    """Inverted pendulum system.

       xp = pendulum(x, u) returns the next state xp of the pendulum using
       previous state x = [theta, dtheta] and action u = [motor_voltage].
    """

    xp = __transition(__ipdynamics, x, u)
    xp[0] = ((xp[0]+pi) % (2*pi)) - pi

    return xp

def __ipdynamics(x, u):
    """Inverted pendulum dynamics.

       xc = ipdynamics(x, u) returns the state vector derivative given a
       current state x = [theta, dtheta] and action u = [motor_voltage].
    """
    J = 0.000191
    m = 0.055
    g = 9.81
    l = 0.042
    b = 0.000003
    K = 0.0536
    R = 9.5

    a = x[0]
    ad = x[1]
    u = np.squeeze(u)

    add = (1/J)*(m*g*l*np.sin(a)-b*ad-(K*K/R)*ad+(K/R)*u)

    return np.asarray([ad, add])

def __transition(eom, x, u):
    """Calculate transition from dynamics.

    xp = transition(eom, x, u) generates the next state xp starting from
    state x and applying actuation u by integrating the dynamics given by
    eom. eom is a function handle taking x and u and returning xdot.
 
    EXAMPLE
       xp = transition(ipdynamics, x, u)
    """

    if np.any(~np.isfinite(x)):
        print("Cowardly refusing to integrate invalid state")
    
    if np.any(~np.isfinite(u)):
        print("Cowardly refusing to integrate invalid action")

    h = 0.05

    # Trapezoid
    d1 = eom(x, u)
    xp = x + h*d1
    d2 = eom(xp, u)
    xp = x + h*(d1+d2)/2

    return xp

# https://stackoverflow.com/questions/43086557/convolve2d-just-by-using-numpy
def __conv2d(a, f):
    s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
    strd = np.lib.stride_tricks.as_strided
    subM = strd(a, shape = s, strides = a.strides * 2)
    return np.einsum('ij,ijkl->kl', f, subM)

def __movavg(a, n=3):
    """Returns a moving average filtered version of matrix a.

       a can be either one or two-dimensional."""
    if a.ndim == 1:
        expanded = True
        a = np.expand_dims(a, axis=0)    
        f = np.full((1, n), 1./n)
    else:
        expanded = False
        f = np.full((n, n), 1./n**2)

    c = __conv2d(a, f)
    m = copy.deepcopy(a)
    d = np.subtract(a.shape, c.shape)
    m[int(np.floor(d[0]/2)):-int(np.ceil(d[0]/2)), int(np.floor(d[1]/2)):-int(np.ceil(d[1]/2))] = c

    if expanded:
        m = np.squeeze(m, axis=0)

    return m

def check(sarsa):
    """Check SARSA class for basic errors."""
    # Initialization
    learner = sarsa()
    learner.maxvoltage = 3

    print('Sanity checking SARSA class')

    # Parameters
    if learner.epsilon <= 0 or learner.epsilon >= 1:
        print('Random action rate out of bounds, check __init__/self.epsilon')
        return
    if learner.gamma <= 0 or learner.gamma > 1:
        print('Discount rate out of bounds, check __init__/self.gamma')
        return
    if learner.alpha <= 0 or learner.alpha > 1:
        print('Learning rate out of bounds, check __init__/self.alpha')
        return
    if learner.pos_states <= 1 or learner.pos_states > 1000:
        print('Number of discretized positions out of bounds, check __init__/self.pos_states')
        return
    if learner.vel_states <= 1 or learner.vel_states > 1000:
        print('Number of discretized velocities out of bounds, check __init__/self.vel_states')
        return
    if learner.actions <= 1 or learner.actions > 100:
        print('Number of actions out of bounds, check __init__/self.actions')
        return

    print('...Parameters are within bounds')

    # Q value initialization
    if not hasattr(learner, 'init_Q'):
        print('Q value initialization unimplemented, implement init_Q')
        return

    learner.init_Q()

    if not hasattr(learner, 'Q') or learner.Q.ndim == 0:
        print('Q value initialization unimplemented, check init_Q')
        return
    if learner.Q.ndim != 3:
        print('Q dimensionality error, check init_Q/Q')
        return
    if not np.all(learner.Q.shape==(learner.pos_states, learner.vel_states, learner.actions)):
        print('Q size error, check init_Q/Q')
        return

    print('...Q value dimensionality OK')

    # State discretization
    x0 = [0, 0]

    if not hasattr(learner, 'discretize_state'):
        print('State discretization unimplemented, implement discretize_state')
        return

    s = learner.discretize_state(x0)

    if s is None:
        print('State discretization unimplemented, check discretize_state')
        return

    print('...State discretization is implemented')

    # Position discretization
    pc = np.arange(-5, 5, 0.1)
    pd = np.zeros(pc.shape)
    for pp in range(len(pc)):
        x0[0] = pc[pp]
        s = learner.discretize_state(x0)
        pd[pp] = s[0]
    
    if np.any((pd < 0) | (pd >= learner.pos_states) | (pd-np.fix(pd) != 0)):
        print('Position discretization out of bounds, check discretize_state/s[0]')
        return

    print('......Position discretization is within bounds')

    fig, axs = plt.subplots(2,3)

    plt.subplot(2, 3, 1)
    plt.plot(pc, pd)
    plt.title('Position discretization')
    plt.xlabel('Continuous position')
    plt.ylabel('Discrete position')
    plt.tight_layout()
    fig.canvas.draw()

    # Velocity discretization
    vc = np.arange(-50, 50, 0.1)
    vd = np.zeros(vc.shape)
    for vv in range(len(vc)):
        x0[1] = vc[vv]
        s = learner.discretize_state(x0)
        vd[vv] = s[1]

    if np.any((vd < 0) | (vd >= learner.vel_states) | (vd-np.fix(vd) != 0)):
        print('Velocity discretization out of bounds, check discretize_state/s[1]')
        return
    
    print('......Velocity discretization is within bounds')

    plt.subplot(2, 3, 2)
    plt.plot(vc, vd)
    plt.title('Velocity discretization')
    plt.xlabel('Continuous velocity')
    plt.ylabel('Discrete velocity')
    plt.tight_layout()
    fig.canvas.draw()

    # Action execution
    if not hasattr(learner, 'take_action'):
        print('Action execution unimplemented, implement take_action')
        return

    u = learner.take_action(0)

    if u is None:
        print('Action execution unimplemented, check take_action')
        return

    u = np.zeros(learner.actions)
    for aa in range(learner.actions):
        u[aa] = learner.take_action(aa)

    if np.any((u < -learner.maxvoltage) | (u > learner.maxvoltage)):
        print('Action out of bounds, check take_action/u')
        return

    plt.subplot(2, 3, 3)
    plt.plot(np.arange(learner.actions), u, '.')
    plt.title('Action execution')
    plt.xlabel('Action')
    plt.ylabel('Applied voltage')
    plt.tight_layout()
    fig.canvas.draw()

    print('...Action execution is within bounds')

    # Reward observation
    if not hasattr(learner, 'observe_reward'):
        print('Reward observation unimplemented, implement observe_reward')
        return

    r = learner.observe_reward(0, s)
    if r is None:
        print('Reward observation unimplemented, check observe_reward')

    r = np.zeros((learner.pos_states,learner.vel_states,learner.actions))
    for pp in range(learner.pos_states):
        for vv in range(learner.vel_states):
            for aa in range(learner.actions):
                s = [pp, vv]
                r[vv,pp,aa] = learner.observe_reward(aa, s)

    plt.subplot(2, 3, 4)
    x, y = np.meshgrid(np.arange(learner.pos_states), np.arange(learner.vel_states))
    plt.contourf(x, y, np.mean(r, axis=2))
    plt.colorbar()
    plt.title('Average reward')
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.tight_layout()
    fig.canvas.draw()

    print('...Reward observation is implemented')

    # Termination criterion
    if not hasattr(learner, 'is_terminal'):
        print('Termination criterion unimplemented, implement is_terminal')
        return

    t = learner.is_terminal(s)

    if t is None:
        print('Termination criterion unimplemented, check is_terminal')
        return

    t = np.zeros((learner.pos_states,learner.vel_states))
    for pp in range(learner.pos_states):
        for vv in range(learner.vel_states):
            s = [pp, vv]
            t[vv,pp] = learner.is_terminal(s)

    t = t>0+0

    plt.subplot(2, 3, 5)
    x, y = np.meshgrid(np.arange(learner.pos_states), np.arange(learner.vel_states))
    plt.contourf(x, y, t)
    plt.colorbar()
    plt.title('Termination criterion')
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.tight_layout()
    fig.canvas.draw()

    print('...Termination criterion is implemented');

    # Action selection
    if not hasattr(learner, 'execute_policy'):
        print('Action selection unimplemented, implement execute_policy')
        return

    a = learner.execute_policy(s)

    if a is None:
        print('Action selection unimplemented, check execute_policy')
        return

    for pp in range(learner.pos_states):
        for vv in range(learner.vel_states):
            s = [pp, vv]
            a = learner.execute_policy(s)
            if a is None or a < 0 or a > learner.actions or (a-np.fix(a)) != 0:
                print('Action selection out of bounds, check execute_policy/a')
                return

    print('...Action selection is within bounds')

    # Q update rule
    if not hasattr(learner, 'update_Q'):
        print('Termination criterion unimplemented, implement update_Q')
        return

    Qprev = copy.deepcopy(learner.Q)
    learner.update_Q(s, a, 100, s, a)
    if np.all(learner.Q == Qprev):
        print('Q update rule unimplemented, check update_Q')
        return
    
    print('...Q update rule is implemented')

    print('Sanity check successfully completed')

def test(learner):
    """Executes testing episode using SARSA class."""

    if not hasattr(learner, 'Q') or learner.Q is None:
        raise ValueError('Must be called with pre-trained SARSA object')
    
    # Initialize a new trial
    x = learner.initial_state
    s = learner.discretize_state(x)
    a = learner.execute_policy(s)

    xall = np.empty((0, 2))
    uall = np.empty((0, 1))

    # Inner loop: simulation steps
    for tt in range(int(learner.simtime/learner.simstep)):
        # Take the chosen action
        u = max(min(learner.take_action(a), learner.maxvoltage), -learner.maxvoltage)

        # Simulate a time step
        x = pendulum(x, u)

        s = learner.discretize_state(x)
        a = learner.execute_policy(s)

        # Accumulate visualization data
        xall = np.vstack((xall, x))
        uall = np.vstack((uall, u))

        # Stop trial if state is terminal
        if learner.is_terminal(s):
            break

    return drawip(xall, uall)

def train(learner):
    """Executes learning episodes using SARSA class."""

    learner.simtime = 3;             # Trial length (s)
    learner.simstep = 0.03;          # Simulation time step
    learner.maxvoltage = 3;          # Maximum applicable voltage
    learner.initial_state = [pi, 0]; # Start in down position

    # Initialize bookkeeping (for plotting only)
    ra = np.zeros(learner.trials)
    tta = np.zeros(learner.trials)
    [xx, yy] = np.meshgrid(np.linspace(-pi, pi, learner.pos_states), np.linspace(-12*pi, 12*pi, learner.vel_states))
    first = True
    fig, axs = plt.subplots(2,2)

    # Initialize value function
    learner.init_Q()

    # Outer loop: trials
    for ii in range(learner.trials):
        # Initialize a new trial
        x = learner.initial_state
        learner.e = np.zeros((learner.pos_states, learner.vel_states, learner.actions))
        s = learner.discretize_state(x)
        a = learner.execute_policy(s)

        # Inner loop: simulation steps
        for tt in range(int(learner.simtime/learner.simstep)):
            # Take the chosen action
            u = max(min(learner.take_action(a), learner.maxvoltage), -learner.maxvoltage)

            # Simulate a time step
            x = pendulum(x, u)
            
            sP = learner.discretize_state(x)
            r = learner.observe_reward(a, sP)
            aP = learner.execute_policy(sP)
            learner.update_Q(s, a, r, sP, aP)
            
            # Back up state and action
            s = sP
            a = aP
            
            # Keep track of cumulative reward
            ra[ii] = ra[ii]+r
            
            # Stop trial if state is terminal
            if learner.is_terminal(s):
                break
        
        tta[ii] = tta[ii] + tt*learner.simstep
        
        # Update plot every ten trials
        if ii % 100 == 99:
            val = np.amax(learner.Q, axis=2)
            pos = np.asarray(np.argmax(learner.Q, axis=2), float)

            # Value function
            data = np.transpose(val)
            if first:
                axs[0,0].imshow(data, origin='lower')
                axs[0,0].set_title('$V = max_a(Q(s, a))$')
                axs[0,0].set_xlabel('Position')
                axs[0,0].set_ylabel('Velocity')
            else:
                axs[0,0].images[0].set_data(data)

            # Policy
            data = __movavg(np.transpose(pos), 3)
            if first:
                axs[0,1].imshow(data, origin='lower')
                axs[0,1].set_title('$\pi = argmax_a(Q(s, a))$')
                axs[0,1].set_xlabel('Position')
                axs[0,1].set_ylabel('Velocity')
            else:
                axs[0,1].images[0].set_data(data)

            # Reward
            data = (np.arange(ii+1), __movavg(ra[0:ii+1], 9))
            if first:
                axs[1,0].plot(data)
                axs[1,0].set_xlim([0, learner.trials])
                axs[1,0].set_title('Progress')
                axs[1,0].set_xlabel('Trial')
                axs[1,0].set_ylabel('Cumulative reward')
            else:
                axs[1,0].lines[0].set_data(data)
                axs[1,0].set_ylim([np.amin(data[1])-0.1, np.amax(data[1])+0.1])

            # Trial duration
            data = (np.arange(ii+1), __movavg(tta[0:ii+1], 9))
            if first:
                axs[1,1].plot(data)
                axs[1,1].set_xlim([0, learner.trials])
                axs[1,1].set_ylim([0, learner.simtime])
                axs[1,1].set_title('Progress')
                axs[1,1].set_xlabel('Trial')
                axs[1,1].set_ylabel('Trial duration')
            else:
                axs[1,1].lines[0].set_data(data)

            if first:
                plt.tight_layout()

            fig.canvas.draw()
            if matplotlib.get_backend().find('inline') != -1:
                display.clear_output(wait=True)
                display.display(fig)

            first = False

    plt.close()
