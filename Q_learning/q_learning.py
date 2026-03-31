import numpy as np
from pprint import pprint
import random
random.seed(42)

# Defining the environment
envr = np.array([[-1, -1, 100],
                [-1, -1, -100]])
act_names = ['Up', 'Down', 'Right', 'Left']


# Seting the hyperparameters
al = 0.1  #"alpha"
gm = 0.9  #"gamma"
epn = 0.1  #"epsilon"
noe = 2000

# Initialize the Q-table
nstates = np.prod(envr.shape)
nactions = 4
Qtable = np.zeros((nstates, nactions))

# Define the helper functions
def get_state(rw, cl):
    number_cols = envr.shape[1]
    return rw * number_cols + cl

def get_rw_cl(state):
    number_cols = envr.shape[1]
    rw = state // number_cols
    cl = state % number_cols
    return rw, cl

def choose_action(state):
    if np.random.uniform() < epn:
        # Choose a random action
        act = np.random.randint(nactions)
    else:
        # Choose the best action based on the Q-table
        act = np.argmax(Qtable[state])
    return act

# Train the Q-table
for epi in range(noe):
    # Reset the envrironment for each episode
    state = get_state(1, 0)
    done = False
    
    while not done:
        # Choose an action
        act = choose_action(state)
        
        # Take the action and observe the next state and reward
        rw, cl = get_rw_cl(state)
        if act == 0:
            rw = max(rw - 1, 0)
        elif act == 1:
            rw = min(rw + 1, envr.shape[0] - 1)
        elif act == 2:
            cl = min(cl + 1, envr.shape[1] - 1)
        elif act == 3:
            cl = max(cl - 1, 0)
        next_state = get_state(rw, cl)
        reward = envr[rw, cl]
        
        # Update the Q-table
        Qtable[state, act] = (1 - al) * Qtable[state, act] + \
                                 al * (reward + gm * np.max(Qtable[next_state]))
        
        # Update the state and check if the episode is done
        state = next_state
        done = (reward == 100) or (reward == -100)
print("Training of the Artificial Intelligence Agent is Complete.")    
def path_policy(init_rw, init_cl, Qtable, envr):
    gr, gc = 0, 2  # preset goal to (0, 2)
    hr, hc = 1, 2  # preset hole to (1, 2)

    path = [(init_rw, init_cl)]

    if (init_rw, init_cl) == (gr, gc): 
        print("Error: Given GOAL")
    elif (init_rw, init_cl) == (hr, hc):
        print("Error: Given HOLE")
    else:
        print("Best Policy and Shortest Path taken by the Agent from Start Location ({}, {}):".format(init_rw, init_cl))

        while (init_rw, init_cl) != (gr, gc) and (init_rw, init_cl) != (hr, hc):
            state = get_state(init_rw, init_cl)
            action = np.argmax(Qtable[state])
            if action == 0:
                init_rw = max(init_rw - 1, 0)
            elif action == 1:
                init_rw = min(init_rw + 1, envr.shape[0] - 1)
            elif action == 2:
                init_cl = min(init_cl + 1, envr.shape[1] - 1)
            elif action == 3:
                init_cl = max(init_cl - 1, 0)
            path.append((init_rw, init_cl))

        # Print the corresponding actions for each state in the path
        print("Best Policy :")
        for i in range(len(path)-1):
            state = get_state(path[i][0], path[i][1])
            act = np.argmax(Qtable[state])
            print("State {}= {}   Action taken = {}".format(i+1, path[i], act_names[act]))

        # Print the goal or hole at the last state of the path
        finalstate = path[-1]
        if envr[finalstate[0], finalstate[1]] == 100:
            print("State {}= GOAL".format(i+2, path[i]))
        else:
            print("State {}= HOLE".format(i+2, path[i]))

        # Print the path taken to reach the goal
        if envr[finalstate[0], finalstate[1]] == 100:
            print("---------------------")
            print("The Quickest Route:")
            for state in path:
                if envr[state[0], state[1]] == 100:
                    print("Goal")
                    break
                else:
                    print(state,"-->")
        else:
            print("No path found to reach the goal.")
print("The Q-Table for the given envrironment is:") 
print(np.array2string(Qtable).replace('[[',' [').replace(']]',']'))
path_policy(0, 0, Qtable, envr)
path_policy(1, 0, Qtable, envr)