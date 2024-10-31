import numpy as np
import argparse
from experience import Experience, check_loop
import matplotlib.pyplot as plt
from utils import parse, genarate_grid

GAMMA = 0.7
LR = 0.5
LOOP_REWARD = -5

np.random.seed(10)

def epsilon_greedy(q_value, prob=0.8, current_action=True):
    rand = np.random.rand()
    if prob<rand:
        if current_action:
            value = np.argmax(q_value)
        else:
            value = np.max(q_value)
    else:
        if current_action:
            value = np.random.choice(np.arange(q_value.size))
        else:
            value = np.random.choice(q_value)
            
    return value
    
    
def get_reward(state_action:dict, **kwargs):

    next = state_action["state"]+state_action["action"]
    print(state_action["state"],next)
    
    if np.any(next < [0,0]) or np.any(next > [kwargs["args"].grid_size-1,kwargs["args"].grid_size-1]):
        return -10, "failed"
    elif kwargs["maze"][next[0],next[1]]:
        return -10, "obstacle"
    elif np.all(next == [0, kwargs["args"].grid_size-1]):
        return 10, "finished"
    else:
        return 1, "success"
    
def episode(maze, current_position, experience_buffer:Experience, Q_value, *args, **kwargs):
    ACTIONS = np.concatenate([np.eye(2,dtype=np.int8),-1*np.eye(2,dtype=np.int8)], axis=0, dtype=np.int8) ## [1,0] - right, [0,1] - up, [-1,0] - left, [0,-1] - down (4,2)
    
    for i in range(5000):
        current_action = epsilon_greedy(Q_value[current_position[0],current_position[1],:])#np.argmax(Q_value[current_position[0],current_position[1],:]) ## Max of (4,)
        print(f"ACTIONS TAKEN - {ACTIONS[current_action]}")
        x,y = current_position+ACTIONS[current_action]
        
        ## Check loop and assign values
        current_reward, flag = get_reward({"state":current_position, "action":ACTIONS[current_action]}, maze=maze, args=kwargs["args"])#is_obstacle=bool(is_obstacle))
        if check_loop(experience_buffer, kwargs["args"].grid_size*x + y, current_reward):
            current_reward = LOOP_REWARD
            flag = "failed"
        
        print(f"REWARD RECEIVED - {current_reward}")
        
        ## Check for termination
        if flag == "finished":
            return 1
        elif flag == "failed":
            position_int = kwargs["args"].grid_size*current_position[0] + current_position[1]
            experience_buffer.store_set({"state":position_int, "action": current_action, "reward":current_reward})
            next_state_Q = 0
            current_Q = Q_value[current_position[0],current_position[1],current_action]
            current_Q = current_Q + LR*(current_reward + GAMMA*next_state_Q - current_Q)
            Q_value[current_position[0],current_position[1],current_action] = current_Q
            print("Q-VALUES - ", Q_value[current_position[0],current_position[1],:])
            break
        elif flag == "obstacle":
            x,y = None, None
            position_int = kwargs["args"].grid_size*current_position[0] + current_position[1]
            experience_buffer.store_set({"state":position_int, "action": current_action, "reward":current_reward})
        else:
            position_int = kwargs["args"].grid_size*current_position[0] + current_position[1] ## R*grid_size + C
            experience_buffer.store_set({"state":position_int, "action": current_action, "reward":current_reward})
        
        ## Update the Q-Value
        if x is None and y is None:
            next_state_Q = 0
            current_Q = Q_value[current_position[0],current_position[1],current_action]
            current_Q = current_Q + LR*(current_reward + GAMMA*next_state_Q - current_Q)
            Q_value[current_position[0],current_position[1],current_action] = current_Q
            print("Q-VALUES - ", Q_value[current_position[0],current_position[1],:])
        else:
            next_state_Q = epsilon_greedy(Q_value[x,y,:], current_action=False)#np.max(Q_value[x,y,:])  
            current_Q = Q_value[current_position[0],current_position[1],current_action]
            current_Q = current_Q + LR*(current_reward + GAMMA*next_state_Q - current_Q)
            Q_value[current_position[0],current_position[1],current_action] = current_Q
            print("Q-VALUES - ", Q_value[current_position[0],current_position[1],:])
            current_position = np.array([x,y]) ## Go to the next state
            
        print("POS - ",current_position,end="\n")
        if i%1000 == 0:
            print(f"ITERATION - {i}\n")
    
    return


def main():
    # Get Attrs
    args = parse()
    #Genarete the grid
    maze = genarate_grid(args.grid_size, args.obs)
    current_agent_location = np.array([args.grid_size-1,0],dtype=np.int8) # (x,y)
    experience = Experience()
    Q_value = np.abs(np.random.normal(0,5,(args.grid_size,args.grid_size,4)))
    num_episode = 1
    while(True):
        status = episode(maze,current_agent_location, experience, Q_value, args=args)
        if status:
            print("The destination has been reached")
            states = experience.replay[-1]
            np.save("SARSA_100x100.npy",states)
            break
        total_reward = experience.get_latest_reward()
        print(f"EPISODE {num_episode} REWARD - ", total_reward, end="\n\n")
        num_episode += 1
        experience.start_new_episode()
        
    
if __name__ == "__main__":
    main()