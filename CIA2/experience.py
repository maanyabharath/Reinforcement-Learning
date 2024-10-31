import numpy as np
    
class Experience:
    def __init__(self):
        self.current = None
        self.replay = []
        self.actions = np.concatenate([np.eye(2),-1*np.eye(2)], axis=0)
    
    def start_new_episode(self):
        if self.current is not None:
            self.replay.append(self.current)
        # if self.current is not None and self.replay is None:
        #     addition = self.current.reshape(1,*(self.current.shape))
        #     self.replay = addition
        # elif self.current is not None and self.replay is not None:
        #     addition = self.current.reshape(1,*(self.current.shape))
        #     self.replay = np.concatenate([self.replay,addition], axis=0)
        # else:
        #     raise Warning("----------- Starting new episode replay -----------")
        
        self.current = None
        
    def store_set(self,SAR:dict):
        add = np.array(list(SAR.values())).reshape(1,-1)
        if self.current is not None:
            self.current = np.concatenate([self.current,add],axis=0)
        else:
            self.current = add
        
        return 1
    
    def get_latest_reward(self):
        total = np.sum(self.current[:,-1])
        return total
    
    def get_path(self):
        for path in self.replay:
            turns = path.shape[0]
            total_reward = np.sum(path[:,-1])
            if turns >= total_reward:
                yield path
    
    # def check_loop(self):
    #     if self.current.shape[0]>=3:
    #         sar = self.current[-3:,:]
    #         if (sar[0,0] == sar[2,0]) and sar[1,-1] < 0:
    #             return 1
    #     return 0
    
def check_loop(exp:Experience, next_state, reward):
    if exp.current is not None:
        for ex in exp.current:
            if ex[0] == next_state and ex[-1]>0:
                print("LOOP FOUND - ",ex, next_state, reward)
                return 1
    return 0