import argparse
import numpy as np
import matplotlib.pyplot as plt

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_size", type=int, default=50)
    parser.add_argument("--obs", type=int, default=100)
    
    args = parser.parse_args()
    return args

def genarate_grid(size=50, num_obstacles=100):
    grid = np.zeros((size,size))
    coord = np.random.randint(0,size,(num_obstacles,2))
    for obs in coord:
        grid[obs[0],obs[1]] = 1
    
    ## Goal cannot be an obstacle
    grid[0,size-1] == 0
    plt.imshow(grid, cmap='gray')
    plt.show()
    return grid