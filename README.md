# rl_tile_encoding
Tile encoding for reinforcement learning Q value function approximation

This repository implements TD(lambda) algorithm using CMAC tiling as a linear function approximation. It also includes SARSA and Qlearning implementation, tested on the mountain car example.
The CMAC tiling is based on code written by Sridhar Mahadevan and Richard Sutton, with few modifications. Code can be found here - http://webdocs.cs.ualberta.ca/~sutton/MountainCar/MountainCar.html ("Mahadevan's earlier C++ version w/ X11")
Additional details can be found in Sutton & Barto. The implementation itself is very straight forward, once the idea is clear. 

Starting condition:
- we have a set of continuous variables.
- tabular Q learning may take too much memory and/or the discretization will degrade learning.
- we want to be able generalize between states.

Algorithm:
- decide on binning of the continuous variables
- for each variable, set up N tiles
- each tile will have different offset. The offset will be a number between 0 and the size of the variable bin.
- a state is represented by a tile.

Function approximation:
- assuming that we have N variables (V0.. VN-1), each has Bi bins, T tiles, and A actions, we will have a total of sum(Bi,i=1..N-1) * T * A functions.

Example:
- a single continuous variable ranging from 0.3 .. 2.0. We want to have 13 bins and 3 tiles. 
- size of each bin is (2.0 - 0.3) / 13 = 0.13
- set a (random) offset per tile. e.g.
	- offset(tile0) = 0 * 0.13
	- offset(tile1) = 0.4 * 0.13
	- offset(tile2) = 0.8 * 0.13
- for a value of 0.99 for the continuous variable, here is the value for each tile:
	- tile0: int((0.99 - 0.3 - 0 * 0.13) / 0.13)   = 5
	- tile1: int((0.99 - 0.3 - 0.4 * 0.13) / 0.13) = 4
	- tile2: int((0.99 - 0.3 - 0.8 * 0.13) / 0.13) = 4
	
- Note that having a single tile (N=1) is identical to tabular based method.




