"""
Boids

Simulates flocking behavior
"""
import math
import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create new Figure and an Axes which fills it.
fig = plt.figure(figsize=(8, 8))
ax = fig.add_axes([0, 0, 1, 1], frameon=False)
ax.set_xlim(0, 1), ax.set_xticks([])
ax.set_ylim(0, 1), ax.set_yticks([])
x_target = random.uniform(0.0, 1.0)
y_target = random.uniform(0.0, 1.0)

# Simulation parameters
N = 200
WORLD_WIDTH = 1.0
REPULSION_LIMIT = WORLD_WIDTH / 50
WALL_REPULSION_LIMIT = WORLD_WIDTH / 10
REPULSION_STRENGTH = 0.005
WALL_REPULSION_STRENGTH = 0.003
ALIGNMENT_LIMIT = WORLD_WIDTH / 8
ATTRACTION_STRENGTH = 3e-6
ALIGNMENT_STRENGTH = 0.01

MAX_SPEED = WORLD_WIDTH / 200
assert REPULSION_LIMIT < ALIGNMENT_LIMIT

# Init boids
boids = np.zeros(N, dtype=[('pos', float, 2), ('vel', float, 2)])
boids['pos'] = np.random.uniform(WORLD_WIDTH * 0.1, WORLD_WIDTH * 0.9, (N, 2))
boids['vel'] = np.random.uniform(-WORLD_WIDTH / 200, WORLD_WIDTH / 200, (N, 2))

# Use a scatter plot to visualize the boids
scatter = ax.scatter(boids['pos'][:, 0], boids['pos'][:, 1])


def update_boids(xs, ys, xvs, yvs, frame):

    # Matrix off position difference and distance
    xdiff = np.add.outer(xs, -xs)
    ydiff = np.add.outer(ys, -ys)
    distance = np.sqrt(xdiff ** 2 + ydiff ** 2)

    # Matrix of velocity difference
    xvdiff = np.add.outer(xvs, -xvs)
    yvdiff = np.add.outer(yvs, -yvs)

    # Calculate the boids that are visible to every other boid
    angle_towards = np.arctan2(-ydiff, -xdiff)
    angle_vel = np.arctan2(yvs, xvs)
    angle_diff = angle_towards - angle_vel[:, np.newaxis]
    visible = np.logical_and(angle_diff < np.pi / 2, angle_diff > -np.pi / 2)

    # Repulse adjacent boids
    repulsion = np.clip(1.0 - distance / REPULSION_LIMIT, 0.0, 1.0) * visible
    repulsion_n = np.maximum(np.add.reduce(repulsion > 0.0).astype(float) - 1, 1)

    # Align with nearby boids
    alignment = (distance < ALIGNMENT_LIMIT).astype(float) * visible
    alignment_n = np.maximum(np.add.reduce(alignment) - 1, 1)
    xvs += np.sum(xvdiff * alignment, axis=0) * ALIGNMENT_STRENGTH / alignment_n
    yvs += np.sum(yvdiff * alignment, axis=0) * ALIGNMENT_STRENGTH / alignment_n

    # Attraction
    xvs += np.sum(xdiff * visible, axis=0) * ATTRACTION_STRENGTH / (N - 1)
    yvs += np.sum(ydiff * visible, axis=0) * ATTRACTION_STRENGTH / (N - 1)

    # Move towards target
    global x_target
    global y_target
    xvs += (x_target - xs) * 30 * ATTRACTION_STRENGTH
    yvs += (y_target - ys) * 30 * ATTRACTION_STRENGTH

    # Repulsion from nearby boids
    yvs += -np.sum(xdiff * repulsion, axis=0) * REPULSION_STRENGTH / repulsion_n
    xvs += -np.sum(ydiff * repulsion, axis=0) * REPULSION_STRENGTH / repulsion_n

    # Wall repulsion
    xvs += np.clip(1.0 - xs / WALL_REPULSION_LIMIT, 0.0, 1.0) * WALL_REPULSION_STRENGTH
    yvs += np.clip(1.0 - ys / WALL_REPULSION_LIMIT, 0.0, 1.0) * WALL_REPULSION_STRENGTH
    xvs -= np.clip(1.0 - (WORLD_WIDTH - xs) / WALL_REPULSION_LIMIT, 0.0, 1.0) * WALL_REPULSION_STRENGTH
    yvs -= np.clip(1.0 - (WORLD_WIDTH - ys) / WALL_REPULSION_LIMIT, 0.0, 1.0) * WALL_REPULSION_STRENGTH

    #xvs = np.clip(xvs, -MAX_SPEED, MAX_SPEED)
    #yvs = np.clip(yvs, -MAX_SPEED, MAX_SPEED)

    # Add some random noise to velocity
    x_noise = np.random.uniform(0.8, 1.2, N)
    y_noise = np.random.uniform(0.8, 1.2, N)
    xs += xvs * x_noise
    ys += yvs * y_noise


def animate(frame):

    update_boids(boids['pos'][:, 0],
                 boids['pos'][:, 1],
                 boids['vel'][:, 0],
                 boids['vel'][:, 1],
                 frame)

    scatter.set_offsets(boids['pos'])

def mouse_move(ev):
    global x_target
    global y_target
    x_target = ev.xdata
    y_target = ev.ydata

cid = fig.canvas.mpl_connect('button_press_event', mouse_move)
animation = FuncAnimation(fig, animate, interval=30)
plt.show()
