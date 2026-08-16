import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random

BOX = 1000  # This give grid size of 1000x1000, you can change it to any value you want
dt = 0.1  # Time step for the simulation
frames = 500  # Number of frames in the animation
interval = 10  # Each time step is animated every 10 milliseconds


def v_func(vx, vy, t):
    # Example: Velocity function of time -- customize freely
    # Add some random noise to the x-velocity
    vx = vx + np.sin(np.exp(t)) + 0.5 * random.random() - 0.25
    # Add some random noise to the y-velocity
    vy = vy + np.cos(np.exp(t)) + 0.5 * random.random() - 0.25
    return vx, vy


class Particle:
    def __init__(self, x, y, t=0.0):
        self.x = x
        self.y = y
        self.t = t
        self.vx = 1.0
        self.vy = 1.0

    # This function updates the position of the particle based 
    # On its velocity and time step.    
    def step(self, dt):
        self.vx, self.vy = v_func(self.vx, self.vy, self.t)
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.t += dt
        self._reflect()

    # This function checks if the particle has hit the walls of the box.
    # In discrete time step particle may overshoot the wall,
    # we deal with this by reflecting particle back from the wall by
    # the overshoot distance. and also,
    # reversing the velocity component in that direction.
    def _reflect(self):
        if self.x < 0:
            self.x = -self.x
            self.vx = -self.vx
        elif self.x > BOX:
            self.x = 2 * BOX - self.x
            self.vx = -self.vx

        if self.y < 0:
            self.y = -self.y
            self.vy = -self.vy
        elif self.y > BOX:
            self.y = 2 * BOX - self.y
            self.vy = -self.vy


# Create a list of particles with initial positions.
particles = [Particle(x=500, y=500),
             Particle(x=500, y=500), Particle(x=800, y=200)]

# Set base plot and axis limits.
fig, ax = plt.subplots()
ax.set_xlim(0, BOX)
ax.set_ylim(0, BOX)
ax.set_aspect('equal')

colors = ['ro', 'bo', 'go']

# Create a list of points for each particle with different colors
# Here elements of list-points are objects of type Line2D. But note that
# ax.plot() returns a list of Line2D objects, we need to extract the first 
# Element of that list to get the Line2D object for each particle. 
# Hence we use [0] at the end of ax.plot().
points = [ax.plot([], [], c, markersize=5)[0] for c, _ in
          zip(colors, particles)]


# frame is just a counter that goes from 0 to frames-1.
# frame is not a subplot layer. and following fuction does not plot 
# any new graph or even points. 
# It just update the position of the existing points in the plot.
# particle <- variable. point <- Line2D object that we see on the plot.
def update(frame):
    for particle, point in zip(particles, points):
        particle.step(dt)
        point.set_data([particle.x], [particle.y])
    return points


# This function runs update in loop for no. of 'frames'  times and
# creates an animation.
# Here each frame is displayed for interval milliseconds.
# blit=True means that only the parts of the plot;
# which have changed will be redrawn.
# i.e. only the points will be redrawn, not the entire plot.
ani = animation.FuncAnimation(
    fig, update, frames=frames, interval=interval, blit=True)
plt.show()  # Display the animation.
