import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def animate_pso(Trajectory, f, S, interval=50):
    frames, N, D = Trajectory.shape
    assert D == 2  # IF evaluates to False, Python immediately halts execution

    fig, ax = plt.subplots()  # set base plot window
    ax.set_xlim(-S, S)
    ax.set_ylim(-S, S)

    # Built background contour of f in the windwo for better visualization.
    grid = np.linspace(-S, S, 2000)  # Store 200 partitions points b/w [-S, S]
    # ie. evenly spaced numbers between -S and S. (1, 200) array/mtx
    XX, YY = np.meshgrid(grid, grid)  # See example in defination.
    # meshgrid(partitions_on_x_axis, partitions_on_y_axis) returns 2 mtx
    # XX : store all x-coordinates of intersection points of grid line
    # YY : store all y-coordinates of intersection points of grid line
    # (XX[i, j], YY[i, j]) <- co-ordinates of grid postion (i, j)
    ZZ = np.array([[f(np.array([x, y])) for x in grid] for y in grid])
    # ZZ: nested list, stores value of f at each of (200 * 200) grid_mesh point.
    # for y  changes along outer nest, x changes along inner arrays.
    # ZZ[0] will give [coordinates] of all points with y = -S.
    ax.contourf(XX, YY, ZZ, levels=30, cmap='viridis', alpha=0.6)
    # conterf: draws filled counter plot : similar f-values get similar colors.
    # Colors are chosen from list "viridis" <- cmap
    # levels: how many discrete color bands.
    # more bands => smoother gradient look
    # alpha : sets transparency.
    # So particle markers drawn on top remain clearly visible.
    # -------------------
    # This codes runs only ones before anu animation.
    # And create Static backdrop that doesn't change frame to frame.

    swarm = ax.scatter(Trajectory[0, :, 0], Trajectory[0, :, 1],
                       c='red', s=20)
    # Trajectory[0, :, 0] <- all N particles' x-coordinates at frame 0.
    # Trajectory[0, :, 1] <- all N particles' y-coordinates at frame 0.
    # plt.scatter(x,y) draws all point on window.
    # where each p_i is placed at (x[i], y[i]).
    # We store all particle-objects in swarm,
    # swarm : N objects of instance 'Pathcollection'
    # scatter() is method of this instance to create these
    # objects at respective postitions.
    # Then we only update the positions afterwards using another
    # method of this instance called set_offset
    # So we do not need to redraw all pints for each update.

    def update(frame):
        # Trajectroy(frame) :(N, 2)position mtx of particle at respective frame
        swarm.set_offsets(Trajectory[frame])  # Updated the particle positions.
        # set_offsets : takes (N, 2) array. (won't work for (N, 3))
        # and just upadte the position of the
        # N objects, created by method scatter()
        ax.set_title(f'Frame {frame}')
        return swarm,
    ani = animation.FuncAnimation(fig, update, frames=frames,
                                  interval=interval, blit=False)
    return ani
