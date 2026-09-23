from Swarm_optimization_algorithm import Particle_swarm_optimization as pso
import Functions as fn
import Animation as anim
import matplotlib.pyplot as plt


functions = [fn.sphere, fn.ackley, fn.rastrigin,
            fn.goldstein_price, fn.levy_n13 ]


# CR, Trajectory = pso(
#   fn.sphere, N=30, S=5, a=0.7, b=1.5, b_hat=1.5, c=1.0, D=2, frames=200)
# ani = anim.animate_pso(Trajectory, fn.sphere, S=5)

# CR, Trajectory = pso(
#     fn.levy_n13, N=30, S=10, a=0.7, b=1.5, b_hat=1.5, c=1.0, D=2, frames=200)
# ani = anim.animate_pso(Trajectory, fn.levy_n13, S=10)

# plt.show()
# ani.save('pso_levy_n13.gif', writer='pillow')

for f in functions:
    f_name = f.__name__
    print(f"Running PSO optimization for: {f_name}...")


    CR, Trajectory = pso(
        f, N=30, S=10, a=0.7, b=1.5, b_hat=1.5, c=1.0, D=2, frames=200
    )

    ani = anim.animate_pso(Trajectory, f, S=10)

    output_filename = f"pso_{f_name}.gif"
    print(f"Saving animation to {output_filename}...")
    ani.save(output_filename, writer='pillow')

    # Close the current plot figure to free up memory before the next iteration
    plt.close()

print("done")
