import numpy as np


def Particle_swarm_optimization(f, N=23, D=2, S=2, a=0.5, b=0.5, b_hat=0.5,
                                c=0.1, frames=1000):
    # f : Objective function to be optimized
    # N : Number of particles in the swarm
    # S : bound of the search space. i.e search space := [-S, S]^D
    # a , b, b_hat, c : tuning parameters.
    # D :  Dimensionality of the problem
    # frames : # no. of times particles jumps
    X = np.random.uniform(-S, S, size=(N, D))  # Initialize particle positions.
    # x <- (N,D) mtx, & i-th row ie. X[i] : position vector of a particle P_i.
    # Elements of i-th row are co-ordinates of position of P_i in space R^D.
    V = np.random.uniform(-1, 1, size=(N, D))  # Initialize particle velocities.
    # v <- (N,D) mtx, & i-th row := The velocity vector of a P_i.
    # Each element in row : velocity component in respective basis derection.
    score = np.apply_along_axis(f, 1, X)  # (1, N) mtx, i-th element is f(X[i])
    # apply_along_axis : takes each row(axis = 1) of X as input of f.
    # And returns N-array
    # with each i-th element representing value of f at position of P_i
    PR = X.copy()  # Initialize personal best positions. (personal record)
    # Initailly, each particle's personal best position is its initial position.
    # PR : (N,D) mtx; i-th row: location vector of personal best of P_i
    PR_score = score.copy()  # As initially PR = X.
    CR = X[np.argmin(score)]  # Initialize community best position.
    # argmin : Returns index of minima in given array (works for mtx too)
    # X[that index] : Returns the position vector
    # of the particle at Community best position.
    # (1, D) mtx
    CR_score = score.min()  # CR_score is a variable storing cummunity best.
    # Initialy, f(CR) = min{score}

    Trajectory = np.empty((frames, N, D))
    # 3-dimentional array. store X i.e. (N, D) mtx for each frame.
    # Trajectory[t] = X at t

    # compute for each frame in a loop.
    for t in range(frames):
        r, r_hat = np.random.uniform(0, 1, (2, N, D))

        # (crux)
        V = a * V + b * r * (PR - X) + b_hat * r_hat * (CR - X)
        # Calculate the velocity at that frame.
        # a : factor of inertial
        # b : weight to give for velecity toward personal best
        # b_hat :  weight to give for velecity toward community best
        # r , r_hat : stocastic factors
        # that changes direction of each component by some random value between (0,1)
        # r, r_hat, b, b_hat > 0 as those postions should attract the particle.
        # a > 0 as it shall retain some volocity due to inertia.

        X = X + c * V  # update the positions of all P_i
        # Here V is actually X_t+1 - X_t.
        # So if you assume that each frame is representing dt simulation time
        # Then V is actually velocity * dt.
        # We got rid of dt by mergeing its contribution value in coeefitient b.
        # So we do not need to care about dt. we only see  how X update
        # under the influence of PR and CR.

        # Note: Distance of particle from CR, PR is linearly prapotional with
        # the velocity of that particle. i.e. if (CR - X) decreases by 1 unit
        # then V decreaces by b_hat units. i.e. as particle gets close to
        # CR or PR their contribution in velocity decreases linearly.

        # So Praticle takes prapotionate jump in direction of CR and PR.
        # Particle at Far takes longer jump, particle nearby takes shorter jump.
        # with inertial on last jump and some random noice affecting the jump.

        # update PR and CR using boolean masking.
        score = np.apply_along_axis(f, 1, X)  # Scores at updated positions X.
        # (N, 1) mtx / array
        mask = score < PR_score  # creates blooean array (True/False) of len N.
        PR[mask] = X[mask]  # update personal best positions.
        # Set PR[i] = X[i] for all i s.t. i-th place of mask-array is true.
        # This is equvalent to 'run if loop-compare vales-replace',
        # but computationally more effitient.
        PR_score[mask] = score[mask]  # Update PR_scores i.e f(PR)
        best_idx = np.argmin(PR_score)
        # This stores index of the lowest (i.e. best) PR_score among all P_i.
        if PR_score[best_idx] < CR_score:  # Compare with CR_score
            CR_score = PR_score[best_idx]
            CR = PR[best_idx]
            # Update CR and CR_score

        Trajectory[t] = X
        # Stores positions of each particle i.e. X at each frame t

    return CR, Trajectory
