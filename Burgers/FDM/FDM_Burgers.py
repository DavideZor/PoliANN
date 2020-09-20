import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

# a is the coordinate of the first point
# b is the coordinate of the last point
# N is the number of intervals, not points
# T is the time interval width
# dt is the time step
# nu is the kinematic viscosity
# theta is the parameter of the theta-method (theta = 0.5)
# visual is a logic variable to plot the results

def FDM_Burgers(a, b, N, T, dt, nu, theta, visual):

  # Some ghost points are inserted not to exceed the stencil width
  n_stencil = 2

  dx = (b - a)/N

  ida = n_stencil # Python index of the first point a
  idb = n_stencil + N # Python index of the last point b

  # Mesh generation
  x = (np.arange(N + 2*n_stencil + 1) - n_stencil) * dx + a

  #Initial condition
  u_old = np.sin(np.pi * x)

  # The BCs are enforced on the ghost cells and on the boundary points
  u_old[0:ida + 1] = 0
  u_old[idb:] = 0

  # Initialization
  u = u_old
  t = 0.0

  # Time integration
  while t < T:

    t_instances = np.arange(8)*0.25
    t_instances = t_instances.tolist()

    if any(np.isclose(t, t_instances)):
      if visual:
        plt.figure(1)
        tit = 'FDM with $\\nu = {:.4f}$'.format(nu)
        plt.plot(x, u, label=r"$t = {:.2f}$".format(t))
        plt.grid(linewidth = 0.1)
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.1)
        plt.xlabel('$x$')
        plt.ylabel('$u$')
        plt.title(tit)
        plt.legend(frameon=True)

    # Time marching scheme
    t += dt

    if (t + dt > T):
      dt = T - t  

    ####################### Explicit advection #################################

    #low = (dt/dx)*(theta * (u_old[ida-1:idb]/2) + theta * (nu/dx))
    #dia = (1 + theta * 2 * nu * (dt/(dx ** 2)))*np.ones(N + 1)
    #upp = - (dt/dx) * (theta * (u_old[ida+1:idb+1]/2) - theta * (nu/dx))

    # Diaganola matrix on the RHS (called B in the thesis)
    #bb = (1 - (1 - theta) * (dt/dx) * (u_old[ida:idb+1] * ((u_old[ida+1:idb+2] \
    #    - u_old[ida-1:idb])/(2)) - nu * (u_old[ida+1:idb+2] \
    #    - 2*u_old[ida:idb+1] + u_old[ida-1:idb])/(dx)))*u_old[ida:idb+1]

    ############################################################################


    ######################## Theta method ######################################

    # Lower, diagonal and upper diagonals of the tridiagonal matrix of the LHS (A)
    low =  - (dt/dx) * theta * ((u_old[ida+1:idb+1]/2) + (nu/dx))
    dia = (1 + theta * 2 * nu * (dt/(dx ** 2)) + theta * (dt/(2*dx)) \
           * (u[ida+1:idb+2] - u[ida-1:idb]))
    upp = (dt/dx) * theta * ((u_old[ida:idb]/2) - (nu/dx))

    # Vector of the RHS (b)
    bb = (u_old[ida:idb+1] + (1 - theta) * (dt/dx) * (nu * (u_old[ida+1:idb+2] \
          - 2*u_old[ida:idb+1] + u_old[ida-1:idb])/(dx)) \
          + (1 - 2*theta) * (u_old[ida-1:idb] * u_old[ida:idb+1] - \
                             u_old[ida+1:idb+2] * u_old[ida:idb+1]))
    
    ############################################################################

    # Construction of the tridiagonal matrix
    A = np.diag(dia, 0) + np.diag(low, -1) + np.diag(upp, 1)

    # Solution of the linear system (the NumPy linear solver is used)
    u[ida:idb+1] = np.linalg.solve(A,bb)

    # Dirichlet BCs imposition
    u[0:ida + 1] = 0
    u[idb:] = 0

    # Solution update
    u_old = u

  return u, t, x
