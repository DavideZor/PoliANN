import numpy as np
import pylab as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
    
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

# a is the coordinate of the first point
# b is the coordinate of the last point
# N is the number of intervals, not points
# T is the time interval width
# dt is the time step
# nu is the kinematic viscosity
# visual is a logic variable used to plot the results

def FVM_Burgers(a, b, N, T, dt, nu, visual):

  # Since the advection term is updated explicitly, the CFL condition
  # related to advection must be satisfied
  CFL = np.pi * N * dt

  if CFL > 1:
    CFL = 0.8
    dt = CFL/(N * np.pi)

  # Number of ghost cells used in order not to exceed the computational
  # stencil
  n_stencil = 2

  dx = (b - a)/N

  ida = n_stencil       # Python index for the first point
  idb = n_stencil + N   # Python index for the last point

  x = (np.arange(N + 2 * n_stencil + 1) - n_stencil) * dx + a

  # Initial condition
  u_init = np.sin(np.pi * x)

  # Dirichlet BCs
  u_init[0:ida + 1] = 0
  u_init[idb:] = 0

  # Initialization
  u_old = u_init
  u = u_old
  t = 0.0

  # Time integration
  while t < T:

    if (t + dt > T):
      dt = T - t

    t_instances = np.array(range(8))*0.25
    t_instances = t_instances.tolist()

    if any(np.isclose(t, t_instances)):
      if visual:

        plt.figure(1)
        tit = 'FVM with $\\nu = {:.4f}$'.format(nu)
        plt.plot(x, u, label=r"$t = {:.2f}$".format(t))
        plt.grid(linewidth = 0.1)
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.1)
        plt.xlabel('$x$')
        plt.ylabel('$u$')
        plt.title(tit)
        plt.legend(frameon=True)

    # The source term is evaluated in the previous time step
    laplacian = np.zeros(N + 2*n_stencil + 1)

    laplacian[ida:idb+1] = (u_old[ida-1:idb] - 2.0*u_old[ida:idb+1] \
                            + u_old[ida+1:idb+2])/dx**2

    # First order derivatives evaluated through a centered finite difference
    cfd = np.zeros(N + 2*n_stencil + 1)
    cfd[ida:idb+1] = 0.5*(u_old[ida+1:idb+2] - u_old[ida-1:idb]) 
    
    # Construction of the interface states that include the diffusion term
    uL = np.zeros(N + 2*n_stencil + 1)
    uR = np.zeros(N + 2*n_stencil + 1)

    uR[ida:idb+1] = u_old[ida+1:idb+2] - 0.5*(1.0 + u_old[ida+1:idb+2]*dt/dx)\
                    *cfd[ida+1:idb+2] + 0.5*dt*laplacian[ida+1:idb+2]

    uL[ida+1:idb+2] = u_old[ida:idb+1] + 0.5*(1.0 - u_old[ida:idb+1]*dt/dx)\
                      *cfd[ida:idb+1] + 0.5*dt*laplacian[ida:idb+1]

    # Riemann problem at the interface
    # Shock
    S = (uL + uR)/2
    u_shock = np.where(S > 0.0, uL, uR)
    u_shock = np.where(S == 0.0, 0.0, u_shock)

    # Rarefaction
    u_rare = np.where(uR <= 0.0, uR, 0.0)
    u_rare = np.where(uL >= 0.0, uL, u_rare)

    # Solution at the interface
    u_interf = np.where(uL > uR, u_shock, u_rare)

    # Advective update
    F = 0.5*u_interf*u_interf
    advection = np.zeros(N + 2*n_stencil + 1)
    advection[ida:idb+1] = (F[ida:idb+1] - F[ida+1:idb+2])/dx

    # Lower, diagonal and upper diagonals of the left-hand side
    low = -0.5 * (nu * dt/(dx**2)) * np.ones(N)
    low[0] = 0.0
    dia = (1 + (nu * dt/(dx**2))) * np.ones(N)
    upp = -0.5 * (nu * dt/(dx**2)) * np.ones(N)
    upp[N-1] = 0

    # Right-hand side of the linear system
    bb = 0.5 * nu * dt * laplacian
    bb = bb[ida:idb]
    bb = bb + u_old[ida:idb] + dt * advection[ida:idb]

    # Solution of the linear system (the NumPy solver is used)
    A = np.matrix([low, dia, upp])
    u[ida:idb] = linalg.solve_banded((1,1), A, bb)

    # Dirichlet BCs imposition
    u[0:ida+1] = 0
    u[idb:] = 0

    u_old = u

    t = t + dt

  return u, t, x
