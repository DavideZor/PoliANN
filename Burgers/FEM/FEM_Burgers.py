from fenics import *
import numpy as np
import pylab as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
    
# deg is the degree of the choosen piece-wise polynomial space
# N is the number of elements, not points
# T is the time interval width
# dt is the time step
# nu is the kinematic viscosity
# theta is the parameter of the theta-method for time integration
# visual is a logic variable used to plot the results


def FEM_Burgers(deg, N, T, dt, nu, theta, visual):

  # The elements are fist defined on the domain
  mesh = UnitIntervalMesh(N)

  # The funcion space of polynomials with global degree
  # less than or equal to deg is created using
  # Continuous Galerkin (CG) elements
  V = FunctionSpace(mesh, "CG", deg)

  # There is no forcing term in the viscous Burgers equation
  f =  Constant(0.0)

  # Definition of the BCs
  g_left = Constant(0.0)
  g_right = Constant(0.0)

  def left_boundary(x, on_boundary):
    return on_boundary and near(x[0], 0)

  def right_boundary(x, on_boundary):
    return on_boundary and near(x[0], 1)

  bc_left = DirichletBC(V, g_left, left_boundary)
  bc_right = DirichletBC(V, g_right, right_boundary)

  bc = [bc_left, bc_right]

  #  Initial condition
  u_init = Expression ("sin(pi*x[0])", pi = np.pi, degree = deg+1)

  u = Function(V)
  u_old = Function(V)
  v = TestFunction(V)

  # Projection of the initial condition on the function space V
  u.interpolate(u_init)
  u_old.assign(u)

  # UFL version of the constant dt
  DT = Constant(dt)

  # Weak formulation in the variational form
  F = (dot(u - u_old, v)/DT + (1 - theta) * nu * inner(grad(u), grad(v)) \
  + theta * inner (u_old * u_old.dx(0), v) \
  + theta * nu * inner(grad(u_old), grad(v)) \
  + (1 - theta) * inner (u * u.dx(0), v) - dot (f, v)) * dx

  # Initialization
  t = 0.0

  while t < T :

    t_instances = np.array(range(8))*0.25
    t_instances = t_instances.tolist()

    if any(np.isclose(t, t_instances)):
      if visual:

        tit = 'FEM with $\\nu = {:.4f}$'.format(nu)
        plot(u, label = r"t = {:.2f}".format(t))
        #plt.grid(linewidth = 0.1)
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.1)
        plt.xlabel('$x$')
        plt.ylabel('$u$')
        plt.title(tit)
        plt.legend(frameon=True)

    
    # Computation of the solution
    solve ( F == 0, u, bc )

    # Storage of the solution in a .txt file
    mesh_points = mesh.coordinates()
    x_coord = np.array(mesh_points[:,0])
    u_array = np.array([u(Point(x)) for x in mesh_points])
    tit = "FENICSUnsteadyBurgers{}".format(int(t/dt)) + ".txt"
    with open(tit, "w") as txt_file:
      txt_file.write('# t = {}'.format(t) + '\n')
      for line in range(len(x_coord)):
        txt_file.write('{}, {};'.format(x_coord[line], u_array[line]) + '\n')

    # Update
    t = t + dt
    u_old.assign(u)

  return u, t, mesh
