import matplotlib.pyplot as plt
from fenics import * 
from mshr import *
import time
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
    
def PoissonSolver(n, degree, f):

  domain_vertices = [Point(0.0, 1.0),
                   Point(-0.3, 0.5),
                   Point(-1.0, 0.5),
                   Point(-0.7, 0.0),
                   Point(-1, -0.5),
                   Point(-0.3, -0.5),
                   Point(0, -1),
                   Point(0.3, -0.5),
                   Point(1.0, -0.5),
                   Point(0.7, 0.0),
                   Point(1.0, 0.5),
                   Point(0.3, 0.5)]

  # Re-shaped star
  domain_vertices = [Point(0.35, 0.75),
                   Point(0.0, 0.75),
                   Point(0.15, 0.5),
                   Point(0, 0.25), 
                   Point(0.35, 0.25),
                   Point(0.5, 0),
                   Point(0.625, 0.25),
                   Point(1.0, 0.25),
                   Point(0.85, 0.5),
                   Point(1.0, 0.75),
                   Point(0.625, 0.75),
                   Point(0.5, 1.0)]

  domain = Polygon(domain_vertices)

  mesh = generate_mesh(domain, n)

  g_D = Expression('0.5 + exp(-(x[0]*x[0] + x[1]*x[1]))', degree = degree)

  # Re-shaped domain
  g_D = Expression('sin(pi*x[0]) * sin(pi * x[1])', pi = np.pi, degree = degree + 1)
  
  def boundary(x, on_boundary):
    return on_boundary

  V = FunctionSpace(mesh, 'CG', degree)

  bc = DirichletBC(V, g_D, boundary)

  u = TrialFunction(V)
  v = TestFunction(V)

  a = (dot(grad(u), grad(v))) * dx
  L = (f * v) * dx

  u = Function(V)

  solve(a == L, u, bc)

  return u, mesh
