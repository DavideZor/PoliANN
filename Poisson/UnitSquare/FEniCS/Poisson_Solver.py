import matplotlib.pyplot as plt
from fenics import * 
import time
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
    
def PoissonSolver(n, degree, f):

  g_D = Constant(0.0)

  mesh = UnitSquareMesh(n, n, 'crossed')
  
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
