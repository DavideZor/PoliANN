from fenics import * 
from mshr import *
import matplotlib.pyplot as plt
import numpy as np

# Definition of the geometry

N = 150     # Discretization resolution (N = 1/h)

L = 15.0    # Domain length
H = 2.0     # Domain height

A = 2.0     # Distance of the vertical plane 
            # from the y axis (the inflow)
B = 1.0     # Distance of the horizontal plane 
            # from the x axis (the floor)

# Definition of the domain vertices
domain_vertices = [Point(0, a),
                   Point(b, a),
                   Point(b, 0),
                   Point(L, 0),
                   Point(L, H),
                   Point(0, H)]

domain = Polygon(domain_vertices)

# Domain meshing
mesh = generate_mesh(domain, N)

# Problem definition
nu = 0.01

# Forcing term
f = Constant((0.0, 0.0))

# Definition of the function spaces for the velocity (V)
# and for the pressure (Q). In this case piecewise
# linear polynomials are used for the pressure and
# quadratic polynomials are used for the velocity
# (Taylor - Hood elements)
degree = 1

V = VectorElement('CG', mesh.ufl_cell(), degree + 1)
Q = FiniteElement('CG', mesh.ufl_cell(), degree)

# A new mixed variable x = [u, p] is defined. This 
# variable belongs to the space X = [V, Q]
X = FunctionSpace(mesh, MixedElement([V, Q]))

# Definition of the domain boundaries
def zero_boundary(x, on_boundary):
  return on_boundary and (near(x[1], 0) or near(x[1], H))

def inflow(x, on_boundary):
  return on_boundary and near(x[0], 0)

def outflow(x, on_boundary):
  return on_boundary and near(x[0], L)

def vertical_wall(x, on_boundary):
  return on_boundary and near(x[0], A) and (x[1] < B)

def horizontal_wall(x, on_boundary):
  return on_boundary and near(x[1], B) and (x[0] < A)

def pressure_point(x, on_boundary):
  return on_boundary and near(x[0], L) and near(x[1], 0)

# Definition of the velocity parabolic
# profile at the inlet
velocity_profile = Expression(('(4/((up - low)*(up - low)))*\
                              (x[1] - low) * (up - x[1])',
                              '0'), low = B, up = H, 
                              degree = degree + 1)

# Definition of the no-slip condition
zero_velocity = Constant((0.0, 0.0))

# Definition of the Dirichlet BCs
bc = [
    DirichletBC(X.sub(0), zero_velocity, zero_boundary),
    DirichletBC(X.sub(0), zero_velocity, vertical_wall),
    DirichletBC(X.sub(0), zero_velocity, horizontal_wall),
    DirichletBC(X.sub(0), velocity_profile, inflow),
    DirichletBC(X.sub(0).sub(1), Constant(0.0), outflow)
]

# Trial and test functions
x = Function(X)
u, p = split(x)

v, q = TestFunctions(X)

# Definition of the weak (variational) form
F = inner(grad(u)*u, v)*dx + nu * inner(grad(u), grad(v))*dx \
     - inner(p, div(v))*dx + inner(q, div(u))*dx + inner(f, v)*dx

# Computation of the solution (FEniCS solver)
solve(F == 0, x, bc)
