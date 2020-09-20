from fenics import *
import matplotlib.pyplot as plt
import numpy as np

# Geometry definition
N = 40
mesh = UnitSquareMesh(N, N, 'crossed')

# Problem definition
nu = 0.025
lamb = 1/(2*nu) - np.sqrt((1/(4*(nu**2))) + 4*(np.pi**2))

# Definition of the function spaces for the velocity (V)
# and for the pressure (Q). In this case piecewise
# linear polynomials are used for the pressure and
# quadratic polynomials are used for the velocity
# (Taylor - Hood elements)
degree = 1

V = VectorElement('CG', mesh.ufl_cell(), degree+1)
Q = FiniteElement('CG', mesh.ufl_cell(), degree)

# A new mixed variable x = [u, p] is defined. This 
# variable belongs to the space X = [V, Q]
X = FunctionSpace(mesh, MixedElement([V, Q]))

# Kovasznay flow exact solution
u_ex = Expression((
    '1 - exp(lamb*x[0]) * cos(2*pi*x[1])',
    '(lamb/(2*pi)) * exp(lamb*x[0]) * sin(2*pi*x[1])'
    ), degree = degree + 1, lamb = lamb)

p_ex = Expression(
    '0.5 * (1 - exp(2*lamb*x[0]))',
    degree = degree, lamb = lamb)

f = Constant((0.0, 0.0))

# Defintion of the domain boundaries
def boundary(x, on_boundary):
  return on_boundary

# Definition of the Dirichlet BCs
bc = [
    DirichletBC(X.sub(0), u_ex, boundary),
    DirichletBC(X.sub(1), p_ex, boundary)
]

# Definition of the variables
x = Function(X)

u, p = split(x)
v, q = TestFunctions(X)

# Definition of the weak (variational) formulation
F = (nu * inner(grad(u), grad(v)) \
     - p * div(v) - div(u) * q \
     + dot(dot(grad(u), u), v) - dot(f, v)) * dx

# Computation of the solution (FEniCS solver)
solve(F == 0, x, bc)
