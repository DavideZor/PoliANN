from fenics import *
from mshr import *
import matplotlib.pyplot as plt

# Definition of the domain geometry
# Definition of the discretization parameters
N_circle = 50
N = 100
theta = 0.5
dt = 0.2

# Definition of the 2D cylinder
x_c = 2.5
y_c = 2.5
center = Point(x_c, y_c)
radius = 0.5

# Definition of the rectangular tube
L = 20
H = 5

geometry = Rectangle(Point(0.0, 0.0), Point(L, H)) \
          - Circle(center, radius, N_circle)

# Domain meshing
mesh = generate_mesh(geometry, N)

# Problem definition
nu = Constant(0.01)
T = 10

# Definition of the domain boundaries
def zero_boundary(x, on_boundary):
  return on_boundary and (near(x[1], 0) or near(x[1], H))

def inlet(x, on_boundary):
  return on_boundary and near(x[0], 0)

def outlet(x, on_boundary):
  return on_boundary and near(x[0], L)

tol = 1e-3

def cylinder(x, on_boundary):
  return on_boundary and ((x[0] - x_c)*(x[0] - x_c) + \
                          (x[1] - y_c)*(x[1] - y_c) <
                          radius * radius + tol)

# Definition of the function spaces for the velocity (V)
# and for the pressure (Q). In this case piecewise
# linear polynomials are used for the pressure and
# quadratic polynomials are used for the velocity
# (Taylor - Hood elements)
degree = 1

V = VectorElement('CG', mesh.ufl_cell(), 2)
Q = FiniteElement('CG', mesh.ufl_cell(), 1)

# A new mixed variable x = [u, p] is defined. This 
# variable belongs to the space X = [V, Q]
X = FunctionSpace(mesh, MixedElement([V, Q]))

# Definition of the velocity parabolic
# profile at the inlet
u_inlet = Expression(("(4/(2*up - 1))*U*x[1]*(up - x[1])", "0.0"),
                  degree = degree + 1, U = 1, up = H)

# Definition of the no-slip condition
zero_velocity = Constant((0.0, 0.0))

# Definition of the Dirichlet BCs
bc = [
      DirichletBC(X.sub(0), zero_velocity, zero_boundary),
      DirichletBC(X.sub(0), zero_velocity, cylinder),
      DirichletBC(X.sub(0), u_inlet, inlet)
]

# Definition of the variables
x = Function(X)
u, p = split(x)

x_old = Function(X)
u_old, p_old = split(x_old)

v, q = TestFunctions(X)

# Definition of the weak (variational) form
F = ( Constant(1/dt)*dot(u - u_old, v)
      + Constant(theta)*nu*inner(grad(u), grad(v))
      + Constant(theta)*dot(dot(grad(u), u), v)
      + Constant(1-theta)*nu*inner(grad(u), grad(v))
      + Constant(1-theta)*dot(dot(grad(u_old), u_old), v)
      - p*div(v)
      - q*div(u)
    )*dx

# Creation of the files in which the solution
# is going to be stored
velfile = File("u.pvd")
prefile = File("p.pvd")

u, p = x.split()

# Initilization
t = 0
x_old.assign(x)

# Time integration
while t < T:
    
    # Computation of the solution (FEniCS solver)
    solve(F == 0, x, bc)

    # Solution update
    t = t + dt
    x_old.assign(x)

    # Storage of the solution in a ParaView (.pvd) file
    velfile << (u, t)
    prefile << (p, t)
