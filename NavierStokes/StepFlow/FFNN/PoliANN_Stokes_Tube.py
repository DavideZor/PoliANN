import deepxde as dde
from deepxde.backend import tf
import numpy as np

# Definition of the FFNN parameters
Nd = 3000       # Number of sampling points
                # for the domain
Nb = 3000       # Number of sampling points
                # for the domain boundaries

Nh = 4          # Number of hidden layers
Nl = 60         # Number of neurons per layer

# Definition of the activation function
sigma = 'tanh'

# Definition of the weights initialization algorithm
initializer = 'Glorot normal'

# Definition of the geometry parameters
L = 15.0    # Domain length
H = 2.0     # Domain height

A = 2.0     # Distance of the vertical plane 
            # from the y axis (the inflow)
B = 1.0     # Distance of the horizontal plane 
            # from the x axis (the floor)

# Definition of the problem parameters
nu = 0.01

# Definition of the Stokes equations
def pde(x, y):

    #
    u, v, p = y[:, 0:1], y[:, 1:2], y[:, 2:3]

    du = tf.gradients(u, x)[0]
    dv = tf.gradients(v, x)[0]
    dp = tf.gradients(p, x)[0]

    p_x, p_y = dp[:, 0:1], dp[:, 1:2]
    u_x, u_y = du[:, 0:1], du[:, 1:2]
    v_x, v_y = dv[:, 0:1], dv[:, 1:2]

    u_xx = tf.gradients(u_x, x)[0][:, 0:1]
    u_yy = tf.gradients(u_y, x)[0][:, 1:2]

    v_xx = tf.gradients(v_x, x)[0][:, 0:1]
    v_yy = tf.gradients(v_y, x)[0][:, 1:2]

    continuity = u_x + v_y
    x_momentum = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    y_momentum = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)

    return [continuity, x_momentum, y_momentum]

# Definition of the boundaries
def zero_boundary(x, on_boundary):
  return on_boundary and np.isclose(x[1], 0) or np.isclose(x[1], H)

def inflow(x, on_boundary):
  return on_boundary and np.isclose(x[0], 0)

def outflow(x, on_boundary):
  return on_boundary and np.isclose(x[0], L)

def vertical_wall(x, on_boundary):
  return on_boundary and np.isclose(x[0], A) and (x[1] < B)

def horizontal_wall(x, on_boundary):
  return on_boundary and np.isclose(x[1], B) and (x[0] < A)

# Definition of the velocity profiles
def inlet_velocity(x):
    return (4/((H - B)**2))*(x[:, 1] - B) \
            * (H - x[:, 1])).reshape(-1, 1)

def zero_velocity(x):
    return np.zeros((x.shape[0], 1))

def zero_pressure(x):
    return np.zeros((x.shape[0], 1))

# Defintion of the geometry
geom = dde.geometry.Polygon([
    [0.0, H], [L, H], [L, 0.0], [A, 0.0], [A, B],
    [0.0, B]
])

# Definition of the Dirichlet BCs
inlet_u = dde.DirichletBC(geom, inlet_velocity, 
                          inflow, component = 0)
inlet_v = dde.DirichletBC(geom, zero_velocity, 
                          inflow, component = 1)
zero_u = dde.DirichletBC(geom, zero_velocity, 
                         zero_boundary, component = 0)
zero_v = dde.DirichletBC(geom, zero_velocity, 
                         zero_boundary, component = 1)
hor_u = dde.DirichletBC(geom, zero_velocity, 
                        horizontal_wall, component = 0)
hor_v = dde.DirichletBC(geom, zero_velocity, 
                        horizontal_wall, component = 1)
ver_u = dde.DirichletBC(geom, zero_velocity, 
                        vertical_wall, component = 0)
ver_v = dde.DirichletBC(geom, zero_velocity, 
                        vertical_wall, component = 1)
outflow_bc = dde.DirichletBC(geom, zero_velocity, 
                             outflow, component = 1)

bcs = [inlet_u, inlet_v, zero_u, zero_v, hor_u, hor_v,
       ver_u, ver_v, outflow_bc]

# Domain sampling and data generation
data = dde.data.PDE(geom, pde, bcs, 
                    num_domain = Nd, num_boundary = Nb)

# Definition of the FFNN architecture
layers = [2] + [Nl] * Nh + [3]

# FFNN building
net = dde.maps.FNN(layers, sigma, initializer)

# Model (Data + FFNN) building
model = dde.Model(data, net)

# FFNN training with both Adam and L-BFGS
model.compile('adam', lr=0.001)
model.train(epochs = 50000)
model.compile('L-BFGS-B')
losshistory, train_state = model.train()

# Storage of the solution
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
