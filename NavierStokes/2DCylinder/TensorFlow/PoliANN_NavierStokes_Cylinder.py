from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function

import numpy as np
import deepxde as dde
from deepxde.backend import tf

# Definition of the FFNN parameters
Nd = 37300      # Number of sampling points
                # for the domain
Nb = 37300      # Number of sampling points
                # for the domain boundaries
Nt = 10         # Number of sampling points
                # for the initial condition

Nh = 5          # Number of hidden layers
Nl = 80         # Number of neurons per layer

# Definition of the activation function
activator = 'tanh'

# Definition of the weights initialization algorithm
initializer = 'Glorot normal'

# Definition of the problem parameters
L = 20          # Length of the tube
H = 5           # Height of the tube

x_c = 2.5       # Cylinder center x-coordinate
y_c = 2.5       # Cylinder center y-coordinate
center = (x_c, y_c)
radius = 0.5    # Cylinder radius

T = 10          # Time interval length
nu = 0.01       # Kinematic viscosity


# Definition of the Stokes equations
def pde(x, y):
    
    # Definition of the unknown functions
    u, v, p = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    
    # Definition of the derivatives
    du = tf.gradients(u, x)[0]
    dv = tf.gradients(v, x)[0]
    dp = tf.gradients(p, x)[0]
    
    dp_x, dp_y = dp[:, 0:1], dp[:, 1:2]
    du_x, du_y, du_t = du[:, 0:1], du[:, 1:2], du[:, 1:2]
    dv_x, dv_y, dv_t = dv[:, 0:1], dv[:, 1:2], dv[:, 2:3]
    
    du_xx = tf.gradients(du_x, x)[0][:, 0:1]
    du_yy = tf.gradients(du_y, x)[0][:, 1:2]

    dv_xx = tf.gradients(dv_x, x)[0][:, 0:1]
    dv_yy = tf.gradients(dv_y, x)[0][:, 1:2]
    
    # Definition of the equations
    continuity = du_x + dv_y
    x_momentum = du_t + u * du_x + v * du_y + \
        dp_x - nu * (du_xx + du_yy)
    y_momentum = dv_t + u * dv_x + v * dv_y + \
        dp_y - nu * (dv_xx + dv_yy)
    
    return [continuity, x_momentum, y_momentum]

# Definition of the domain boundaries
def zero_boundary(x, on_boundary):
  return on_boundary and \
      (np.isclose(x[1], 0) or np.isclose(x[1], H))

def inlet(x, on_boundary):
  return on_boundary and np.isclose(x[0], 0)

def outlet(x, on_boundary):
  return on_boundary and np.isclose(x[0], L)

tol = 1e-3

def cylinder(x, on_boundary):
  return on_boundary and ((x[0] - x_c)*(x[0] - x_c) + \
                          (x[1] - y_c)*(x[1] - y_c) <
                          radius * radius + tol)
# Definition of the velocity profiles
def zero_velocity(x):
    return np.zeros((x.shape[0], 1))

def inlet_velocity(x):
    return ((4/(2 * H - 1))* x[:, 1] \
            * (H - x[:, 1])).reshape(-1, 1)

def zero_pressure(x):
    return np.zeros((x.shape[0], 1))

def on_initial(_, on_initial):
    return on_initial


# Definition of the geometry
geom = dde.geometry.Rectangle([0.0, 0.0], [L, H]) - \
    dde.geometry.Disk(center, radius)
    
# Definition of the time axis
timedomain = dde.geometry.TimeDomain(0, T)

# Definition of the generalized domain
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Definition of the Dirichlet BCs
bc_u_no_slip = dde.DirichletBC(geomtime, zero_velocity, 
                               zero_boundary, component = 0)
bc_v_no_slip = dde.DirichletBC(geomtime, zero_velocity, 
                               zero_boundary, component = 1)
bc_u_cylinder = dde.DirichletBC(geomtime, zero_velocity, 
                               cylinder, component = 0)
bc_v_cylinder = dde.DirichletBC(geomtime, zero_velocity, 
                               cylinder, component = 1)
bc_u_inlet = dde.DirichletBC(geomtime, inlet_velocity, 
                             inlet, component = 0)
bc_v_inlet = dde.DirichletBC(geomtime, inlet_velocity, 
                             inlet, component = 1)
bc_p = dde.DirichletBC(geomtime, zero_pressure, 
                       outlet, component = 2)

ic = dde.IC(geomtime, zero_velocity, on_initial)

bc = [bc_u_no_slip, bc_v_no_slip, 
      bc_u_cylinder, bc_v_cylinder,
      bc_u_inlet, bc_v_inlet,
      bc_p, ic]

# Domain sampling and data generation
data = dde.data.TimePDE(geomtime, pde, bc, 
                        num_domain = Nd, 
                        num_boundary = Nb,
                        num_initial = Nt)

# Definition of the FFNN architecture
layers = [3] + [Nl] * Nh + [3]

# FFNN building
net = dde.maps.FNN(layers, activator, initializer)

# Model (Data + FFNN) building
model = dde.Model(data, net)

# FFNN training with both Adam and L-BFGS
model.compile('adam', lr=0.001)
model.train(epochs = 5000)
model.compile('L-BFGS-B')
losshistory, train_state = model.train()

# Storage of the solution
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
