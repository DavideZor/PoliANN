from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function

import numpy as np
import deepxde as dde
from deepxde.backend import tf



# Definition of the FFNN parameters
Nd = 1000       # Number of sampling points
                # for the domain
Nb = 1000       # Number of sampling points
                # for the domain boundaries

Nh = 3          # Number of hidden layers
Nl = 40         # Number of neurons per layer

# Definition of the activation function
sigma = 'tanh'

# Definition of the weights initialization algorithm
initializer = 'Glorot uniform'

# Definition of the problem parameters
fx = 0.0        # Force term x-component
fy = 0.0        # Force term y-component
nu = 0.025      # Kinematic viscosity

# Definition of the lambda parameter
lam = (1/(2*nu))-np.sqrt((1/(4*(nu**2)))+4*(np.pi**2))

# Definition of the Stokes equations
def pde(x, y):
    
    # Definition of the unknown functions
    u, v, p = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    
    # Definition of the derivatives
    du = tf.gradients(u, x)[0]
    dv = tf.gradients(v, x)[0]
    dp = tf.gradients(p, x)[0]
    
    dp_x, dp_y = dp[:, 0:1], dp[:, 1:2]
    du_x, du_y = du[:, 0:1], du[:, 1:2]
    dv_x, dv_y = dv[:, 0:1], dv[:, 1:2]
    
    du_xx = tf.gradients(du_x, x)[0][:, 0:1]
    du_yy = tf.gradients(du_y, x)[0][:, 1:2]

    dv_xx = tf.gradients(dv_x, x)[0][:, 0:1]
    dv_yy = tf.gradients(dv_y, x)[0][:, 1:2]
    
    # Definition of the equations
    continuity = du_x + dv_y
    x_momentum = u * du_x + v * du_y + dp_x \
                - nu * (du_xx + du_yy) - fx
    y_momentum = u * dv_x + v * dv_y + dp_y \
                - nu * (dv_xx + dv_yy) - fy
    
    return [continuity, x_momentum, y_momentum]

# Definition of the domain boundaries
def boundary(x, on_boundary):
    return on_boundary

def kovasznay_u(x):
    return 1 - np.exp(lam * x[:,0:1]) \
        * np.cos(2*np.pi*x[:,1:2])

def kovasznay_v(x):
    return (lam/(2*np.pi)) * np.exp(lam * x[:,0:1]) \
        * np.sin(2*np.pi*x[:,1:2])

def kovasznay_p(x):
    return 0.5*(1-np.exp(2*lam*x[:,0:1]))

# Definition of the geometry
geom = dde.geometry.Rectangle([0.0, 0.0], [1.0, 1.0])

# Definition of the Dirichlet BCs
bc_u = dde.DirichletBC(geom, kovasznay_u, boundary, component = 0)
bc_v = dde.DirichletBC(geom, kovasznay_v, boundary, component = 1)
bc_p = dde.DirichletBC(geom, kovasznay_p, boundary, component = 2)

bc = [bc_u, bc_v, bc_p]

# Domain sampling and data generation
data = dde.data.PDE(geom, pde, bc, 
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
