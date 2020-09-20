import numpy as np
import deepxde as dde

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from deepxde.backend import tf

# Definition of the problem parameters
a = 0		# First point of the domain
b = 1		# Last point of the domain

T = 1.0		# Time interval length
nu = 0.0001	# Kinematic viscosity

# Definition of the ANN parameters
Nd = 4000       # Number of collocation points in the
                # generalized domain
Nb = 200        # Number of collocation points used to
                # enforce the boundary conditions
Ni = 250        # Number of points used to enforce the
                # initial condition
Nh = 3          # Number of hidden layers
Nl = 40         # Number of nodes per layer
sigma = 'tanh'  # Activation function

# Definition of the problem geometry
geom = dde.geometry.Interval(a, b)

# Definition of the PDE to be solved using
# the TensorFlow 2.0 syntax
def pde(x, y):
    
    dy_x = tf.gradients(y, x)[0]
    dy_x, dy_t = dy_x[:, 0:1], dy_x[:, 1:2]
    dy_xx = tf.gradients(dy_x, x)[0][:, 0:1]
    return dy_t + y * dy_x - nu * dy_xx
        
        def left_boundary(x, on_boundary):
            return on_boundary and np.isclose(x[0], a)

        def right_boundary(x, on_boundary):
            return on_boundary and np.isclose(x[0], b)

        def on_initial(_, on_initial):
            return on_initial
        
        
        def g_left(x):
            return np.zeros((len(x), 1))
        
        def g_right(x):
            return np.zeros((len(x), 1))
        
        def u_init(x):
            return np.sin(np.pi*x[:, 0:1])
        
# Time axis definition
timedomain = dde.geometry.TimeDomain(0, T)

# Generalized domain construction    
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Definition of the Dirichlet BCs
bc_l = dde.DirichletBC(geomtime, g_left, left_boundary)
bc_r = dde.DirichletBC(geomtime, g_right, right_boundary)

# Definition of the Initial Condition
ic = dde.IC(geomtime, u_init, on_initial)

# Definition of the generalized BCs
bc = [bc_l, bc_r, ic]

# Definition of the collocation points
data = dde.data.TimePDE(geomtime, pde, bc, num_domain=Nd,
                        num_boundary=Nb, num_initial=Nt)

# Definition of the FFNN architecture 
layers = [2] + [Nl] * Nh + [1]

# Definition of the weights initializer
initializer = 'Glorot uniform'

# Definition of the ANN learning algorithm
net = dde.maps.FNN(layers, sigma, initializer)

# Building of the complete model
model = dde.Model(data, net)

# Model training with the L-BFGS-B algorithm
model.compile('L-BFGS-B')
losshistory, train_state = model.train()
