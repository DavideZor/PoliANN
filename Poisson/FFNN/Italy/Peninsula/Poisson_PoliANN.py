from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import deepxde as dde
from deepxde.backend import tf

# Nd is the number of sampling points in the domain
# Nb is the number of sampling points on the domain boundary
# Nh is the number of hidden layers
# Nl is the number of nodes for each layer
# sigma is the activation function ('tanh', 'sigmoid', 'relu')
# opt is the optimization algorithm ('adam', 'L-BFGS-B', 'mixed')
# initializer is the weights initialization algorithm 
#               ('Glorot uniform', 'Glorot normal')


def Poisson_PoliANN(Nd, Nb, Nh, Nl, sigma, opt, initializer):

    # PDE Definition using the TensorFlow notation
    def pde(x, y):
        
        f = 1.0
        
        # Definition of the spatial derivatives
        dy_x = tf.gradients(y, x)[0]
        dy_x, dy_y = dy_x[:, 0:1], dy_x[:, 1:]
        dy_xx = tf.gradients(dy_x, x)[0][:, 0:1]
        dy_yy = tf.gradients(dy_y, x)[0][:, 1:]
        
        # Definition of the Poisson equation
        return dy_xx + dy_yy + f
    
    # Definition of the boundary
    def boundary(_, on_boundary):
        return on_boundary
    
    # Definition of the homogeneous Dirichlet 
    # boundary conditions
    def func(x):
        return np.zeros([len(x), 1])
    
    # Geometry definition
    # The Mercator projection coordinates 
    # are loaded for the Italian peninsula
    file_title = "Italy_Peninsula_XY.txt"
    
    with open(file_title) as f:
        x = []
        for line in f:
            if not line.strip(',') or line.startswith('#'):
                continue
            currentline = line.split(',')
            x.append(float(currentline[0]))
            x.append(float(currentline[1]))
            
    # The coordinates are scaled to make the 
    # typical length of order O(1)       
    xy_italy = 200 * np.array(x)
    
    # The coordinates are sorted in a [x, y] array
    xy_italy = xy_italy.reshape(int(len(xy_italy)/2.0),2)
    
    # Geometry definition
    geom = dde.geometry.Polygon(xy_italy)
    
    # Imposition of the Dirichlet boundary condition
    bc = dde.DirichletBC(geom, func, boundary)
    
    # ANN model definition
    data = dde.data.PDE(geom, pde, bc, num_domain = Nd, 
                        num_boundary = Nb)
    net = dde.maps.FNN([2] + [Nl] * Nh + [1], 
                       sigma, initializer)
    
    model = dde.Model(data, net)
    
    
    # ANN model training
    if opt == 'adam':
        model.compile(opt, lr=0.001)
        losshistory, train_state = model.train(epochs = 10000)
    elif opt == 'L-BFGS-B':
        model.compile(opt)
        losshistory, train_state = model.train()
    elif opt == 'mixed':
        model.compile('adam', lr=0.001)
        model.train(epochs = 5000)
        model.compile('L-BFGS-B')
        losshistory, train_state = model.train()
        dde.saveplot(losshistory, train_state, 
                     issave=True, isplot=True)
    
    return model
