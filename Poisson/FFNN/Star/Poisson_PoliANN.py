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
        
        f = 4*(1 - (x[:,0:1]**2 + x[:, 1:2]**2))\
            *tf.exp(-(x[:, 0:1]**2 + x[:,1:2]**2))
            
        # Definition of the forcing term for the
        # re-scaled star-shaped domain
        # f = 2 * (np.pi**2) * tf.sin(np.pi * x[:, 0:1]) \
        #     * tf.sin(np.pi * x[:, 1:2])
      
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
    
    # Definition of the star-shaped domain
    star = ([0.0, 1.0], [0.3, 0.5], [1.0, 0.5], 
            [0.7, 0.0], [1.0, -0.5], [0.3, -0.5], 
            [0, -1], [-0.3, -0.5], [-1, -0.5], 
            [-0.7, 0.0], [-1.0, 0.5], [-0.3, 0.5])
            
            
    # Definition of the re-scaled star-shaped domain
    # star = ([0.5, 1.0], [0.625, 0.75], [1.0, 0.75], 
    #         [0.85, 0.5], [1.0, 0.25], [0.625, 0.25], 
    #         [0.5, 0], [0.35, 0.25], [0, 0.25], 
    #         [0.15, 0.5], [0.0, 0.75], [0.35, 0.75])
    
    geom = dde.geometry.Polygon(star)
    
    # Imposition of the Dirichlet boundary condition
    bc = dde.DirichletBC(geom, func, boundary)
    
    # ANN model definition
    data = dde.data.PDE(geom, pde, bc, num_domain = Nd, 
                        num_boundary = Nb)
    net = dde.maps.FNN([2] + [Nl] * Nh + [1], 
                       sigma, initializer)
                       
    # Strict enforcement of the BCs for the unit square domain
    # net.apply_output_transform(lambda x, y: \
    #                           x[:,0:1] * (1 - x[:, 0:1]) * \
    #                           x[:,1:2] * (1 - x[:, 1:2]) * y)
    
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
