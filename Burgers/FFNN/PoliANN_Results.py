import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

xx = np.linspace(0, 1, 101)

N = len(xx)

t_list = np.arange(8)*0.25

nu = 0.01

for i in t_list:
    
    tt = i * np.ones(N)
    xxtt = np.column_stack((xx,tt))
    u = model.predict(xxtt)
    
    plt.figure(2)
    tit = 'FFNN with $\\nu = {:.4f}$'.format(nu)
    plt.plot(xx, u, label=r"$t = {:.2f}$".format(i))
    plt.grid(linewidth = 0.1)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.1)
    plt.xlabel('$x$')
    plt.ylabel('$u$')
    plt.title(tit)
    plt.legend(frameon=True)
    plt.savefig('NNSolutionBurgersNh3Nl6nu001HBC.png')
    
    
T = 0.4

tt = T * np.ones(N)

xxtt = np.column_stack((xx,tt))
u_NN_1 = model.predict(xxtt)
u_NN_1.flatten()
