import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.tri as tri

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

with open("FEniCSPoissonItaly.txt") as f:
    rawx = []
    rawy = []
    rawz = []
    rawxy = []
    for line in f:
        line = line[:-2]
        row = line.split(',')
        rawx.append(float(row[0]))
        rawy.append(float(row[1]))
        rawz.append(float(row[2]))
        rawxy.append([float(row[0]), float(row[1])])
        
x = np.array(rawx)
y = np.array(rawy)
u_FEM = np.array(rawz)

xy = np.array(rawxy) #Needed to evaluate the Neural Network in the same points


triang = tri.Triangulation(x, y)

def apply_mask(triang, alpha=0.4):
    # Mask triangles with sidelength bigger some alpha
    triangles = triang.triangles
    # Mask off unwanted triangles.
    xtri = x[triangles] - np.roll(x[triangles], 1, axis=1)
    ytri = y[triangles] - np.roll(y[triangles], 1, axis=1)
    maxi = np.max(np.sqrt(xtri**2 + ytri**2), axis=1)
    # apply masking
    triang.set_mask(maxi > alpha)

apply_mask(triang, alpha=0.05)


def fmt(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

#model = dde.model.restore("D:\Models\poissonstarStar", verbose = 1)

# u_ex = np.sin(np.pi * x) * np.sin(np.pi * y)
u_ex = 0.5 + np.exp(-((x - 4) * (x - 4) + (y - 17.2) * (y - 17.2)))

u_NN = model.predict(xy)
u_NN = u_NN.flatten()
diff_NN = abs(u_NN - u_ex)
diff_FEM = abs(u_FEM - u_ex)

fig1 = plt.figure()
ax1 = fig1.gca(projection='3d')
ax1.plot_trisurf(triang, u_NN, cmap = 'jet')
ax1.set_xlabel('$x$')
ax1.set_ylabel('$y$')
ax1.set_zlabel('$\\hat{u}$')
plt.savefig('FFNNpoissonItalysol3d.pdf')

fig1 = plt.figure()
ax1 = fig1.gca(projection='3d')
ax1.plot_trisurf(triang, diff_NN, cmap = 'inferno')
niceMathTextForm = ticker.ScalarFormatter(useMathText=True)
niceMathTextForm.set_scientific(True)
niceMathTextForm.set_powerlimits((-2,2))
ax1.w_zaxis.set_major_formatter(niceMathTextForm)
ax1.set_xlabel('$x$')
ax1.set_ylabel('$y$')
ax1.set_zlabel('$|\\hat{u} - u_{ex}|$')
plt.savefig('FFNNpoissonItalyabserr3d.pdf')

fig2 = plt.figure()
ax2 = fig2.gca(projection='3d')
ax2.plot_trisurf(triang, diff_FEM, cmap = 'inferno')
niceMathTextForm = ticker.ScalarFormatter(useMathText=True)
niceMathTextForm.set_scientific(True)
niceMathTextForm.set_powerlimits((-2,2))
ax2.w_zaxis.set_major_formatter(niceMathTextForm)
ax2.set_xlabel('$x$')
ax2.set_ylabel('$y$')
ax2.set_zlabel('$|u_{h} - u_{ex}|$')
plt.savefig('FEMpoissonItalyabserr3d.pdf')


figsol = plt.figure()
axsol = figsol.add_subplot(111)
contsol = axsol.tricontourf(triang, u_NN, 40, cmap="jet")
plt.colorbar(contsol)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('FFNN solution with $N_{\\theta} = 11220$')
axsol.set_aspect('equal')
plt.savefig('FFNNpoissonItalysol2d.pdf')
plt.show()

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
cont1 = ax1.tricontourf(triang, diff_NN, 40, cmap="inferno")
plt.colorbar(cont1, format=ticker.FuncFormatter(fmt))
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('FFNN absolute error with $N_{\\theta} = 11220$')
ax1.set_aspect('equal')
plt.savefig('FFNNpoissonItalyabserr2d.pdf')
plt.show()


fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
cont2 = ax2.tricontourf(triang, diff_FEM, 40, cmap="inferno")
plt.colorbar(cont2, format=ticker.FuncFormatter(fmt))
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('FEM absolute error with N = 10309')
ax2.set_aspect('equal')
plt.savefig('FEMpoissonItalyabserr2d.pdf')
plt.show()
