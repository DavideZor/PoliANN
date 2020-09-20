nu = 0.1

deg = 2
N = 100
dt = 0.001
T = 2
theta = 0.5

u, t, mesh = FEM_Burgers(deg, N, T, dt, nu, theta, True)
plt.savefig('FEMsolutionNOGRID.pdf')

plot(mesh)
plt.title('1D Mesh')
plt.savefig('1DMesh.pdf')
