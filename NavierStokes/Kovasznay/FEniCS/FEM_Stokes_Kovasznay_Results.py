figu = plot(u[0], levels = 40, cmap = 'jet')
plt.colorbar(figu)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('$u_{h}$')
#plt.savefig('FEMkovasznayu.pdf')

figv = plot(u[1], levels = 40, cmap = 'jet')
plt.colorbar(figv)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('$v_{h}$')
#plt.savefig('FEMkovasznayv.pdf')

figp = plot(p, levels = 40, cmap = 'jet')
plt.colorbar(figp)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('$p_{h}$')
plt.savefig('spuriouspressure.pdf')

u_array = []
v_array = []
p_array = []

mesh_points = mesh.coordinates()

x_coord = np.array(mesh_points[:,0])
y_coord = np.array(mesh_points[:,1])

for i,j in mesh_points:
  temp = x(Point(i,j))
  u_array.append(temp[0])
  v_array.append(temp[1])
  p_array.append(temp[2])

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(x_coord, y_coord, np.array(v_array), cmap = 'jet')

u_ex_array = 1 - np.exp(lamb*x_coord) * np.cos(2*np.pi*y_coord)
v_ex_array = (lamb/(2*np.pi)) * np.exp(lamb*x_coord) * np.sin(2*np.pi*y_coord)
p_ex_array = 0.5 * (1 - np.exp(2*lamb*x_coord))

diff_u = np.abs(np.array(u_array).flatten() - np.array(u_ex_array).flatten())
diff_v = np.abs(np.array(v_array).flatten() - np.array(v_ex_array).flatten())
diff_p = np.abs(np.array(p_array).flatten() - np.array(p_ex_array).flatten())

import matplotlib.ticker as ticker

def fmt(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

fig = plt.figure()
ax = fig.add_subplot(111)
cont = plt.tricontourf(x_coord, y_coord, diff_u, levels = 40, cmap = 'inferno')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.colorbar(cont, format=ticker.FuncFormatter(fmt))
tit = '$|u_{h} - u_{ex}|$'
plt.title(tit)
ax.set_aspect('equal')
plt.savefig('FEMkovasznayerru.pdf')

fig = plt.figure()
ax = fig.add_subplot(111)
cont = plt.tricontourf(x_coord, y_coord, diff_v, levels = 40, cmap = 'inferno')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.colorbar(cont, format=ticker.FuncFormatter(fmt))
tit = '$|v_{h} - v_{ex}|$'
plt.title(tit)
ax.set_aspect('equal')
plt.savefig('FEMkovasznayerrv.pdf')

fig = plt.figure()
ax = fig.add_subplot(111)

cont = plt.tricontourf(x_coord, y_coord, diff_p, levels = 40, cmap = 'inferno')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.colorbar(cont, format=ticker.FuncFormatter(fmt))
tit = '$|p_{h} - p_{ex}|$'
plt.title(tit)
ax.set_aspect('equal')
plt.savefig('FEMkovasznayerrp.pdf')

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(x_coord, y_coord, diff_p, cmap = 'inferno')
niceMathTextForm = ticker.ScalarFormatter(useMathText=True)
niceMathTextForm.set_scientific(True)
niceMathTextForm.set_powerlimits((-2,2))
ax.w_zaxis.set_major_formatter(niceMathTextForm)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$|u_{h} - u_{ex}|$')

with open('kovasznayflow.txt', "w") as txt_file:
  for line in range(len(x_coord)):
    txt_file.write('{}, {}, {}, {}, {};'.format(x_coord[line], 
                                                y_coord[line], 
                                                u_array[line],
                                                v_array[line],
                                                p_array[line]) + "\n")
