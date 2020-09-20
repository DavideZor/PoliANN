tit = 'FEniCS solution for u with N = {}'.format(Td)

fig = plot(u[0], cmap = 'jet', levels = 40, title = '$u_{h}$')
plt.colorbar(fig)
plt.xlim((0.0, 8.0))
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.savefig('FEMstepflowu.pdf')

tit = 'FEniCS solution for v with N = {}'.format(Td)

fig = plot(u[1], cmap = 'jet', levels = 40, title = '$v_{h}$')
plt.xlim((0.0, 8.0))
plt.colorbar(fig)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.savefig('FEMstepflowv.pdf')

tit = 'FEniCS solution for p with N = {}'.format(Td)

fig = plot(p, cmap = 'jet', title = '$p_{h}$')
plt.xlim((0.0, 8.0))
plt.colorbar(fig)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.savefig('FEMstepflowp.pdf')

u_array = []
v_array = []
p_array = []

for i,j in mesh_points:
  temp = x(Point(i,j))
  u_array.append(temp[0])
  v_array.append(temp[1])
  p_array.append(temp[2])

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(x_coord, y_coord, np.array(v_array), cmap = 'jet')

p_array = np.array(p_array) - np.mean(p_array)

with open("FEniCSNSTube.txt", "w") as txt_file:
    for line in range(len(x_coord)):
      txt_file.write('{}, {}, {}, {}, {};'.format(x_coord[line], 
                                                  y_coord[line], 
                                                  u_array[line], 
                                                  v_array[line], 
                                                  p_array[line]) + "\n")
