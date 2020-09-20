times = []
points = []
h = []
s = ''

plots = 0
save_txt = 0
save_plots = 0

degree_list = [1, 2, 3, 4]

res_list = [10, 20, 30, 40, 50, 60, 80, 100, 150, 200]

timeplot = plt.figure(1)
errorL2plot = plt.figure(2)
errorH1plot = plt.figure(3)

for i in degree_list:

  t_plot = []
  n_plot = []
  h_plot = []
  vec_errorL2 = []
  vec_errorH1 = []

  s = s + '\\begin{table}[H] \n'
  s = s + '\\centering \n'
  s = s + '\\begin{tabular}{c|ccc} \n'
  s = s + '${}$ & $\\lVert u_{{h}} - u_{{ex}} \\rVert_{{L^{{2}}}} $ & $\\lVert u_{{h}} - u_{{ex}} \\rVert_{{H^{{1}}}}$ \\\\'.format(i) + 'Execution Time' + '\n'
  s = s + '\\hline \n'


  for j in res_list:

    u_ex = Expression('sin(pi*x[0])*sin(pi*x[1])', degree = i + 1)
    f = Expression('2*pi*pi*sin(pi*x[0])*sin(pi*x[1])', degree = i + 1)

    start_time = time.time()
    u, mesh = PoissonSolver(j, i, f)
    h_max = mesh.hmax()
    h_plot.append(h_max)
    h.append(h_max)
    Delta = time.time() - start_time 
    t_plot.append(Delta)
    times.append(Delta)
    Num_Points = mesh.num_vertices()
    n_plot.append(Num_Points)
    points.append(Num_Points)

    errL2 = errornorm(u_ex, u, 'L2')
    vec_errorL2.append(errL2)
    errH1 = errornorm(u_ex, u, 'H1')
    vec_errorH1.append(errH1)

    s = s + '${:.4f}$ & ${:.2e}$ & ${:.2e}$ & ${:.2f}$ \\\\'.format(h_max, errL2, errH1, Delta) + '\n'

    print('h = {:.4f} degree = {} eL2 = {:.2e} eH1 = {:.2e}'.format(h_max, i, errL2, errH1))
    
    tit = 'FEniCS solution for N = {}'.format(Num_Points)

  s = s + '\\end{tabular} \n'
  s = s + '\\caption{{FEniCS solution error for the Poisson equation with r = {} }}'.format(int(i)) + '\n'
  s = s + '\\label{{tab:PoissonSquare{}}}'.format(int(i)) + '\n'
  s = s + '\\end{table}' +  '\n'
  s = s + '\n'

  plt.figure(1)
  lbl = 'r = {}'.format(i)
  plt.scatter(n_plot, t_plot)
  plt.plot(n_plot, t_plot, label = lbl)
  plt.grid(linewidth=0.1)
  plt.xlabel('Number of vertices')
  plt.ylabel('Execution Time [s]')

  plt.figure(2)
  lbl = 'r = {}'.format(i)
  plt.scatter(res_list, vec_errorL2)
  plt.plot(res_list, vec_errorL2, label = lbl)
  plt.xscale("log")
  plt.yscale("log")
  plt.grid(linewidth=0.01)
  plt.xlabel('$h^{-1}$')
  plt.ylabel('$L^{2}$ error')

  plt.figure(3)
  lbl = 'r = {}'.format(i)
  plt.scatter(res_list, vec_errorH1)
  plt.plot(res_list, vec_errorH1, label = lbl)
  plt.grid(linewidth=0.01)
  plt.xscale("log")
  plt.yscale("log")
  plt.xlabel('$h^{-1}$')
  plt.ylabel('$H^{1}$ error')

plt.figure(1)
plt.legend(framealpha=1, frameon=True);
plt.savefig('PoissonSquareDegreevsTime.pdf')

h_vec = np.array(res_list, dtype=np.float64)

plt.figure(2)
plt.plot(h_vec, h_vec**(-1),'-.', label='$h^{-1}$')
plt.plot(h_vec, h_vec**(-2),'-.', label='$h^{-2}$')
plt.plot(h_vec, h_vec**(-3),'-.', label='$h^{-3}$')
plt.plot(h_vec, h_vec**(-4),'-.', label='$h^{-4}$')
plt.legend(framealpha=1, frameon=True);
plt.savefig('PoissonSquareDegreeErrorL2.pdf')

plt.figure(3)
plt.plot(h_vec, h_vec**(-1),'-.', label='$h^{-1}$')
plt.plot(h_vec, h_vec**(-2),'-.', label='$h^{-2}$')
plt.plot(h_vec, h_vec**(-3),'-.', label='$h^{-3}$')
plt.plot(h_vec, h_vec**(-4),'-.', label='$h^{-4}$')
plt.legend(framealpha=1, frameon=True);
plt.savefig('PoissonSquareDegreeErrorH1.pdf')

plt.figure(4)
fig = plot(u, cmap = 'jet', title = tit)
plt.grid(fig, linewidth = 0.01)
plt.colorbar(fig)
plt.xlabel('$x$')
plt.ylabel('$y$')
tit = 'FEniCS solution with r = {} and N = {}'.format(i, Num_Points)
plt.title(tit)
plt.savefig('PoissonSquareDegree4Solution.pdf')

times = np.array(times)
times = times.reshape((len(res_list), -1))

with open('FEniCSPoissonSquareErrorsLateX.txt', 'w') as f:
  f.write(s)
  
figura = plot(abs(u-u_ex), cmap = 'inferno')
plt.colorbar(figura)
plt.xlabel('$x$')
plt.ylabel('$y$')
tit = 'FEniCS solution absolute error'
plt.title(tit)
plt.savefig('PoissonSquareSolutionError.png')

tit_points = "FEniCSPoissonSquare{}".format(Num_Points)
tit_points = tit_points + ".txt"

mesh_points = mesh.coordinates()

x_coord = np.array(mesh_points[:,0])
y_coord = np.array(mesh_points[:,1])

u_array = np.array([u(Point(x,y)) for x,y in mesh_points])

with open(tit_points, "w") as txt_file:
  for line in range(len(x_coord)):
    txt_file.write('{}, {}, {};'.format(x_coord[line], y_coord[line], u_array[line]) + "\n")
