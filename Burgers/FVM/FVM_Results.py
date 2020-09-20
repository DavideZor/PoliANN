nu = 0.1

N = 100
dt = 0.001
T = 2

u, t, x = FVM_Burgers(0, 1, N, T, dt, nu, True)
plt.savefig('FVMSolution.pdf')

# Necessary time step size for stability
dt = 0.0001

# Number of intervals
N_list = [20, 40, 50, 80, 100, 200, 500]

# Table list of points

N_tab = (1 + np.arange(9))/10

line = ''

for i in N_list:
  u, t, x = FVM_Burgers(0, 1, i, 0.1, dt, nu, False)

  uu = u[2:2+i+1]

  xx = x[2:2+i+1]

  file_tit = 'FVMN{}'.format(i) + 't{}'.format(t) + '.txt'

  for j in range(len(xx)):
    if any(np.isclose(xx[j], N_tab, atol = 1e-5)):
      line = line + '{:.2f}, {:.6f} '.format(xx[j], uu[j]) + '\n'

  with open(file_tit, 'w') as new_file:
    new_file.write(line)

  line = ''
