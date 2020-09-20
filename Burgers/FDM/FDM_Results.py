N = 100
T = 2
dt = 0.001
nu = 0.1
theta = 0.5

u, t, x = FDM_Burgers(0, 1, N, T, dt, nu, theta, True)
titolo = 'FDMwithN{}nu{}dt{}'.format(N, nu, dt) + '.png'
plt.savefig('FDMsolution.pdf')

import time

last_line = 'Execution time '
N_list = [20, 40, 50, 80, 100, 200, 500]
dt_list = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
t_vec = []

for j in dt_list:
  t = []
  for i in N_list:
    start_time = time.time()
    u, t, x = FDM_Burgers(0, 1, i, 2, j, nu, theta, False)
    tt = time.time()-start_time
    t_vec.append(tt)
    last_line = last_line + '& ${:.2f}$ '.format(tt)

  dt_vec = 1/np.array(dt_list)
  lbl = '$dt = {}$'.format(j)
  plt.figure(1)
  plt.scatter(N_list, t_vec)
  plt.plot(N_list, t_vec, label=lbl)
  plt.grid(linewidth = 0.1)
  plt.xlabel('$N$')
  plt.ylabel('Time [s]')
  plt.legend(frameon=True)

  t_vec = []

plt.savefig('with000001.pdf')

last_line = 'Execution time '
N_list = [20, 40, 50, 80, 100, 200]
dt_list = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
t_vec = []

for i in N_list:
  t = []
  for j in dt_list:
    start_time = time.time()
    u, t, x = FDM_Burgers(0, 1, i, 2, j, nu, theta, False)
    tt = time.time()-start_time
    t_vec.append(tt)
    last_line = last_line + '& ${:.2f}$ '.format(tt)

  dt_vec = 1/np.array(dt_list)
  lbl = '$N = {}$'.format(i)
  plt.figure(1)
  plt.scatter(dt_vec, t_vec)
  plt.plot(dt_vec, t_vec, label=lbl)
  plt.grid(linewidth = 0.1)
  plt.xlabel('$1/dt$')
  plt.ylabel('Time [s]')
  plt.legend(frameon=True)

  t_vec = []

plt.savefig('vs1dt.pdf')
