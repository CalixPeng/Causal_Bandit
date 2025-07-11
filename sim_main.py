import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from multiprocessing import Pool
from functools import partial
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from utils import System
from algos import vanila_UCB, golem_MAB, LinSEM_TS, CSL_UCB
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

N_criterion, N_alg = 2, 4
N_case, N_worker = 100, 8

def one_case(T_list, S):
    T_c, S_list = [0], [S]
    R_1, F_1 = vanila_UCB(T_list[-1], T_c, S_list)
    R_2, F_2 = golem_MAB(T_list[-1], T_c, S_list)
    R_3, F_3 = LinSEM_TS(T_list[-1], T_c, S_list)
    R_4, F_4 = CSL_UCB(T_list[-1], T_c, S_list, n_max=100)
    Reward = np.vstack((R_1, R_2, R_3, R_4))
    Regt = max(S.Reward_exp)*np.ones((4, T_list[-1])) - Reward
    Regt_acc = np.zeros((N_alg, T_list.size))
    for i_t, t in enumerate(T_list):
        Regt_acc[:,i_t] = np.sum(Regt[:,:t], axis=1)
    Regt_acc = np.append(np.zeros((N_alg, 1)), Regt_acc, axis=1)
    Freq = np.vstack((F_1, F_2, F_3, F_4))
    Freq_avg = np.zeros((N_alg,T_list.size))
    dt = T_list[1] - T_list[0]
    for i_t, t in enumerate(T_list):
        Freq_avg[:,i_t] = np.mean(Freq[:,t-dt:t], axis=1)
    Freq_avg = np.append(np.zeros((N_alg,1)), Freq_avg, axis=1)
    return Regt_acc, Freq_avg

if __name__ == '__main__':
    np.random.seed(2025)
    N, sigma = 10, 1
    Nu, Sigma = np.ones(N), sigma*np.eye(N)
    T_list = np.arange(0, 1501, 100)
    func = partial(one_case, T_list[1:])
    Regt = np.zeros((N_alg, T_list.size, N_case))
    Freq = np.zeros((N_alg, T_list.size, N_case))
    S_list = [System(N, Nu, Sigma) for _ in range(N_case)]
    print('Start at ' + datetime.now().strftime('%H:%M:%S'))
    p = Pool(N_worker)
    n = 0
    for i_case, Result in enumerate(p.imap(func, S_list)):
        Regt[:,:,i_case] = Result[0]
        Freq[:,:,i_case] = Result[1]
        n += 1
        print(f'\r{n}/{N_case}, ' + datetime.now().strftime('%H:%M:%S'), 
              end='', flush=True)
    p.close()
    Regt_avg = np.mean(Regt, axis=2)
    Regt_low = np.quantile(Regt, 0.025, axis=2)
    Regt_up = np.quantile(Regt, 0.975, axis=2)
    Freq_avg = np.mean(Freq, axis=2)
    Freq_low = np.quantile(Freq, 0.025, axis=2)
    Freq_up = np.quantile(Freq, 0.975, axis=2)
    
    C_list = ['#377eb8', '#4daf4a', '#984ea3', '#e41a1c', '#ff7f00', 
              '#ffff33', '#a65628', '#f781bf']
    M_list = ['o', '^', 's', 'd']
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams.update({'font.size': 12})
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
    plt.figure(1, dpi=800)
    plt.plot(T_list, Regt_avg[0,:], C_list[0], marker=M_list[0], label='Vanilla UCB')
    plt.plot(T_list, Regt_avg[1,:], C_list[1], marker=M_list[1], label='GOLEM-MAB')
    plt.plot(T_list, Regt_avg[2,:], C_list[2], marker=M_list[2], label='LinSEM-TS')
    plt.plot(T_list, Regt_avg[3,:], C_list[3], marker=M_list[3], label='CSL-UCB')
    plt.xticks(np.arange(0, 1501, 250))
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.xlabel('Time Steps')
    plt.ylabel('Cumulative Regret')
    plt.legend(labelspacing=0.2, loc='upper left', bbox_to_anchor=(0, 1))
    plt.grid()
    # plt.savefig('cum_regret.eps')

    plt.figure(2, dpi=800)
    plt.plot(T_list, Freq_avg[0,:], C_list[0], marker=M_list[0], label='Vanilla UCB')
    plt.plot(T_list, Freq_avg[1,:], C_list[1], marker=M_list[1], label='GOLEM-MAB')
    plt.plot(T_list, Freq_avg[2,:], C_list[2], marker=M_list[2], label='LinSEM-TS')
    plt.plot(T_list, Freq_avg[3,:], C_list[3], marker=M_list[3], label='CSL-UCB')
    plt.xticks(np.arange(0, 1501, 250))
    plt.ylim(-0.05, 1.05)
    plt.xlabel('Time Steps')
    plt.ylabel('Optimal Intervention Percentage')
    plt.legend(labelspacing=0.2, loc='lower left', bbox_to_anchor=(0.2, 0.15))
    plt.grid()
    # plt.savefig('opt_percent.eps')
    plt.show()
    