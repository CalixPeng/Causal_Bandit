import numpy as np
from numpy.linalg import inv
from utils import constr_Ba, golem, det_change_all
from utils import learn_topo, est_wt, conf_bound, det_change_col

###############################################################################
#                               Vanila-UCB
###############################################################################
def vanila_UCB(T, T_c, S_list):
    static = (len(T_c) == 1)
    coef, w_size = 1, 10
    N_act = 2 ** S_list[0].N
    Id_act = -1 * np.ones(T, dtype=np.int32)
    V_act, N_visit = np.zeros(N_act), np.zeros(N_act)
    Reward, Freq_opt = np.zeros(T), np.zeros(T)
    for t in range(T):
        if t in T_c:
            S = S_list[T_c.index(t)]
        if t < N_act:
            id_act = t
        else:
            UCB = V_act + coef * np.sqrt(np.log(t)/N_visit)
            id_act = UCB.argmax()
        Id_act[t] = id_act
        N_visit[id_act] += 1
        X = S.step(id_act).squeeze()
        Reward[t] = X[-1]
        if S.is_opt(id_act):
            Freq_opt[t] = 1
        id_relate = np.where(Id_act==id_act)[0]
        if not static and id_relate.size > w_size:
            id_relate = id_relate[-w_size:]
        V_act[id_act] = np.mean(Reward[id_relate])
    return Reward, Freq_opt

###############################################################################
#                             GOLEM-MAB
###############################################################################
def golem_MAB(T, T_c, S_list):
    static = (len(T_c) == 1)
    N, N_act = S_list[0].N, 2 ** S_list[0].N
    Action = np.zeros((N,N_act), dtype=int)
    for a in range(N_act):
        Action[:,a] = [int(j) for j in bin(a)[2:].zfill(N)]
    N_learn = 300
    Reward, Freq_opt = np.zeros(T), np.zeros(T)
    X = np.zeros((N,T))
    t_i, Ba = 0, None
    for t in range(T):
        if t in T_c:
            S = S_list[T_c.index(t)]
        if not static and t >= t_i + 2 * N_learn + 20:
            id_relate = range(t_i + 2 * N_learn, t)
            if det_change_all(S, X[:,id_relate], Ba) == True:
                t_i = t
        if t < t_i + N_learn:
            id_act = 0
        elif t < t_i + 2 * N_learn:
            id_act = N_act - 1
        elif t == t_i + 2 * N_learn:
            id_relate_1 = range(t_i, t_i + N_learn) 
            id_relate_2 = range(t_i + N_learn, t_i + 2*N_learn)
            B_hat = golem(X[:,id_relate_1].T, lambda_1=2e-2, lambda_2=5.0)
            BI_hat = golem(X[:,id_relate_2].T, lambda_1=2e-2, lambda_2=5.0)
            Graph_mean = np.zeros(N_act)
            for a in range(N_act):
                Ba_hat = constr_Ba(Action[:,a], B_hat, BI_hat)
                Mu_a_hat = inv(np.eye(S.N)-Ba_hat.T) @ S.Nu
                Graph_mean[a] = Mu_a_hat[-1]
            id_act = Graph_mean.argmax()
            Ba = constr_Ba(Action[:,id_act], B_hat, BI_hat)
        X[:,t] = S.step(id_act).squeeze()
        Reward[t] = X[-1,t]
        if S.is_opt(id_act):
            Freq_opt[t] = 1
    return Reward, Freq_opt

###############################################################################
#                           LinSEM-TS
###############################################################################
def LinSEM_TS(T, T_c, S_list):
    var, N = 1, S_list[0].N
    N_act = 2 ** N
    Action = np.zeros((N,N_act), dtype=int)
    for a in range(N_act):
        Action[:,a] = [int(j) for j in bin(a)[2:].zfill(N)]
    Act, X = np.zeros((N,T), dtype=int), np.zeros((N,T))
    Reward, Freq_opt = np.zeros(T), np.zeros(T)
    for t in range(T):
        if t == 0:
            S = S_list[0]
            list_pa, list_pa_I = {}, {}
            for n in range(N):
                list_pa[n] = list(np.where(S.B[:,n]!=0)[0])
                list_pa_I[n] = list(np.where(S.BI[:,n]!=0)[0])
            B_hat, BI_hat = np.zeros((N,N)), np.zeros((N,N))
            B_tilde, BI_tilde = np.zeros((N,N)), np.zeros((N,N))
            Vobs, Vint, gobs, gint = {}, {}, {}, {}
            for n in range(N):
                Vobs[n] = np.eye(len(list_pa[n]))
                Vint[n] = np.eye(len(list_pa_I[n]))
                gobs[n] = np.zeros((len(list_pa[n]), 1))
                gint[n] = np.zeros((len(list_pa_I[n]), 1))
        elif t in T_c[1:]:
            S = S_list[T_c.index(t)]
        for n in range(N):
            if len(list_pa[n]) > 0:
                B_tilde[n][list_pa[n]] = np.random.multivariate_normal(
                    B_hat[n][list_pa[n]], var*inv(Vobs[n]))
            if len(list_pa_I[n]) > 0:
                BI_tilde[n][list_pa_I[n]] = np.random.multivariate_normal(
                    BI_hat[n][list_pa_I[n]], var*inv(Vint[n]))
        Graph_mean = np.zeros(N_act)
        for a in range(N_act):
            Ba_hat = constr_Ba(Action[:, a], B_tilde.T, BI_tilde.T)
            Mu_a_hat = inv(np.eye(N)-Ba_hat.T) @ S.Nu
            Graph_mean[a] = Mu_a_hat[-1]
        id_act = Graph_mean.argmax()
        Act[:,t] = Action[:,id_act]
        X[:,t] = S.step(id_act).squeeze()
        Reward[t] = X[-1,t]
        if S.is_opt(id_act):
            Freq_opt[t] = 1
        for n in range(N):
            Xt = X[:,t]
            Xt_pai = np.zeros((N,1))
            if len(list_pa_I[n]) > 0 and Act[n,t] == 1:
                Xt_pai = Xt[list_pa_I[n]][:,np.newaxis]
                Vint[n] += Xt_pai @ Xt_pai.T
                gint[n] += Xt_pai * (Xt[n] - S.Nu[n])
                BI_hat[n,list_pa_I[n]] = (inv(Vint[n]) @ gint[n])[:,0]
            elif len(list_pa[n]) > 0 and Act[n,t] == 0:
                Xt_pai = Xt[list_pa[n]][:,np.newaxis]
                Vobs[n] += Xt_pai @ Xt_pai.T
                gobs[n] += Xt_pai * (Xt[n] - S.Nu[n])
                B_hat[n,list_pa[n]] = (inv(Vobs[n]) @ gobs[n])[:,0]
    return Reward, Freq_opt

###############################################################################
#                                  CSL-UCB
###############################################################################
def CSL_UCB(T, T_c, S_list, n_max=np.inf):
    static = (len(T_c) == 1)
    N, N_act = S_list[0].N, 2 ** S_list[0].N
    coef_init, N_ES, T_static = 0, 20, 200
    F_graph, F_est = 50, 20
    Action = np.zeros((N,N_act), dtype=int)
    for a in range(N_act):
        Action[:,a] = [int(j) for j in bin(a)[2:].zfill(N)]
    Act, X = np.zeros((N,T), dtype=int), np.zeros((N,T))
    Reward, Freq_opt = np.zeros(T), np.zeros(T)
    B_hat, BI_hat = None, None
    Graph_mean, UpperB = np.zeros(N_act), np.zeros(N_act)
    T_i = np.zeros((2,N), dtype=int)
    N_sample, Learn = np.zeros((2,N), dtype=int), np.ones((2,N), dtype=bool)
    for t in range(T):
        if t in T_c:
            S = S_list[T_c.index(t)]
        if not static and t > T_static:
            Change = det_change_col(S, X[:,:t], Act[:,:t], B_hat, BI_hat)
            T_i[(Change==1) & (Learn==False)] = t - 1
            N_sample[(Change==1) & (Learn==False)] = 0
            Learn[Change==1] = True
        if np.min(N_sample) < N_ES:
            id_act = np.random.randint(0, N_act)
        else:
            if Learn.any() or t % F_graph == 0:
                Topo_B, Topo_BI = learn_topo(X[:,:t], Act[:,:t], T_i, S.Nu, n_max)
            if Learn.any() or t % F_est == 0:
                B_hat = est_wt(X[:,:t], Act[:,:t], T_i, S.Nu, Topo_B, 0, n_max)
                BI_hat = est_wt(X[:,:t], Act[:,:t], T_i, S.Nu, Topo_BI, 1, n_max)
                Learn[:] = False
                for a in range(N_act):
                    Ba_hat = constr_Ba(Action[:,a], B_hat, BI_hat)
                    Mu_a_hat = inv(np.eye(N)-Ba_hat.T) @ S.Nu
                    Graph_mean[a] = Mu_a_hat[-1]
                    UpperB[a] = conf_bound(X[:,:t], Act[:,:t], T_i, Ba_hat, 
                                           Action[:,a], S.Nu, S.Sigma, n_max)
                if coef_init == 0:
                    coef_init = 0.5 * max(Graph_mean)/max(UpperB)
                    coef = coef_init
                else:
                    t_i = np.max(T_i)
                    coef = coef_init * (1 + np.cos(np.pi*(t-t_i)/(T-t_i)))/2
                UCB = Graph_mean + coef * UpperB
                id_act = UCB.argmax()
        Act[:,t] = Action[:,id_act]
        X[:,t] = S.step(id_act).squeeze()
        Reward[t] = X[-1,t]
        for n, a in enumerate(Action[:,id_act]):
            N_sample[a,n] += 1
        if S.is_opt(id_act):
            Freq_opt[t] = 1
    return Reward, Freq_opt
