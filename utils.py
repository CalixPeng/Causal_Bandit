import numpy as np
import networkx as nx
from math import sqrt, log
from numpy.linalg import norm, inv, det, eigvalsh
from sklearn.linear_model import LinearRegression
from scipy.special import digamma
from utils_golem.golem_model import GolemModel
from utils_golem.golem_trainer import GolemTrainer
from utils_golem.train import postprocess

###############################################################################
#                                  general
###############################################################################
class System:
    def __init__(self, N, Nu, Sigma):
        self.N = N
        self.Nu = Nu
        self.Sigma = Sigma
        qualify = False
        while not qualify:
            Exist = np.random.randint(2, size=(N,N))
            Sign = 2*np.random.randint(2,size=(N,N)) - 1
            B = np.triu(np.random.uniform(0.5,2,size=(N,N)),k=1)*Exist*Sign
            Exist = np.random.randint(2, size=(N,N))
            Sign = 2*np.random.randint(2,size=(N,N)) - 1
            BI = np.triu(np.random.uniform(0.5,2,size=(N,N)),k=1)*Exist*Sign
            qualify = True
            for n in range(1, N):
                qualify = qualify and not np.array_equal(B[:,n], BI[:,n])
        self.B = B
        self.BI = BI
        self.Reward_exp = self.exp_reward()
    
    def exp_reward(self):
        N_act = 2 ** self.N
        Reward_exp = np.zeros(N_act)
        for i in range(N_act):
            act = np.array([int(j) for j in bin(i)[2:].zfill(self.N)])
            Ba = constr_Ba(act, self.B, self.BI)
            X = np.linalg.inv(np.eye(self.N)-Ba.T) @ self.Nu
            Reward_exp[i] = X[-1]
        return Reward_exp
    
    def is_opt(self, i_act):
        if abs(max(self.Reward_exp)-self.Reward_exp[i_act]) < 1e-2:
            return True
        else:
            return False
    
    def change_graph(self, n_col):
        reward_op = max(self.Reward_exp)
        I_op = np.where(abs(self.Reward_exp-reward_op)<1e-2)[0]
        Act_op = np.zeros((I_op.size, self.N))
        for i, i_op in enumerate(I_op):
            Act_op[i, :] = [int(j) for j in bin(i_op)[2:].zfill(self.N)]
        Col = np.where((Act_op.mean(axis=0)==0) | (Act_op.mean(axis=0)==1))[0]
        n_col = min(n_col, Col.size)
        B, BI = np.array(self.B), np.array(self.BI)
        while max(self.Reward_exp[I_op]) == max(self.Reward_exp):
            self.B, self.BI = np.array(B), np.array(BI)
            Col_change = np.random.choice(Col, n_col)
            for n in Col_change:
                Exist = np.zeros(self.N)
                Exist[:n] = np.random.randint(2, size=n)
                Sign = 2*np.random.randint(2, size=self.N) - 1
                W = np.random.uniform(0.5, 2, size=self.N) * Exist * Sign
                if Act_op[0, n] == 0:
                    self.B[:,n] = W
                else:
                    self.BI[:,n] = W
            self.Reward_exp = self.exp_reward()
        return
    
    def step(self, Id_act):
        if type(Id_act) not in [list, np.ndarray]:
            Id_act = [Id_act]
        T = len(Id_act)
        Eps = np.random.multivariate_normal(self.Nu, self.Sigma, T).T
        X = np.empty((self.N,T), dtype=float)
        for t in range(T):
            act = np.array([int(i) for i in bin(Id_act[t])[2:].zfill(self.N)])
            Ba = constr_Ba(act, self.B, self.BI)
            X[:,t] = np.linalg.inv(np.eye(self.N)-Ba.T) @ Eps[:,t]
        return X

def constr_Ba(act, B, BI):
    N = act.size
    Ba = np.zeros((N, N))
    for n in range(N):
        if act[n] == 0:
            Ba[:,n] = B[:,n]
        else:
            Ba[:,n] = BI[:,n]
    return Ba

###############################################################################
#                             for GOLEM-MAB
###############################################################################
def golem(X, lambda_1, lambda_2, equal_variances=True,
          num_iter=1e+5, learning_rate=1e-3, seed=1,
          checkpoint_iter=None, output_dir=None, B_init=None):
    X = X - X.mean(axis=0, keepdims=True)
    n, d = X.shape
    model = GolemModel(n, d, lambda_1, lambda_2, equal_variances, seed, B_init)
    trainer = GolemTrainer(learning_rate)
    B_est = trainer.train(model, X, num_iter, checkpoint_iter, output_dir)
    B_est = postprocess(B_est, graph_thres=0.3)
    return B_est

def det_change_all(S, X, Ba, thre=20):
    w_size = 20
    X = X[:,-w_size:]
    C = inv(np.eye(S.N)-Ba)
    Mu, Cov = C.T @ S.Nu, C.T @ S.Sigma @ C
    Mu_hat, Cov_hat = np.mean(X, axis=1), np.cov(X)
    num = X.shape[1]
    inv_C, inv_C_hat = inv(Cov), inv(Cov_hat)
    diff = num * log(det(Cov)/det(Cov_hat)) / 2
    for i in range(X.shape[1]):
        diff += ((X[:,i]-Mu).T @ inv_C @ (X[:,i]-Mu)) / 2
        diff -= ((X[:,i]-Mu_hat).T @ inv_C_hat @ (X[:,i]-Mu_hat)) / 2
    diff /= num * S.N
    return (diff > thre)

###############################################################################
#                               for CSL-UCB
###############################################################################
def est_MI(X, Y, k=5):
    N, num = len(X), 0
    for i in range(N):
        X_i, Y_i = np.delete(X, i), np.delete(Y, i)
        Dis_x, Dis_y = np.abs(X_i - X[i]), np.abs(Y_i - Y[i])
        Dis = np.maximum(Dis_x, Dis_y)
        Id_nb = np.argpartition(Dis, k)[1:k]
        eps_x, eps_y = np.max(Dis_x[Id_nb]), np.max(Dis_y[Id_nb])
        n_x, n_y = (Dis_x<=eps_x).sum(), (Dis_y<=eps_y).sum()
        num += digamma(n_x) + digamma(n_y)
    I_XY = digamma(k) - 1/k - num/N + digamma(N)
    return I_XY

def update_MI(X, Act, Nu, Ba_hat, MI_w, Adj, n, act, n_max, weight):
    Ba_hat[:,n] = 0
    learning_set = np.where(Adj[:,n]==1)[0]
    if learning_set.size == 0:
        return Ba_hat, MI_w
    id_relate = np.where(Act[n,:]==act)[0]
    if id_relate.size > n_max:
        id_relate = id_relate[:n_max]
    y = X[n,id_relate] - Nu[n]
    X_reg = X[learning_set,:]
    X_reg = X_reg[:,id_relate]
    reg = LinearRegression(fit_intercept=False).fit(X_reg.T, y)
    Ba_hat[learning_set,n] = reg.coef_
    Res = y - X_reg.T @ Ba_hat[learning_set,n]
    for p in learning_set:
        MI_w[p,n] = est_MI(Res,X[p,id_relate]) - weight*log(abs(Ba_hat[p,n]))
    return Ba_hat, MI_w

def learn_topo(X, Act, T_i, Nu, n_max, weight=1):
    N = X.shape[0]
    for a in [0, 1]:
        Ba_hat = np.zeros((N,N))
        MI_w = np.diag(-np.inf * np.ones(N))
        Adj = np.ones((N,N)) - np.eye(N)
        for n in range(N):
            t_i = T_i[a, n]
            Ba_hat, MI_w = update_MI(X[:,t_i:], Act[:,t_i:], Nu, Ba_hat, 
                                      MI_w, Adj, n, a, n_max, weight)
        while not nx.is_directed_acyclic_graph(nx.DiGraph(Adj)):
            index = np.unravel_index(MI_w.argmax(), MI_w.shape)
            Adj[index], MI_w[index] = 0, -np.inf
            n = index[1]
            t_i = T_i[a, n]
            Ba_hat, MI_w = update_MI(X[:,t_i:], Act[:,t_i:], Nu, Ba_hat, 
                                      MI_w, Adj, n, a, n_max, weight)
        if a == 0:
            Topo_B = Adj
        else:
            Topo_BI = Adj
    return Topo_B, Topo_BI

def est_wt(X, Act, T_i, Nu, Topo, i_act, n_max):
    N = X.shape[0]
    Ba_hat = np.zeros((N,N))
    for n in range(N):
        t_i = T_i[i_act, n]
        id_relate = t_i + np.where(Act[n,t_i:]==i_act)[0]
        if id_relate.size > n_max:
            id_relate = id_relate[:n_max]
        y = X[n,id_relate]-Nu[n]
        learning_set = np.where(Topo[:,n]==1)[0]
        if learning_set.size == 0:
            continue
        X_reg = X[learning_set,:]
        X_reg = X_reg[:,id_relate]
        reg = LinearRegression(fit_intercept=False).fit(X_reg.T, y)
        Ba_hat[learning_set,n] = reg.coef_
    return Ba_hat

def conf_bound(X, Act, T_i, Ba_hat, act, Nu, Sigma, n_max, prob_error=0.05):
    N = X.shape[0]
    C_hat = inv(np.eye(N)-Ba_hat)
    Mu_a = C_hat.T @ Nu
    weighted_eig_sum = 0
    for n in range(N):
        t_i = T_i[act[n], n]
        id_relate = t_i + np.where(Act[n,t_i:]==act[n])[0]
        if id_relate.size > n_max:
            id_relate = id_relate[:n_max]
        learning_set = np.where(Ba_hat[:,n]!=0)[0]
        if learning_set.size==0:
            continue
        X_reg = X[learning_set,:]
        X_reg = X_reg[:,id_relate]
        Cov = (Sigma[n,n]**2) * inv(X_reg@X_reg.T)
        Np = len(learning_set)
        weighted_eig_sum += sqrt(Np*(Np+2))*np.max(eigvalsh(Cov))
    sv_bound = sqrt(4*weighted_eig_sum*log(2*N/prob_error))
    return norm(C_hat[:,-1]) * sv_bound * norm(Mu_a)

def det_change_col(S, X, Act, B_hat, BI_hat, thre=2):
    w_size = 20
    Change = np.zeros((2, S.N))
    for a in [0, 1]:
        B = a * BI_hat + (1-a) * B_hat
        for n in range(S.N):
            sigma = S.Sigma[n,n]
            id_relate = np.where(Act[n,:]==a)[0]
            if id_relate.size > w_size:
                id_relate = id_relate[-w_size:]
            Eps_hat = (np.eye(S.N)-B)[:,n].T @ X[:,id_relate] - S.Nu[n]
            mu_hat, sigma_hat = np.mean(Eps_hat), np.std(Eps_hat)
            diff = log(sigma/sigma_hat) + (1/2)*np.mean((Eps_hat/sigma)**2) -\
                (1/2)*np.mean(((Eps_hat-mu_hat)/sigma_hat)**2)
            if diff > thre:
                Change[a, n] = 1
    return Change
