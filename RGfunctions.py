from Parameters import J, L, Sigma, Kappa
from Parameters import TolLie, TolMin, TolMax, Dist_Surf, precision, MaxIter, MaxLie, norm_choice
from Parameters import Radius, Modes_Perturb, Nh, Dist_Circle
from Parameters import K_indx, K_amp_inf, K_amp_sup, N_cs, TolCS, Save_Data
from Parameters import N, omega_0, eigenvalues, Omega, fixed_Omega, K, Number_of_Iterations
import numpy as xp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import scipy.signal as sps
from scipy.io import savemat
import time
import copy
import warnings
from datetime import date
import multiprocessing
from functools import partial
from tqdm import tqdm
warnings.filterwarnings("ignore")

num_cores = multiprocessing.cpu_count()

if precision == 32:
    precision_ = xp.float32
elif precision == 128:
    precision_ = xp.float128
else:
    precision_ = xp.float64

N = xp.array(N, dtype=int)
omega_0 = xp.array(omega_0, dtype=precision_)
Omega = xp.array(Omega, dtype=precision_)
eigenvalues = xp.array(eigenvalues, dtype=precision_)
es = xp.sign(eigenvalues).astype(int)
dim = len(omega_0)

zero_ = dim * (0,)
one_ = dim * (1,)
L_ = dim * (L,)
nL_ = dim * (-L,)
axisdim = tuple(range(1, dim+1))
reshape_J = (1,) + dim * (2*L+1,)
reshape_L = (J+1,) + dim * (1,)
JLconvdim = xp.index_exp[:J+1] + dim * xp.index_exp[L:3*L+1]

Index = xp.hstack((xp.arange(0, L+1), xp.arange(-L, 0)))
Index_ = dim * (Index,)
Nu = xp.meshgrid(*Index_, indexing='ij')
N_Nu = es[0] * xp.einsum('ij,j...->i...', N, Nu)
omega_0_Nu = xp.einsum('i,i...->...', omega_0, Nu).reshape(reshape_J)
mask = xp.prod(abs(N_Nu) <= L, axis=0, dtype=bool)
norm_Nu = xp.sqrt(xp.sum(xp.array(Nu)**2, axis=0)).reshape(reshape_J)
J_ = xp.arange(J+1, dtype=precision_).reshape(reshape_L)
CompIm = Sigma * xp.repeat(norm_Nu, J+1, axis=0) + Kappa * J_
omega_0_Nu_ = xp.repeat(xp.abs(omega_0_Nu), J+1, axis=0) / xp.sqrt((omega_0 ** 2).sum())
Iminus = omega_0_Nu_ > CompIm
Nu_mask = xp.index_exp[:J+1]
N_Nu_mask = xp.index_exp[:J+1]
for it in range(dim):
    Nu_mask += (Nu[it][mask],)
    N_Nu_mask += (N_Nu[it][mask],)

def plotf(fun):
    plt.rcParams.update({'font.size': 22})
    if dim == 2:
        fig, ax = plt.subplots(1,1)
        ax.set_xlim(-L, L)
        ax.set_ylim(-L, L)
        color_map = 'hot_r'
        im = ax.imshow(xp.abs(xp.roll(fun, (L, L), axis=(0,1))).transpose(), origin='lower', extent=[-L, L, -L, L], \
                        norm=colors.LogNorm(vmin=TolMin, vmax=xp.abs(fun).max()), cmap=color_map)
        fig.colorbar(im, orientation='vertical')
    elif dim == 3:
        Y, Z = xp.meshgrid(xp.arange(-L, L+1), xp.arange(-L, L+1))
        X = xp.zeros_like(Y)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_box_aspect((5/3, 1/3, 1/3))
        norm_c = colors.LogNorm(vmin=TolMin, vmax=xp.abs(fun).max())
        for k_ in range(-L, L+1):
            A = xp.abs(xp.roll(fun[k_, :, :], (L,L), axis=(0,1)))
            ax.plot_surface(X + k_, Y, Z, rstride=1, cstride=1, facecolors=cm.hot(norm_c(A)), alpha=0.4, linewidth=0.0, \
                            shade=False)
        ax.set_xticks((-L,0,L))
        ax.set_yticks((-L,0,L))
        ax.set_zticks((-L,0,L))
        ax.set_xlim((-L-1/2, L+1/2))
        ax.set_ylim((-L-1/2, L+1/2))
        ax.set_zlim((-L-1/2, L+1/2))
        ax.view_init(elev=20, azim=120)
    plt.pause(1e-17)


def conv_product(fun1, fun2):
    fun1_ = xp.roll(fun1, L_, axis=axisdim)
    fun2_ = xp.roll(fun2, L_, axis=axisdim)
    fun3_ = sps.convolve(fun1_, fun2_, mode='full', method='auto')
    return xp.roll(fun3_[JLconvdim], nL_, axis=axisdim)


def converge(h, display=False):
    h_ = copy.deepcopy(h)
    h_.error = ''
    itconv = 0
    while (TolMax > norm_int(h_.f) > TolMin) and (not h_.error):
        h_ = RG(h_)
        itconv += 1
        if display:
            print("norm = {:4e}".format(norm_int(h_.f)))
    if (norm_int(h_.f) <= TolMin) and (not h_.error):
        return True
    elif (norm_int(h_.f) >= TolMax) and (not h_.error):
        return False
    else:
        h.error = h_.error
        h.count = itconv
        return False


def approach(h_inf, h_sup, dist=Dist_Surf, strict=False):
    h_inf_ = copy.deepcopy(h_inf)
    h_sup_ = copy.deepcopy(h_sup)
    h_inf_.error = ''
    h_mid_ = copy.deepcopy(h_inf_)
    while norm(h_inf_.f - h_sup_.f) >= dist:
        h_mid_.f = (h_inf_.f + h_sup_.f) / 2.0
        if converge(h_mid_):
            h_inf_.f = h_mid_.f.copy()
        else:
            h_sup_.f = h_mid_.f.copy()
    if strict:
        h_mid_.f = (h_inf_.f + h_sup_.f) / 2.0
        delta_ = dist / norm(h_inf_.f - h_sup_.f)
        h_sup_.f = h_mid_.f + delta_ * (h_sup_.f - h_mid_.f)
        h_inf_.f = h_mid_.f + delta_ * (h_inf_.f - h_mid_.f)
    if not converge(h_inf_):
        print('Warning (approach): ' + h_inf_.error)
    if converge(h_sup_):
        print('Warning (approach): h_sup not above crit. surf.')
    else:
        h_sup_.error = ''
    return h_inf_, h_sup_


def norm(fun):
    if norm_choice == 'max':
        return xp.abs(fun).max()
    elif norm_choice == 'sum':
        return xp.abs(fun).sum()
    elif norm_choice == 'Euclidian':
        return xp.sqrt((xp.abs(fun) ** 2).sum())
    else:
        return (xp.exp(xp.log(xp.abs(fun)) + norm_choice * xp.sum(xp.abs(Nu), axis=0)).reshape(reshape_J)).max()


def norm_int(fun):
    fun_ = fun.copy()
    fun_[xp.index_exp[:J+1] + zero_] = 0.0
    return norm(fun_)


def sym(fun):
    fun_ = (fun + xp.roll(xp.flip(fun, axis=axisdim), 1, axis=axisdim).conj()) / 2.0
    fun_[0][zero_] = 0.0
    return fun_


class Hamiltonian:
    def __init__(self, omega, f, error='', count=0):
        self.Omega = omega
        self.f = f
        self.error = error
        self.count = count


def renormalization_group(h):
    h_ = copy.deepcopy(h)
    h_.error = ''
    if not fixed_Omega:
        Omega_ = (N.transpose()).dot(h_.Omega)
        ren = (2.0 * xp.sqrt((Omega_ ** 2).sum()) / eigenvalues[0] * h_.f[2][zero_]) \
                ** (2 - xp.arange(J+1, dtype=int)) / (2.0 * h_.f[2][zero_])
        h_.Omega = Omega_ / xp.sqrt((Omega_ ** 2).sum())
    else:
        ren = (2.0 * eigenvalues[1] / eigenvalues[0] * h_.f[2][zero_]) \
                ** (2 - xp.arange(J+1, dtype=int)) / (2.0 * h_.f[2][zero_])
    f_ = xp.zeros_like(h_.f)
    f_[Nu_mask] = h_.f[N_Nu_mask]
    f_ *= ren.reshape(reshape_L)
    Omega_Nu = xp.einsum('i,i...->...', h_.Omega, Nu).reshape(reshape_J)
    km_ = 0
    Iminus_f = xp.zeros_like(f_)
    Iminus_f[Iminus] = f_[Iminus]
    while (TolMax > norm(Iminus_f) > TolMin) and (TolMax > norm(f_) > TolMin) and (km_ < MaxIter) and (not h_.error):
        y_ = xp.zeros_like(f_)
        ao2 = - f_[1][zero_] / (2.0 * f_[2][zero_])
        y_[0][Iminus[0]] = f_[0][Iminus[0]] / omega_0_Nu[0][Iminus[0]]
        for m in range(1, J+1):
            y_[m][Iminus[m]] = (f_[m][Iminus[m]] - 2.0 * f_[2][zero_] * Omega_Nu[0][Iminus[m]] * y_[m-1][Iminus[m]])\
                                / omega_0_Nu[0][Iminus[m]]
        y_t = xp.roll(y_ * J_, -1, axis=0)
        f_t = xp.roll(f_ * J_, -1, axis=0)
        y_o = Omega_Nu * y_
        f_o = Omega_Nu * f_
        g_ = ao2 * f_t - omega_0_Nu * y_ + conv_product(y_t, f_o) - conv_product(y_o, f_t)
        k_ = 2
        while (TolMax > norm(g_) > TolLie) and (TolMax > norm(f_) > TolMin) and (k_ < MaxLie):
            f_ += g_
            g_t = xp.roll(g_ * J_, -1, axis=0)
            g_o = Omega_Nu * g_
            g_ = (ao2 * g_t + conv_product(y_t, g_o) - conv_product(y_o, g_t)) / precision_(k_)
            k_ += 1
        Iminus_f[Iminus] = f_[Iminus]
        km_ += 1
        f_ = sym(f_)
        if not (norm(g_) <= TolLie):
            if (norm(g_) >= TolMax):
                h_.error = 'Lie transform diverging ({}-th)'.format(km_)
            elif (k_ >= MaxLie):
                h_.error = 'Lie transform not converging ({}-th)'.format(km_)
    if (not (norm(Iminus_f) <= TolMin)) and (not h_.error):
        if (norm(Iminus_f) >= TolMax):
            h_.error = 'I- iterations diverging'
        elif (km_ >= MaxIter):
            h_.error = 'I- iterations not converging'
    h_.f = f_
    return h_


def generate_1Hamiltonian(k_modes, k_amp, omega, symmetric=False):
    f_ = xp.zeros((J+1,) + dim * (2*L+1,), dtype=precision_)
    if k_amp == 'random':
        for it in range(len(k_modes)):
            f_[k_modes[it]] = 2.0 * xp.random.random() - 1.0
    else:
        for it in range(len(k_modes)):
            f_[k_modes[it]] = k_amp[it]
    if symmetric:
        f_ = sym(f_)
    f_[2][zero_] = 0.5
    return Hamiltonian(omega, f_)


def generate_2Hamiltonians(k_modes, k_amp_inf, k_amp_sup, omega):
    h_inf = generate_1Hamiltonian(k_modes, k_amp_inf, omega, symmetric=True)
    h_sup = generate_1Hamiltonian(k_modes, k_amp_sup, omega, symmetric=True)
    h_sup.error = ''
    if not converge(h_inf):
        if not h_inf.error:
            h_inf.error = 'problem in generate_2Hamiltonians'
    if converge(h_sup):
        h_sup.error = 'problem in generate_2Hamiltonians'
    return h_inf, h_sup


def save_data(name, data, info=[]):
    mdic = {'L': L, 'J': J, 'TolMax': TolMax, 'TolMin': TolMin, 'TolLie': TolLie, 'Dist_Surf': Dist_Surf, \
            'MaxLie': MaxLie, 'MaxIter': MaxIter, 'Sigma': Sigma, 'Kappa': Kappa}
    mdic.update({'Dist_Surf': Dist_Surf, 'Radius': Radius, 'Modes_Perturb': Modes_Perturb})
    mdic.update({'data': data})
    mdic.update({'info': info})
    today = date.today()
    date_today = today.strftime(" %B %d, %Y\n")
    timestr = time.strftime("%Y%m%d_%H%M")
    email = ' cristel.chandre@univ-amu.fr'
    mdic.update({'date': date_today, 'author': email})
    if Save_Data:
        savemat(name + '_' + timestr + '.mat', mdic)


def iterates():
    h_inf, h_sup = generate_2Hamiltonians(K, K_amp_inf, K_amp_sup, Omega)
    if (not h_inf.error) and (not h_sup.error):
        plt.close('all')
        data = []
        print('starting iterations...')
        plotf(h_sup.f[0])
        start = time.time()
        k_ = 0
        while (k_ < Number_of_Iterations) and (not h_inf.error) and (not h_sup.error):
            k_ += 1
            start_k = time.time()
            h_inf, h_sup = approach(h_inf, h_sup, dist=Dist_Surf, strict=True)
            if k_ == 1:
                print('Critical parameter = {}'.format(2.0 * h_inf.f[K[0]]))
            h_inf_ = renormalization_group(h_inf)
            h_sup_ = renormalization_group(h_sup)
            plotf(h_inf_.f[0])
            mean2_p = 2.0 * h_inf.f[2][zero_]
            diff_p = norm(xp.abs(h_inf.f) - xp.abs(h_inf_.f))
            delta_p = norm(xp.abs(h_inf_.f) - xp.abs(h_sup_.f)) / norm(h_inf.f - h_sup.f)
            h_inf = copy.deepcopy(h_inf_)
            h_sup = copy.deepcopy(h_sup_)
            end_k = time.time()
            data.append([diff_p, delta_p, mean2_p])
            print("diff = %.3e    delta = %.7f   <f2> = %.7f    (done in %d seconds)" % \
                    (diff_p, delta_p, mean2_p, int(xp.rint(end_k-start_k))))
        if (k_ < Number_of_Iterations):
            print('Warning (iterates): ' + h_inf.error + ' / ' + h_sup.error)
        end = time.time()
        print("Computation done in {} seconds".format(int(xp.rint(end-start))))
        info = 'diff     delta     <f2>'
        save_data('RG_iterates', data, info=info)
        plt.ioff()
        plt.show()
    else:
        print('Warning (iterates): ' + h_inf.error + ' / ' + h_sup.error)


def renormalization_group_set(k, set):
    set[k] = renormalization_group(set[k])

def approach_set(k, set1, set2, dist, strict):
    set1[k], set2[k] = approach(set1[k], set2[k], dist=dist, strict=strict)

def iterate_circle():
    start = time.time()
    h_inf, h_sup = generate_2Hamiltonians(K, K_amp_inf, K_amp_sup, Omega)
    h_inf, h_sup = approach(h_inf, h_sup, dist=Dist_Surf, strict=True)
    h_inf = renormalization_group(h_inf)
    h_sup = renormalization_group(h_sup)
    h_inf, h_sup = approach(h_inf, h_sup, dist=Dist_Circle, strict=True)
    if (not h_inf.error) and (not h_sup.error):
        print('starting circle')
        hc_inf, hc_sup = approach(h_inf, h_sup, dist=Dist_Surf, strict=True)
        v1 = xp.zeros((J+1,) + dim * (2*L+1,), dtype=precision_)
        v2 = xp.zeros((J+1,) + dim * (2*L+1,), dtype=precision_)
        v1[0][dim * xp.index_exp[:Modes_Perturb]] = 2.0 * xp.random.random(dim * (Modes_Perturb,)) - 1.0
        v2[0][dim * xp.index_exp[:Modes_Perturb]] = 2.0 * xp.random.random(dim * (Modes_Perturb,)) - 1.0
        v1 = sym(v1)
        v2 = sym(v2)
        v2 = v2 - xp.vdot(v2, v1) * v1 / xp.vdot(v1, v1)
        v1 = Radius * v1 / xp.sqrt(xp.vdot(v1, v1))
        v2 = Radius * v2 / xp.sqrt(xp.vdot(v2, v2))
        circle_inf = []
        circle_sup = []
        for k_ in range(Nh+1):
            h_inf_ = copy.deepcopy(h_inf)
            h_sup_ = copy.deepcopy(h_sup)
            theta = precision_(k_) * 2.0 * xp.pi / precision_(Nh)
            h_inf_.f = h_inf.f + v1 * xp.cos(theta) + v2 * xp.sin(theta)
            h_sup_.f = h_sup.f + v1 * xp.cos(theta) + v2 * xp.sin(theta)
            circle_inf.append(h_inf_)
            circle_sup.append(h_sup_)
        pool = multiprocessing.Pool(num_cores)
        approach_circle = partial(approach_set, set1=circle_inf, set2=circle_sup, dist=Dist_Surf, strict=True)
        pool.imap(approach_circle, range(Nh+1))
        Coord = xp.zeros((Nh+1, 2, Number_of_Iterations))
        for i_ in tqdm(range(Number_of_Iterations)):
            for k_ in range(Nh+1):
                Coord[k_, :, i_] = [xp.vdot(circle_inf[k_].f - hc_inf.f, v1), xp.vdot(circle_inf[k_].f - hc_inf.f, v2)]
            save_data('RG_circle', Coord / Radius ** 2)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(Coord[:, 0, i_] / Radius ** 2, Coord[:, 1, i_] / Radius ** 2, label='%d -th iterate' % i_)
            ax.legend()
            plt.pause(1e-17)
            image_set = partial(renormalization_group_set, set=circle_inf)
            pool.map(image_set, range(Nh+1), chunksize=1)
            image_set = partial(renormalization_group_set, set=circle_sup)
            pool.map(image_set, range(Nh+1), chunksize=1)
            approach_circle = partial(approach_set, set1=circle_inf, set2=circle_sup, dist=Dist_Surf, strict=True)
            pool.map(approach_circle, range(Nh+1), chunksize=1)
            hc_inf = renormalization_group(hc_inf)
            hc_sup = renormalization_group(hc_sup)
            hc_inf, hc_sup = approach(hc_inf, hc_sup, dist=Dist_Surf, strict=True)
        end = time.time()
        print("Computation done in {} seconds".format(int(xp.rint(end-start))))
        plt.ioff()
        plt.show()
    else:
        print('Warning (iterate_circle): ' + h_inf.error + ' / ' + h_sup.error)


def compute_cr(epsilon):
    k_inf_ = K_amp_inf.copy()
    k_sup_ = K_amp_sup.copy()
    k_inf_[K_indx[0]] = K_amp_inf[K_indx[0]] + epsilon * (K_amp_sup[K_indx[0]] - K_amp_inf[K_indx[0]])
    k_sup_[K_indx[0]] = k_inf_[K_indx[0]]
    h_inf, h_sup = generate_2Hamiltonians(K, k_inf_, k_sup_, Omega)
    if converge(h_inf) and not converge(h_sup):
        h_inf, h_sup = approach(h_inf, h_sup, dist=TolCS)
        return 2.0 * xp.array([h_inf.f[K[K_indx[0]]], h_inf.f[K[K_indx[1]]]])
    else:
        return 2.0 * xp.array([h_inf.f[K[K_indx[0]]], xp.nan])

def critical_surface():
    if len(K_indx) != 2:
        print('Warning: 2 modes are required for the critical surface')
    else:
        epsilon_ = xp.linspace(0.0, 1.0, N_cs)
        start = time.time()
        pool = multiprocessing.Pool(num_cores)
        data = []
        for result in tqdm(pool.imap(func=compute_cr, iterable=epsilon_), total=len(epsilon_)):
            data.append(result)
        data = xp.array(data).transpose()
        end = time.time()
        print('Computation done in {} seconds'.format(int(xp.rint(end-start))))
        save_data('RG_critical_surface', data)
        fig = plt.figure()
        ax = fig.gca()
        ax.set_box_aspect(1)
        plt.plot(data[0, :], data[1, :], color='b', linewidth=2)
        ax.set_xlim(K_amp_inf[K_indx[0]], K_amp_sup[K_indx[0]])
        ax.set_ylim(K_amp_inf[K_indx[1]], K_amp_sup[K_indx[1]])
        plt.show()


def converge_point(epsilon1, epsilon2):
    k_amp_ = K_amp_sup.copy()
    k_amp_[K_indx[0]] = K_amp_inf[K_indx[0]] + epsilon1 * (K_amp_sup[K_indx[0]] - K_amp_inf[K_indx[0]])
    k_amp_[K_indx[1]] = K_amp_inf[K_indx[1]] + epsilon2 * (K_amp_sup[K_indx[1]] - K_amp_inf[K_indx[1]])
    h_ = generate_1Hamiltonian(K, k_amp_, Omega, symmetric=True)
    return [converge(h_), h_.count]

def layer():
    if len(K_indx) != 2:
        print('Warning: 2 modes are required for the layer')
    else:
        epsilon_ = xp.linspace(0.0, 1.0, N_cs)
        start = time.time()
        pool = multiprocessing.Pool(num_cores)
        data = []
        for i_ in tqdm(range(N_cs)):
            epsilon2_ = epsilon_[i_]
            converge_point_ = partial(converge_point, epsilon2=epsilon2_)
            data_temp = pool.imap(converge_point_, epsilon_)
            data.append(data_temp)
        end = time.time()
        print('Computation done in {} seconds'.format(int(xp.rint(end-start))))
        save_data('RG_layer', data)
        fig = plt.figure()
        ax = fig.gca()
        ax.set_box_aspect(1)
        im = ax.imshow(xp.array(data)[:, :, 0], cmap='Reds_r', extent=(K_amp_inf[K_indx[0]], K_amp_sup[K_indx[0]], K_amp_inf[K_indx[1]], K_amp_sup[K_indx[1]]))
        ax.set_xlim(K_amp_inf[K_indx[0]], K_amp_sup[K_indx[0]])
        ax.set_ylim(K_amp_inf[K_indx[1]], K_amp_sup[K_indx[1]])
        fig.colorbar(im)
        plt.show()
