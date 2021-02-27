from Parameters import J, L, Sigma, Kappa, ChoiceIm, Case
from Parameters import TolLie, TolMin, TolMax, DistSurf, Precision, MaxIter, MaxLie, NormChoice, CanonicalTransformation, MaxA
from Parameters import Radius, ModesPerturb, Nh, DistCircle
from Parameters import Kindx, KampInf, KampSup, Ncs, TolCS, SaveData, PlotResults
from Parameters import N, Omega0, Eigenvalues, Omega, FixedOmega, K, NumberOfIterations
import numpy as xp
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import scipy.signal as sps
from scipy.io import savemat
import copy
import time
from datetime import date
import multiprocessing
from functools import partial
from tqdm import tqdm, trange
import warnings
warnings.filterwarnings("ignore")

num_cores = multiprocessing.cpu_count()

if Precision == 32:
    precision_ = xp.float32
elif Precision == 128:
    precision_ = xp.float128
else:
    precision_ = xp.float64

N = xp.asarray(N, dtype=int)
dim = len(N[:, 0])
omega_0 = xp.asarray(Omega0, dtype=precision_)

eigenval, w_eig = LA.eig(N.transpose())

zero_ = dim * (0,)
one_ = dim * (1,)
L_ = dim * (L,)
nL_ = dim * (-L,)
axis_dim = tuple(range(1, dim+1))
reshape_J = (1,) + dim * (2*L+1,)
reshape_L = (J+1,) + dim * (1,)

conv_dim = xp.index_exp[:J+1] + dim * xp.index_exp[L:3*L+1]

indx = dim * (xp.hstack((xp.arange(0, L+1), xp.arange(-L, 0))),)
nu = xp.meshgrid(*indx, indexing='ij')
N_nu = xp.sign(eigenval[xp.abs(eigenval) < 1]).astype(int) * xp.einsum('ij,j...->i...', N, nu)
omega_0_nu = xp.einsum('i,i...->...', omega_0, nu).reshape(reshape_J)
mask = xp.prod(abs(N_nu) <= L, axis=0, dtype=bool)
norm_nu = LA.norm(nu, axis=0).reshape(reshape_J)
J_ = xp.arange(J+1, dtype=precision_).reshape(reshape_L)

if ChoiceIm == 'AKP1998':
    comp_im = Sigma * xp.repeat(norm_nu, J+1, axis=0) + Kappa * J_
elif ChoiceIm =='K1999':
    comp_im = xp.maximum(Sigma * xp.repeat(norm_nu, J+1, axis=0), Kappa * J_)
else:
    w_nu = xp.einsum('ij,j...->i...', w_eig, nu)
    norm_w_nu = LA.norm(w_nu, axis=0).reshape(reshape_J)
    comp_im = Sigma * xp.repeat(norm_w_nu, J+1, axis=0) + Kappa * J_
    inv_eig = xp.sort(1.0 / xp.abs(eigenval))[::-1]
    if inv_eig[1] + Sigma * (inv_eig[0] - inv_eig[1]) >= 1:
        print('Sigma is too large')

omega_0_nu_ = xp.repeat(xp.abs(omega_0_nu), J+1, axis=0) / xp.sqrt((omega_0 ** 2).sum())
iminus = omega_0_nu_ > comp_im
nu_mask = xp.index_exp[:J+1]
N_nu_mask = xp.index_exp[:J+1]
for it in range(dim):
    nu_mask += (nu[it][mask],)
    N_nu_mask += (N_nu[it][mask],)

if CanonicalTransformation in ['Type2', 'Type3']:
    reshape_Le = (J+1,) + dim * (1,) + (1,) + dim * (1,)
    reshape_Je = (1,) + dim * (2*L+1,) + (1,) + dim * (1,)
    reshape_oa = (J+1,) + dim * (1,) + (J+1,) + dim * (1,)
    reshape_cs = (1,) + dim * (2*L+1,) + (1,) + dim * (2*L+1,)
    reshape_t = (J+1,) + dim * (2*L+1,) + (1,) + dim * (1,)
    reshape_av = (1,) + dim * (1,) + (J+1,) + dim * (1,)
    sum_dim = tuple(range(dim+1))
    Je_ = xp.arange(J+1, dtype=precision_).reshape(reshape_Le)
    omega_0_nu_e = omega_0_nu.reshape(reshape_Je)
    oa_vec = 2.0 * MaxA * xp.random.rand(J+1) - MaxA
    oa_mat = xp.vander(oa_vec, increasing=True).transpose()
    indx = dim * (xp.hstack((xp.arange(0, L+1), xp.arange(-L, 0))),) + dim * (xp.arange(0, 2*L+1),)
    nu_nu = xp.meshgrid(*indx, indexing='ij')
    nu_phi = (2.0 * xp.pi * sum(nu_nu[k] * nu_nu[k+dim] for k in range(dim)) / precision_(2*L+1)).reshape(reshape_cs)
    exp_nu = xp.exp(1j * nu_phi)
    r3_av = (1,) + (J+1,) + dim * (1,)
    r3_oy = (1,) + (J+1,) + dim * (2*L+1,)
    r3_j = (J+1,) + (1,) + dim * (1,)
    r3_h = (J+1,) + (1,) + dim * (2*L+1,)


def plotf(fun):
    plt.rcParams.update({'font.size': 22})
    if dim == 2 and PlotResults:
        fig, ax = plt.subplots(1,1)
        ax.set_xlim(-L, L)
        ax.set_ylim(-L, L)
        color_map = 'hot_r'
        im = ax.imshow(xp.abs(xp.roll(fun, (L, L), axis=(0,1))).transpose(), origin='lower', extent=[-L, L, -L, L], \
                        norm=colors.LogNorm(vmin=TolMin, vmax=xp.abs(fun).max()), cmap=color_map)
        fig.colorbar(im, orientation='vertical')
    elif dim == 3 and PlotResults:
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
    fun1_ = xp.roll(fun1, L_, axis=axis_dim)
    fun2_ = xp.roll(fun2, L_, axis=axis_dim)
    fun3_ = sps.convolve(fun1_, fun2_, mode='full', method='auto')
    return xp.roll(fun3_[conv_dim], nL_, axis=axis_dim)


def converge(h, display=False):
    h_ = copy.deepcopy(h)
    h_.error = ''
    it_conv = 0
    while (TolMax > norm_int(h_.f) > TolMin) and (not h_.error):
        h_ = renormalization_group(h_)
        it_conv += 1
        if display:
            print("|H_{}| = {:4e}".format(it_conv, norm_int(h_.f)))
    if (norm_int(h_.f) <= TolMin) and (not h_.error):
        return True
    elif (norm_int(h_.f) >= TolMax) and (not h_.error):
        return False
    else:
        h.count = it_conv
        h.error = h_.error
        return False


def approach(h_inf, h_sup, dist=DistSurf, strict=False):
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
        h_sup_.error = 'below (approach)'
    else:
        h_sup_.error = ''
    return h_inf_, h_sup_


def norm(fun):
    if NormChoice == 'max':
        return xp.abs(fun).max()
    elif NormChoice == 'sum':
        return xp.abs(fun).sum()
    elif NormChoice == 'Euclidian':
        return xp.sqrt((xp.abs(fun) ** 2).sum())
    else:
        return (xp.exp(xp.log(xp.abs(fun)) + NormChoice * xp.sum(xp.abs(nu), axis=0)).reshape(reshape_J)).max()


def norm_int(fun):
    fun_ = fun.copy()
    fun_[xp.index_exp[:J+1] + zero_] = 0.0
    return norm(fun_)


def sym(fun):
    fun_ = (fun + xp.roll(xp.flip(fun, axis=axis_dim), 1, axis=axis_dim).conj()) / 2.0
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
    if not FixedOmega:
        omega_ = (N.transpose()).dot(h_.Omega)
        ren = (2.0 * xp.sqrt((omega_ ** 2).sum()) / Eigenvalues[0] * h_.f[2][zero_]) \
                ** (2 - xp.arange(J+1, dtype=int)) / (2.0 * h_.f[2][zero_])
        h_.Omega = omega_ / xp.sqrt((omega_ ** 2).sum())
    else:
        ren = (2.0 * Eigenvalues[1] / Eigenvalues[0] * h_.f[2][zero_]) \
                ** (2 - xp.arange(J+1, dtype=int)) / (2.0 * h_.f[2][zero_])
    f_ = xp.zeros_like(h_.f)
    f_[nu_mask] = h_.f[N_nu_mask]
    f_ *= ren.reshape(reshape_L)
    omega_nu = xp.einsum('i,i...->...', h_.Omega, nu).reshape(reshape_J)
    km_ = 0
    iminus_f = xp.zeros_like(f_)
    iminus_f[iminus] = f_[iminus]
    while (TolMax > norm(iminus_f) > TolMin) and (TolMax > norm(f_) > TolMin) and (km_ < MaxIter) and (not h_.error):
        y_ = xp.zeros_like(f_)
        ao2 = - f_[1][zero_] / (2.0 * f_[2][zero_])
        y_[0][iminus[0]] = f_[0][iminus[0]] / omega_0_nu[0][iminus[0]]
        for m in range(1, J+1):
            y_[m][iminus[m]] = (f_[m][iminus[m]] - 2.0 * f_[2][zero_] * omega_nu[0][iminus[m]] * y_[m-1][iminus[m]])\
                                / omega_0_nu[0][iminus[m]]
        if CanonicalTransformation == 'Lie':
            y_t = xp.roll(y_ * J_, -1, axis=0)
            f_t = xp.roll(f_ * J_, -1, axis=0)
            y_o = omega_nu * y_
            f_o = omega_nu * f_
            sh_ = ao2 * f_t - omega_0_nu * y_ + conv_product(y_t, f_o) - conv_product(y_o, f_t)
            k_ = 2
            while (TolMax > norm(sh_) > TolLie) and (TolMax > norm(f_) > TolMin) and (k_ < MaxLie):
                f_ += sh_
                sh_t = xp.roll(sh_ * J_, -1, axis=0)
                sh_o = omega_nu * sh_
                sh_ = (ao2 * sh_t + conv_product(y_t, sh_o) - conv_product(y_o, sh_t)) / precision_(k_)
                k_ += 1
            if not (norm(sh_) <= TolLie):
                if (norm(sh_) >= TolMax):
                    h_.error = 'Lie transform diverging ({}-th)'.format(km_)
                elif (k_ >= MaxLie):
                    h_.error = 'Lie transform not converging ({}-th)'.format(km_)
        elif CanonicalTransformation == 'Type2':
            dy_doa = xp.einsum('ji,j...->i...', oa_mat, xp.fft.ifftn(xp.roll(1j * y_ * J_, -1, axis=0), axes=axis_dim) * (2*L+1)**dim)
            ody_dphi = - xp.einsum('ji,j...->i...', oa_mat, xp.fft.ifftn(omega_nu.reshape(reshape_J) * y_, axes=axis_dim) * (2*L+1)**dim)
            o0dy_dphi = - xp.einsum('ji,j...->i...', oa_mat, xp.fft.ifftn(omega_0_nu.reshape(reshape_J) * y_, axes=axis_dim) * (2*L+1)**dim)
            exp_nu_mod = exp_nu * xp.exp(1j * omega_nu.reshape(reshape_Je) * dy_doa)
            coeff_f = xp.moveaxis(oa_mat.reshape(reshape_oa) * exp_nu_mod, range(dim+1), range(-dim-1, 0))
            oa_p = xp.power((oa_vec + ao2).reshape(r3_av) + ody_dphi.reshape(r3_oy), J_.reshape(r3_j))
            h_val = xp.fft.ifftn(f_, axes=axis_dim) * (2*L+1)**dim
            coeff_g = xp.einsum('j...,j...->...', oa_p, h_val.reshape(r3_h)) + o0dy_dphi
            f_ = xp.real(LA.tensorsolve(coeff_f, coeff_g))
        elif CanonicalTransformation == 'Type3':
            omega_nu_e = omega_nu.reshape(reshape_Je)
            f_e = f_.reshape(reshape_t)
            y_t = xp.sum(xp.roll(y_ * J_, -1, axis=0).reshape(reshape_t) * oa_mat.reshape(reshape_oa) * exp_nu, axis=sum_dim)
            y_e = y_.reshape(reshape_t) * oa_mat.reshape(reshape_oa) * exp_nu
            coeff_f = xp.moveaxis(xp.power(oa_vec.reshape(reshape_av) - ao2 + xp.sum(omega_nu_e * y_e, axis=sum_dim), Je_) * exp_nu, range(dim+1), range(-dim-1, 0))
            coeff_g = xp.sum(f_e * oa_mat.reshape(reshape_oa) * exp_nu * xp.exp(omega_nu_e * y_t) - omega_0_nu_e * y_e, axis=sum_dim)
            f_ = xp.real(LA.tensorsolve(coeff_f, coeff_g))
        iminus_f[iminus] = f_[iminus]
        km_ += 1
        f_ = sym(f_)
    if (not (norm(iminus_f) <= TolMin)) and (not h_.error):
        if (norm(iminus_f) >= TolMax):
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
    if not converge(h_inf):
        h_inf.error = 'above (generate_2Hamiltonians)'
    if converge(h_sup):
        h_sup.error = 'below (generate_2Hamiltonians)'
    else:
        h_sup.error = ''
    return h_inf, h_sup


def save_data(name, data, timestr, info=[]):
    mdic = {'L': L, 'J': J, 'TolMax': TolMax, 'TolMin': TolMin, 'TolLie': TolLie, 'DistSurf': DistSurf, \
            'MaxLie': MaxLie, 'MaxIter': MaxIter, 'Sigma': Sigma, 'Kappa': Kappa}
    mdic.update({'DistCircle': DistCircle, 'Radius': Radius, 'ModesPerturb': ModesPerturb})
    mdic.update({'Kindx': Kindx, 'KampInf': KampInf, 'KampSup': KampSup, 'TolCS': TolCS})
    mdic.update({'data': data})
    mdic.update({'info': info})
    today = date.today()
    date_today = today.strftime(" %B %d, %Y\n")
    email = ' cristel.chandre@univ-amu.fr'
    mdic.update({'date': date_today, 'author': email})
    if SaveData:
        savemat(name + '_' + Case + '_' + timestr + '.mat', mdic)


def iterates():
    h_inf, h_sup = generate_2Hamiltonians(K, KampInf, KampSup, Omega)
    if (not h_inf.error) and (not h_sup.error):
        timestr = time.strftime("%Y%m%d_%H%M")
        plt.close('all')
        plotf(h_sup.f[0])
        start = time.time()
        data = []
        k_ = 0
        while (k_ < NumberOfIterations) and (not h_inf.error) and (not h_sup.error):
            k_ += 1
            start_k = time.time()
            h_inf, h_sup = approach(h_inf, h_sup, dist=DistSurf, strict=True)
            h_inf_ = renormalization_group(h_inf)
            h_sup_ = renormalization_group(h_sup)
            if k_ == 1:
                print('Critical parameter = {}'.format(2.0 * h_inf.f[K[0]]))
            plotf(h_inf_.f[0])
            mean2_p = 2.0 * h_inf.f[2][zero_]
            diff_p = norm(xp.abs(h_inf.f) - xp.abs(h_inf_.f))
            delta_p = norm(xp.abs(h_inf_.f) - xp.abs(h_sup_.f)) / norm(h_inf.f - h_sup.f)
            data.append([diff_p, delta_p, mean2_p])
            h_inf = copy.deepcopy(h_inf_)
            h_sup = copy.deepcopy(h_sup_)
            end_k = time.time()
            print("diff = %.3e    delta = %.7f   <f2> = %.7f    (done in %d seconds)" % \
                    (diff_p, delta_p, mean2_p, int(xp.rint(end_k-start_k))))
            info = 'diff     delta     <f2>'
            save_data('RG_iterates', data, timestr, info=info)
        if (k_ < NumberOfIterations):
            print('Warning (iterates): ' + h_inf.error + ' / ' + h_sup.error)
        end = time.time()
        print("Computation done in {} seconds".format(int(xp.rint(end-start))))
        plt.show()
    else:
        print('Warning (iterates): ' + h_inf.error + ' / ' + h_sup.error)


def approach_set(k, set1, set2, dist, strict):
    set1[k], set2[k] = approach(set1[k], set2[k], dist=dist, strict=strict)

def iterate_circle():
    start = time.time()
    h_inf, h_sup = generate_2Hamiltonians(K, KampInf, KampSup, Omega)
    h_inf, h_sup = approach(h_inf, h_sup, dist=DistSurf, strict=True)
    h_inf = renormalization_group(h_inf)
    h_sup = renormalization_group(h_sup)
    h_inf, h_sup = approach(h_inf, h_sup, dist=DistCircle, strict=True)
    if (not h_inf.error) and (not h_sup.error):
        print('starting circle')
        timestr = time.strftime("%Y%m%d_%H%M")
        hc_inf, hc_sup = approach(h_inf, h_sup, dist=DistSurf, strict=True)
        v1 = xp.zeros((J+1,) + dim * (2*L+1,), dtype=precision_)
        v2 = xp.zeros((J+1,) + dim * (2*L+1,), dtype=precision_)
        v1[0][dim * xp.index_exp[:ModesPerturb]] = 2.0 * xp.random.random(dim * (ModesPerturb,)) - 1.0
        v2[0][dim * xp.index_exp[:ModesPerturb]] = 2.0 * xp.random.random(dim * (ModesPerturb,)) - 1.0
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
        approach_circle = partial(approach_set, set1=circle_inf, set2=circle_sup, dist=DistSurf, strict=True)
        pool.imap(approach_circle, iterable=range(Nh+1))
        Coord = xp.zeros((Nh+1, 2, NumberOfIterations))
        for i_ in trange(NumberOfIterations):
            for k_ in range(Nh+1):
                Coord[k_, :, i_] = [xp.vdot(circle_inf[k_].f - hc_inf.f, v1), xp.vdot(circle_inf[k_].f - hc_inf.f, v2)]
            save_data('RG_circle', Coord / Radius ** 2, timestr)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(Coord[:, 0, i_] / Radius ** 2, Coord[:, 1, i_] / Radius ** 2, label='%d -th iterate' % i_)
            ax.legend()
            plt.pause(1e-17)
            circle_inf = pool.imap(renormalization_group, iterable=circle_inf)
            circle_sup = pool.imap(renormalization_group, iterable=circle_sup)
            approach_circle = partial(approach_set, set1=circle_inf, set2=circle_sup, dist=DistSurf, strict=True)
            pool.imap(approach_circle, iterable=range(Nh+1))
            hc_inf = renormalization_group(hc_inf)
            hc_sup = renormalization_group(hc_sup)
            hc_inf, hc_sup = approach(hc_inf, hc_sup, dist=DistSurf, strict=True)
        end = time.time()
        print("Computation done in {} seconds".format(int(xp.rint(end-start))))
        plt.show()
    else:
        print('Warning (iterate_circle): ' + h_inf.error + ' / ' + h_sup.error)


def compute_cr(epsilon):
    k_inf_ = KampInf.copy()
    k_sup_ = KampSup.copy()
    k_inf_[Kindx[0]] = KampInf[Kindx[0]] + epsilon * (KampSup[Kindx[0]] - KampInf[Kindx[0]])
    k_sup_[Kindx[0]] = k_inf_[Kindx[0]]
    h_inf, h_sup = generate_2Hamiltonians(K, k_inf_, k_sup_, Omega)
    if converge(h_inf) and not converge(h_sup):
        h_inf, h_sup = approach(h_inf, h_sup, dist=TolCS)
        return 2.0 * xp.array([h_inf.f[K[Kindx[0]]], h_inf.f[K[Kindx[1]]]])
    else:
        return 2.0 * xp.array([h_inf.f[K[Kindx[0]]], xp.nan])

def critical_surface():
    if len(Kindx) != 2:
        print('Warning: 2 modes are required for the critical surface')
    else:
        timestr = time.strftime("%Y%m%d_%H%M")
        epsilon_ = xp.linspace(0.0, 1.0, Ncs)
        pool = multiprocessing.Pool(num_cores)
        data = []
        for result in tqdm(pool.imap(compute_cr, iterable=epsilon_), total=len(epsilon_)):
            data.append(result)
        data = xp.array(data).transpose()
        save_data('RG_critical_surface', data, timestr)
        fig = plt.figure()
        ax = fig.gca()
        ax.set_box_aspect(1)
        plt.plot(data[0, :], data[1, :], color='b', linewidth=2)
        ax.set_xlim(KampInf[Kindx[0]], KampSup[Kindx[0]])
        ax.set_ylim(KampInf[Kindx[1]], KampSup[Kindx[1]])
        plt.show()


def converge_point(val1, val2, display):
    k_amp_ = KampSup.copy()
    k_amp_[Kindx[0]] = val1
    k_amp_[Kindx[1]] = val2
    h_ = generate_1Hamiltonian(K, k_amp_, Omega, symmetric=True)
    return [int(converge(h_, display)), h_.count, h_.error]

def converge_region():
    if len(Kindx) != 2:
        print('Warning: 2 modes are required for converge_region')
    else:
        timestr = time.strftime("%Y%m%d_%H%M")
        x_vec = xp.linspace(KampInf[Kindx[0]], KampSup[Kindx[0]], Ncs)
        y_vec = xp.linspace(KampInf[Kindx[1]], KampSup[Kindx[1]], Ncs)
        pool = multiprocessing.Pool(num_cores)
        data = []
        for y_ in tqdm(y_vec):
            converge_point_ = partial(converge_point, val2=y_, display=False)
            for result in tqdm(pool.imap(converge_point_, iterable=x_vec), total=Ncs, leave=False):
                data.append(result)
            save_data('RG_converge_region', data, timestr)
        fig = plt.figure()
        ax = fig.gca()
        ax.set_box_aspect(1)
        im = ax.pcolor(x_vec, y_vec, xp.array(data)[:, 0].reshape((Ncs, Ncs)).astype(int), cmap='Reds_r')
        ax.set_xlim(KampInf[Kindx[0]], KampSup[Kindx[0]])
        ax.set_ylim(KampInf[Kindx[1]], KampSup[Kindx[1]])
        fig.colorbar(im)
        plt.show()
