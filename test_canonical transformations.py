J = 10
L = 10
dim = 2

Sigma = 0.26
Kappa = 0.3

TolLie = 1e-12
TolMin = 1e-12
TolMax = 1e+6
MaxLie = 50000

maxA = 0.1

N = [(1, 1), (1, 0)]
Eigenvalues = [-0.618033988749895, 1.618033988749895]
Omega0 = (Eigenvalues[0], 1.0)
Omega = [1.0, 0.0]
#K = ((0, 1, 0), (0, 1, 1))
#K_amp = (0.03, 0.03)
import random
nk = 10
Lk = 4
Jk = 4
K = [(random.randrange(0, Jk), random.randrange(-Lk, Lk), random.randrange(-Lk, Lk)) for i in range(nk)]
K_amp = nk * (0.05,)

import numpy as xp
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.signal as sps
import time

precision_ = xp.float64
zero_ = dim * (0,)
one_ = dim * (1,)
L_ = dim * (L,)
nL_ = dim * (-L,)
axis_dim = tuple(range(1, dim+1))
reshape_J = (1,) + dim * (2*L+1,)
reshape_L = (J+1,) + dim * (1,)
reshape_Le = (J+1,) + dim * (1,) + (1,) + dim * (1,)
reshape_Je = (1,) + dim * (2*L+1,) + (1,) + dim * (1,)
reshape_oa = (J+1,) + dim * (1,) + (J+1,) + dim * (1,)
reshape_cs = dim * (2*L+1,) + (1,) + dim * (2*L+1,)
reshape_t = (J+1,) + dim * (2*L+1,) + (1,) + dim * (1,)
reshape_ti = (J+1,) + dim * (1,) + (J+1,) + dim * (2*L+1,)
reshape_av = (1,) + dim * (1,) + (J+1,) + dim * (1,)
r3_av = (1,) + (J+1,) + dim * (1,)
r3_oy = (1,) + (J+1,) + dim * (2*L+1,)
r3_j = (J+1,) + (1,) + dim * (1,)
r3_h = (J+1,) + (1,) + dim * (2*L+1,)
sum_dim = tuple(range(dim+1))
conv_dim = xp.index_exp[:J+1] + dim * xp.index_exp[L:3*L+1]

J_ = xp.arange(J+1, dtype=precision_).reshape(reshape_L)
Je_ = xp.arange(J+1, dtype=precision_).reshape(reshape_Le)
indx = dim * (xp.hstack((xp.arange(0, L+1), xp.arange(-L, 0))),)
nu = xp.meshgrid(*indx, indexing='ij')
norm_nu = LA.norm(nu, axis=0).reshape(reshape_J)
omega_0 = xp.asarray(Omega0, dtype=precision_)
omega_0_nu = xp.einsum('i,i...->...', omega_0, nu).reshape(reshape_J)
omega_0_nu_e = omega_0_nu.reshape(reshape_Je)

comp_im = Sigma * xp.repeat(norm_nu, J+1, axis=0) + Kappa * J_
omega_0_nu_ = xp.repeat(xp.abs(omega_0_nu), J+1, axis=0) / xp.sqrt((omega_0 ** 2).sum())
iminus = omega_0_nu_ > comp_im

oa_vec = 2.0 * maxA * xp.random.rand(J+1) - maxA
oa_inv = LA.inv(xp.vander(oa_vec, increasing=True))
oa_mat = xp.vander(oa_vec, increasing=True).transpose()
indx = dim * (xp.hstack((xp.arange(0, L+1), xp.arange(-L, 0))),) + dim * (xp.arange(0, 2*L+1),)
nu_nu = xp.meshgrid(*indx, indexing='ij')
nu_phi = (2.0 * xp.pi * sum(nu_nu[k] * nu_nu[k+dim] for k in range(dim)) / precision_(2*L+1)).reshape(reshape_cs)
exp_nu = xp.exp(1j * nu_phi)

class Hamiltonian:
    def __init__(self, omega, f, error='', count=0):
        self.Omega = omega
        self.f = f
        self.error = error
        self.count = count

def Lie_transf(h, y):
    omega_nu = xp.einsum('i,i...->...', h.Omega, nu).reshape(reshape_J)
    f_ = h.f.copy()
    ao2 = - f_[1][zero_] / (2.0 * f_[2][zero_])
    y_ = y.copy()
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
    return f_

def type3_transf(h, y):
    omega_nu_e = xp.einsum('i,i...->...', h.Omega, nu).reshape(reshape_Je)
    ao2 = - h.f[1][zero_] / (2.0 * h.f[2][zero_])
    f_e = h.f.copy().reshape(reshape_t)
    y_t = xp.sum(xp.roll(y * J_, -1, axis=0).reshape(reshape_t) * oa_mat.reshape(reshape_oa) * exp_nu, axis=sum_dim)
    y_e = y.reshape(reshape_t) * oa_mat.reshape(reshape_oa) * exp_nu
    coeff_f = xp.moveaxis(xp.power(oa_vec.reshape(reshape_av) - ao2 + xp.sum(omega_nu_e * y_e, axis=sum_dim), Je_) * exp_nu, range(dim+1), range(-dim-1, 0))
    coeff_g = xp.sum(f_e * oa_mat.reshape(reshape_oa) * exp_nu * xp.exp(omega_nu_e * y_t) - omega_0_nu_e * y_e, axis=sum_dim)
    temp_ = xp.real(LA.tensorsolve(coeff_f, coeff_g))
    return temp_

def type2_transf_old(h, y):
    omega_nu_ = xp.einsum('i,i...->...', h.Omega, nu)
    ao2 = - h.f[1][zero_] / (2.0 * h.f[2][zero_])
    f_ = h.f.copy()
    dy_doa = xp.einsum('ji,j...->i...', oa_mat, xp.fft.ifftn(xp.roll(1j * y * J_, -1, axis=0), axes=axis_dim) * (2*L+1)**dim)
    ody_dphi = - xp.einsum('ji,j...->i...', oa_mat, xp.fft.ifftn(omega_nu_.reshape(reshape_J) * y, axes=axis_dim) * (2*L+1)**dim)
    o0dy_dphi = - xp.einsum('ji,j...->i...', oa_mat, xp.fft.ifftn(omega_0_nu.reshape(reshape_J) * y, axes=axis_dim) * (2*L+1)**dim)
    exp_nu_mod = exp_nu * xp.exp(1j * omega_nu_.reshape(reshape_Je) * dy_doa)
    coeff_f = xp.moveaxis(oa_mat.reshape(reshape_oa) * exp_nu_mod, range(dim+1), range(-dim-1, 0))
    oa_p = xp.power((oa_vec + ao2).reshape(r3_av) + ody_dphi.reshape(r3_oy), J_.reshape(r3_j))
    h_val = xp.fft.ifftn(f_, axes=axis_dim) * (2*L+1)**dim
    coeff_g = xp.einsum('j...,j...->...', oa_p, h_val.reshape(r3_h)) + o0dy_dphi
    return xp.real(LA.tensorsolve(coeff_f, coeff_g))

def type2_transf(h, y):
    reshape_je = dim * (2*L+1,) + (1,) + dim * (1,)
    r2_av = (J+1,) + dim * (1,)
    r2_j = (J+1,) + dim * (1,)
    r2_h = (J+1,) + dim * (2*L+1,)
    omega_nu_ = xp.einsum('i,i...->...', h.Omega, nu)
    ao2 = - h.f[1][zero_] / (2.0 * h.f[2][zero_])
    f_ = h.f.copy()
    dy_doa = xp.einsum('ji,j...->i...', oa_mat, xp.fft.ifftn(xp.roll(1j * y * J_, -1, axis=0), axes=axis_dim) * (2*L+1)**dim)
    ody_dphi = - xp.einsum('ji,j...->i...', oa_mat, xp.fft.ifftn(omega_nu_.reshape(reshape_J) * y, axes=axis_dim) * (2*L+1)**dim)
    o0dy_dphi = - xp.einsum('ji,j...->i...', oa_mat, xp.fft.ifftn(omega_0_nu.reshape(reshape_J) * y, axes=axis_dim) * (2*L+1)**dim)
    exp_nu_mod = exp_nu * xp.exp(1j * omega_nu_.reshape(reshape_je) * dy_doa)
    coeff_f = xp.moveaxis(exp_nu_mod, range(dim), range(-dim, 0))
    print(coeff_f.shape)
    oa_p = xp.power((oa_vec + ao2).reshape(r2_av) + ody_dphi, J_.reshape(r2_j))
    h_val = xp.fft.ifftn(f_, axes=axis_dim) * (2*L+1)**dim
    coeff_g = xp.einsum('j...,j...->...', oa_p, h_val.reshape(r2_h)) + o0dy_dphi
    return xp.einsum('ij,j...->i...', oa_inv, xp.real(LA.tensorsolve(coeff_f, coeff_g)))

def generating_function(h):
    omega_nu = xp.einsum('i,i...->...', h.Omega, nu).reshape(reshape_J)
    f_ = h.f
    y_ = xp.zeros_like(h.f)
    y_[0][iminus[0]] = f_[0][iminus[0]] / omega_0_nu[0][iminus[0]]
    for m in range(1, J+1):
        y_[m][iminus[m]] = (f_[m][iminus[m]] - 2.0 * f_[2][zero_] * omega_nu[0][iminus[m]] * y_[m-1][iminus[m]])\
                            / omega_0_nu[0][iminus[m]]
    return y_

def conv_product(fun1, fun2):
    fun1_ = xp.roll(fun1, L_, axis=axis_dim)
    fun2_ = xp.roll(fun2, L_, axis=axis_dim)
    fun3_ = sps.convolve(fun1_, fun2_, mode='full', method='auto')
    return xp.roll(fun3_[conv_dim], nL_, axis=axis_dim)

def sym(fun):
    fun_ = (fun + xp.roll(xp.flip(fun, axis=axis_dim), 1, axis=axis_dim).conj()) / 2.0
    fun_[0][zero_] = 0.0
    return fun_

def norm(fun):
    return xp.abs(fun).max()

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

def plotf(fun, plot_results=True, vmin=None, vmax=None):
    plt.rcParams.update({'font.size': 22})
    if dim == 2 and plot_results:
        fig, ax = plt.subplots(1,1)
        ax.set_xlim(-L, L)
        ax.set_ylim(-L, L)
        color_map = 'hot_r'
        im = ax.imshow(xp.roll(fun, (L, L), axis=(0,1)).transpose(), origin='lower', extent=[-L, L, -L, L], cmap=color_map, vmin=vmin, vmax=vmax)
        fig.colorbar(im, orientation='vertical')
    elif dim == 3 and plot_results:
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

if __name__ == '__main__':
    h = generate_1Hamiltonian(K, K_amp, Omega, symmetric=True)
    y = generating_function(h)
    ind_repr = 1
    start = time.time()
    f1 = Lie_transf(h, y)
    print(time.time()-start)
    start = time.time()
    f2 = type2_transf(h, y)
    print(time.time()-start)
    start = time.time()
    f3 = type3_transf(h, y)
    print(time.time()-start)
    plotf(f1[ind_repr])
    vmin = xp.amin(f1[ind_repr])
    vmax = xp.amax(f1[ind_repr])
    plotf(f2[ind_repr], vmin=vmin, vmax=vmax)
    plotf(f3[ind_repr], vmin=vmin, vmax=vmax)
    plt.show()
