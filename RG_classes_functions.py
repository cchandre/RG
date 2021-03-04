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

class Hamiltonian:
    def __init__(self, omega, f, error=[0, 0], count=0):
        self.Omega = omega
        self.f = f
        self.error = error
        self.count = count

class ModesInit:
    def __init__(self, k, kinf, ksup, omega):
        self.K = k
        self.KampInf = kinf
        self.KampSup = ksup
        self.Omega = omega

class RGcase:
    def __init__(self, N, omega_0, params):
        if params['Precision'] == 32:
            self.Precision = xp.float32
        elif params['Precision'] == 128:
            self.Precision = xp.float128
        else:
            self.Precision = xp.float64
        self.params = params
        self.NumCores = multiprocessing.cpu_count()
        self.L = params['L']
        self.J = params['J']
        self.TolMin = params['TolMin']
        self.TolMax = params['TolMax']
        self.TolLie = params['TolLie']
        self.DistSurf = params['DistSurf']
        self.MaxLie = params['MaxLie']
        self.MaxIter = params['MaxIter']
        self.Sigma = params['Sigma']
        self.Kappa = params['Kappa']
        self.SaveData = params['SaveData']
        self.PlotResults = params['PlotResults']
        self.ChoiceIm = params['ChoiceIm']
        self.NormChoice = params['NormChoice']
        self.CanonicalTransformation = params['CanonicalTransformation']
        self.NumberOfIterations = params['NumberOfIterations']
        self.MaxA = params['MaxA']
        self.DistCircle = params['DistCircle']
        self.Radius = params['Radius']
        self.ModesPerturb = params['ModesPerturb']
        self.Nh = params['Nh']
        self.Ncs = params['Ncs']
        self.TolCS = params['TolCS']
        self.N = xp.asarray(N, dtype=int)
        self.dim = len(omega_0)
        self.omega_0 = xp.asarray(omega_0, dtype=self.Precision)
        self.zero_ = self.dim * (0,)
        self.one_ = self.dim * (1,)
        self.L_ = self.dim * (self.L,)
        self.nL_ = self.dim * (-self.L,)
        self.axis_dim = tuple(range(1, self.dim+1))
        self.reshape_J = (1,) + self.dim * (2*self.L+1,)
        self.reshape_L = (self.J+1,) + self.dim * (1,)
        self.conv_dim = xp.index_exp[:self.J+1] + self.dim * xp.index_exp[self.L:3*self.L+1]
        indx = self.dim * (xp.hstack((xp.arange(0, self.L+1), xp.arange(-self.L, 0))),)
        self.nu = xp.meshgrid(*indx, indexing='ij')
        eigenval, w_eig = LA.eig(self.N.transpose())
        self.Eigenvalue = eigenval[xp.abs(eigenval) < 1]
        N_nu = xp.sign(self.Eigenvalue).astype(int) * xp.einsum('ij,j...->i...', self.N, self.nu)
        self.omega_0_nu = xp.einsum('i,i...->...', self.omega_0, self.nu).reshape(self.reshape_J)
        mask = xp.prod(abs(N_nu) <= self.L, axis=0, dtype=bool)
        norm_nu = LA.norm(self.nu, axis=0).reshape(self.reshape_J)
        self.J_ = xp.arange(self.J+1, dtype=self.Precision).reshape(self.reshape_L)
        if self.ChoiceIm == 'AKP1998':
            comp_im = self.Sigma * xp.repeat(norm_nu, self.J+1, axis=0) + self.Kappa * self.J_
        elif self.ChoiceIm =='K1999':
            comp_im = xp.maximum(self.Sigma * xp.repeat(norm_nu, self.J+1, axis=0), self.Kappa * self.J_)
        else:
            w_nu = xp.einsum('ij,j...->i...', w_eig, self.nu)
            norm_w_nu = LA.norm(w_nu, axis=0).reshape(self.reshape_J)
            comp_im = self.Sigma * xp.repeat(norm_w_nu, self.J+1, axis=0) + self.Kappa * self.J_
            inv_eig = xp.sort(1.0 / xp.abs(eigenval))[::-1]
            if inv_eig[1] + self.Sigma * (inv_eig[0] - inv_eig[1]) >= 1:
                print('Sigma is too large')
        omega_0_nu_ = xp.repeat(xp.abs(self.omega_0_nu), self.J+1, axis=0) / xp.sqrt((self.omega_0 ** 2).sum())
        self.iminus = omega_0_nu_ > comp_im
        self.nu_mask = xp.index_exp[:self.J+1]
        self.N_nu_mask = xp.index_exp[:self.J+1]
        for it in range(self.dim):
            self.nu_mask += (self.nu[it][mask],)
            self.N_nu_mask += (N_nu[it][mask],)
        if self.CanonicalTransformation in ['Type2', 'Type3']:
            self.reshape_Le = (self.J+1,) + self.dim * (1,) + (1,) + self.dim * (1,)
            self.reshape_Je = (1,) + self.dim * (2*self.L+1,) + (1,) + self.dim * (1,)
            self.reshape_oa = (self.J+1,) + self.dim * (1,) + (self.J+1,) + self.dim * (1,)
            self.reshape_cs = (1,) + self.dim * (2*self.L+1,) + (1,) + self.dim * (2*self.L+1,)
            self.reshape_t = (self.J+1,) + self.dim * (2*self.L+1,) + (1,) + self.dim * (1,)
            self.reshape_av = (1,) + self.dim * (1,) + (self.J+1,) + self.dim * (1,)
            sum_dim = tuple(range(self.dim+1))
            self.Je_ = xp.arange(self.J+1, dtype=self.Precision).reshape(self.reshape_Le)
            self.omega_0_nu_e = self.omega_0_nu.reshape(self.reshape_Je)
            self.oa_vec = 2.0 * self.MaxA * xp.random.rand(self.J+1) - self.MaxA
            self.oa_mat = xp.vander(self.oa_vec, increasing=True).transpose()
            indx = self.dim * (xp.hstack((xp.arange(0, self.L+1), xp.arange(-self.L, 0))),) + self.dim * (xp.arange(0, 2*self.L+1),)
            nu_nu = xp.meshgrid(*indx, indexing='ij')
            nu_phi = (2.0 * xp.pi * sum(nu_nu[k] * nu_nu[k+self.dim] for k in range(self.dim)) / self.Precision_(2*self.L+1)).reshape(self.reshape_cs)
            self.exp_nu = xp.exp(1j * nu_phi)
            self.r3_av = (1,) + (self.J+1,) + self.dim * (1,)
            self.r3_oy = (1,) + (self.J+1,) + self.dim * (2*self.L+1,)
            self.r3_j = (self.J+1,) + (1,) + self.dim * (1,)
            self.r3_h = (self.J+1,) + (1,) + self.dim * (2*self.L+1,)

    def norm(self, fun):
        if self.NormChoice == 'max':
            return xp.abs(fun).max()
        elif self.NormChoice == 'sum':
            return xp.abs(fun).sum()
        elif self.NormChoice == 'Euclidian':
            return xp.sqrt((xp.abs(fun) ** 2).sum())
        else:
            return (xp.exp(xp.log(xp.abs(fun)) + self.NormChoice * xp.sum(xp.abs(self.nu), axis=0)).reshape(self.reshape_J)).max()

    def norm_int(self, fun):
        fun_ = fun.copy()
        fun_[xp.index_exp[:self.J+1] + self.zero_] = 0.0
        return self.norm(fun_)

    def conv_product(self, fun1, fun2):
        fun1_ = xp.roll(fun1, self.L_, axis=self.axis_dim)
        fun2_ = xp.roll(fun2, self.L_, axis=self.axis_dim)
        fun3_ = sps.convolve(fun1_, fun2_, mode='full', method='auto')
        return xp.roll(fun3_[self.conv_dim], self.nL_, axis=self.axis_dim)

    def sym(self, fun):
        fun_ = (fun + xp.roll(xp.flip(fun, axis=self.axis_dim), 1, axis=self.axis_dim).conj()) / 2.0
        fun_[0][self.zero_] = 0.0
        return fun_

    def plotf(self, fun):
        plt.rcParams.update({'font.size': 22})
        if self.dim == 2:
            fig, ax = plt.subplots(1,1)
            ax.set_xlim(-self.L, self.L)
            ax.set_ylim(-self.L, self.L)
            color_map = 'hot_r'
            im = ax.imshow(xp.abs(xp.roll(fun, (self.L, self.L), axis=(0,1))).transpose(), origin='lower', extent=[-self.L, self.L, -self.L, self.L], \
                            norm=colors.LogNorm(vmin=self.TolMin, vmax=xp.abs(fun).max()), cmap=color_map)
            fig.colorbar(im, orientation='vertical')
        elif self.dim == 3:
            Y, Z = xp.meshgrid(xp.arange(-self.L, self.L+1), xp.arange(-self.L, self.L+1))
            X = xp.zeros_like(Y)
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.set_box_aspect((5/3, 1/3, 1/3))
            norm_c = colors.LogNorm(vmin=self.TolMin, vmax=xp.abs(fun).max())
            for k_ in range(-self.L, self.L+1):
                A = xp.abs(xp.roll(fun[k_, :, :], (self.L,self.L), axis=(0,1)))
                ax.plot_surface(X + k_, Y, Z, rstride=1, cstride=1, facecolors=cm.hot(norm_c(A)), alpha=0.4, linewidth=0.0, \
                                shade=False)
            ax.set_xticks((-self.L,0,self.L))
            ax.set_yticks((-self.L,0,self.L))
            ax.set_zticks((-self.L,0,self.L))
            ax.set_xlim((-self.L-1/2, self.L+1/2))
            ax.set_ylim((-self.L-1/2, self.L+1/2))
            ax.set_zlim((-self.L-1/2, self.L+1/2))
            ax.view_init(elev=20, azim=120)
        plt.pause(1e-17)

    def converge(self, h, display=False):
        h_ = copy.deepcopy(h)
        h_.error = [0, 0]
        it_conv = 0
        while (self.TolMax > self.norm_int(h_.f) > self.TolMin) and (h_.error == [0, 0]):
            h_ = self.renormalization_group(h_)
            it_conv += 1
            if display:
                print("|H_{}| = {:4e}".format(it_conv, self.norm_int(h_.f)))
        if (self.norm_int(h_.f) <= self.TolMin) and (h_.error == [0, 0]):
            return True
        elif (self.norm_int(h_.f) >= self.TolMax) and (h_.error == [0, 0]):
            return False
        else:
            h.count = it_conv
            h.error = h_.error
            return False

    def approach(self, h_inf, h_sup, dist, strict=False):
        h_inf_ = copy.deepcopy(h_inf)
        h_sup_ = copy.deepcopy(h_sup)
        h_inf_.error = [0, 0]
        h_mid_ = copy.deepcopy(h_inf_)
        while self.norm(h_inf_.f - h_sup_.f) >= dist:
            h_mid_.f = (h_inf_.f + h_sup_.f) / 2.0
            if self.converge(h_mid_):
                h_inf_.f = h_mid_.f.copy()
            else:
                h_sup_.f = h_mid_.f.copy()
        if strict:
            h_mid_.f = (h_inf_.f + h_sup_.f) / 2.0
            delta_ = dist / self.norm(h_inf_.f - h_sup_.f)
            h_sup_.f = h_mid_.f + delta_ * (h_sup_.f - h_mid_.f)
            h_inf_.f = h_mid_.f + delta_ * (h_inf_.f - h_mid_.f)
        if not self.converge(h_inf_):
            print('Warning (approach): ' + h_inf_.error)
        if self.converge(h_sup_):
            print('Warning (approach): h_sup not above crit. surf.')
            h_sup_.error = [3, 0]
        else:
            h_sup_.error = [0, 0]
        return h_inf_, h_sup_

    def renormalization_group(self, h):
        h_ = copy.deepcopy(h)
        h_.error = [0, 0]
        omega_ = (self.N.transpose()).dot(h_.Omega)
        ren = (2.0 * xp.sqrt((omega_ ** 2).sum()) / self.Eigenvalue * h_.f[2][self.zero_]) \
                ** (2 - xp.arange(self.J+1, dtype=int)) / (2.0 * h_.f[2][self.zero_])
        h_.Omega = omega_ / xp.sqrt((omega_ ** 2).sum())
        f_ = xp.zeros_like(h_.f)
        f_[self.nu_mask] = h_.f[self.N_nu_mask]
        f_ *= ren.reshape(self.reshape_L)
        omega_nu = xp.einsum('i,i...->...', h_.Omega, self.nu).reshape(self.reshape_J)
        km_ = 0
        iminus_f = xp.zeros_like(f_)
        iminus_f[self.iminus] = f_[self.iminus]
        while (self.TolMax > self.norm(iminus_f) > self.TolMin) and (self.TolMax > self.norm(f_) > self.TolMin) and (km_ < self.MaxIter) and (h_.error == [0, 0]):
            y_ = xp.zeros_like(f_)
            ao2 = - f_[1][self.zero_] / (2.0 * f_[2][self.zero_])
            y_[0][self.iminus[0]] = f_[0][self.iminus[0]] / self.omega_0_nu[0][self.iminus[0]]
            for m in range(1, self.J+1):
                y_[m][self.iminus[m]] = (f_[m][self.iminus[m]] - 2.0 * f_[2][self.zero_] * omega_nu[0][self.iminus[m]] * y_[m-1][self.iminus[m]])\
                                    / self.omega_0_nu[0][self.iminus[m]]
            if self.CanonicalTransformation == 'Lie':
                y_t = xp.roll(y_ * self.J_, -1, axis=0)
                f_t = xp.roll(f_ * self.J_, -1, axis=0)
                y_o = omega_nu * y_
                f_o = omega_nu * f_
                sh_ = ao2 * f_t - self.omega_0_nu * y_ + self.conv_product(y_t, f_o) - self.conv_product(y_o, f_t)
                k_ = 2
                while (self.TolMax > self.norm(sh_) > self.TolLie) and (self.TolMax > self.norm(f_) > self.TolMin) and (k_ < self.MaxLie):
                    f_ += sh_
                    sh_t = xp.roll(sh_ * self.J_, -1, axis=0)
                    sh_o = omega_nu * sh_
                    sh_ = (ao2 * sh_t + self.conv_product(y_t, sh_o) - self.conv_product(y_o, sh_t)) / self.Precision(k_)
                    k_ += 1
                if not (self.norm(sh_) <= self.TolLie):
                    if (self.norm(sh_) >= self.TolMax):
                        h_.error = [1, km_]
                    elif (k_ >= self.MaxLie):
                        h_.error = [-1, km_]
            elif self.CanonicalTransformation == 'Type2':
                dy_doa = xp.einsum('ji,j...->i...', self.oa_mat, xp.fft.ifftn(xp.roll(1j * y_ * self.J_, -1, axis=0), axes=self.axis_dim) * (2*self.L+1)**self.dim)
                ody_dphi = - xp.einsum('ji,j...->i...', self.oa_mat, xp.fft.ifftn(omega_nu.reshape(self.reshape_J) * y_, axes=self.axis_dim) * (2*self.L+1)**self.dim)
                o0dy_dphi = - xp.einsum('ji,j...->i...', self.oa_mat, xp.fft.ifftn(freq.omega_0_nu.reshape(self.reshape_J) * y_, axes=self.axis_dim) * (2*self.L+1)**self.dim)
                exp_nu_mod = self.exp_nu * xp.exp(1j * omega_nu.reshape(self.reshape_Je) * dy_doa)
                coeff_f = xp.moveaxis(self.oa_mat.reshape(self.reshape_oa) * exp_nu_mod, range(self.dim+1), range(-self.dim-1, 0))
                oa_p = xp.power((self.oa_vec + ao2).reshape(self.r3_av) + ody_dphi.reshape(self.r3_oy), self.J_.reshape(self.r3_j))
                h_val = xp.fft.ifftn(f_, axes=self.axis_dim) * (2*self.L+1)**self.dim
                coeff_g = xp.einsum('j...,j...->...', oa_p, h_val.reshape(self.r3_h)) + o0dy_dphi
                f_ = xp.real(LA.tensorsolve(coeff_f, coeff_g))
            elif self.CanonicalTransformation == 'Type3':
                omega_nu_e = omega_nu.reshape(self.reshape_Je)
                f_e = f_.reshape(self.reshape_t)
                y_t = xp.sum(xp.roll(y_ * self.J_, -1, axis=0).reshape(self.reshape_t) * self.oa_mat.reshape(self.reshape_oa) * self.exp_nu, axis=self.sum_dim)
                y_e = y_.reshape(self.reshape_t) * self.oa_mat.reshape(self.reshape_oa) * self.exp_nu
                coeff_f = xp.moveaxis(xp.power(self.oa_vec.reshape(self.reshape_av) - ao2 + xp.sum(omega_nu_e * y_e, axis=self.sum_dim), self.Je_) * self.exp_nu, range(self.dim+1), range(-self.dim-1, 0))
                coeff_g = xp.sum(f_e * self.oa_mat.reshape(self.reshape_oa) * self.exp_nu * xp.exp(omega_nu_e * y_t) - omega_0_nu_e * y_e, axis=self.sum_dim)
                f_ = xp.real(LA.tensorsolve(coeff_f, coeff_g))
            iminus_f[self.iminus] = f_[self.iminus]
            km_ += 1
            f_ = self.sym(f_)
        if (not (self.norm(iminus_f) <= self.TolMin)) and (h_.error == [0, 0]):
            if (self.norm(iminus_f) >= self.TolMax):
                h_.error = [2, 0]
            elif (km_ >= self.MaxIter):
                h_.error = [-2, 0]
        h_.f = f_
        return h_

    def generate_1Hamiltonian(self, k_modes, k_amp, omega, symmetric=False):
        f_ = xp.zeros((self.J+1,) + self.dim * (2*self.L+1,), dtype=self.Precision)
        if k_amp == 'random':
            for it in range(len(k_modes)):
                f_[k_modes[it]] = 2.0 * xp.random.random() - 1.0
        else:
            for it in range(len(k_modes)):
                f_[k_modes[it]] = k_amp[it]
        if symmetric:
            f_ = self.sym(f_)
        f_[2][self.zero_] = 0.5
        return Hamiltonian(omega, f_)

    def generate_2Hamiltonians(self, modes_i):
        h_inf = self.generate_1Hamiltonian(modes_i.K, modes_i.KampInf, modes_i.Omega, symmetric=True)
        h_sup = self.generate_1Hamiltonian(modes_i.K, modes_i.KampSup, modes_i.Omega, symmetric=True)
        if not self.converge(h_inf):
            h_inf.error = [4, 0]
        if self.converge(h_sup):
            h_sup.error = [-4, 0]
        else:
            h_sup.error = [0, 0]
        return h_inf, h_sup

    def iterates(self, modes_i):
        h_inf, h_sup = self.generate_2Hamiltonians(modes_i)
        if (h_inf.error == [0, 0]) and (h_sup.error == [0, 0]):
            timestr = time.strftime("%Y%m%d_%H%M")
            if self.PlotResults:
                self.plotf(h_sup.f[0])
            start = time.time()
            data = []
            k_ = 0
            while (k_ < self.NumberOfIterations) and (h_inf.error == [0, 0]) and (h_sup.error == [0, 0]):
                k_ += 1
                start_k = time.time()
                h_inf, h_sup = self.approach(h_inf, h_sup, dist=self.DistSurf, strict=True)
                h_inf_ = self.renormalization_group(h_inf)
                h_sup_ = self.renormalization_group(h_sup)
                if k_ == 1:
                    print('Critical parameter = {}'.format(2.0 * h_inf.f[modes_i.K[0]]))
                if self.PlotResults:
                    self.plotf(h_inf_.f[0])
                mean2_p = 2.0 * h_inf.f[2][self.zero_]
                diff_p = self.norm(xp.abs(h_inf.f) - xp.abs(h_inf_.f))
                delta_p = self.norm(xp.abs(h_inf_.f) - xp.abs(h_sup_.f)) / self.norm(h_inf.f - h_sup.f)
                data.append([diff_p, delta_p, mean2_p])
                h_inf = copy.deepcopy(h_inf_)
                h_sup = copy.deepcopy(h_sup_)
                end_k = time.time()
                print("diff = %.3e    delta = %.7f   <f2> = %.7f    (done in %d seconds)" % \
                        (diff_p, delta_p, mean2_p, int(xp.rint(end_k-start_k))))
                if self.SaveData:
                    info = 'diff     delta     <f2>'
                    save_data('RG_iterates', data, self.params, modes_i, timestr, info=info)
            if (k_ < self.NumberOfIterations):
                print('Warning (iterates): ' + h_inf.error + ' / ' + h_sup.error)
            end = time.time()
            print("Computation done in {} seconds".format(int(xp.rint(end-start))))
            plt.show()
        else:
            print('Warning (iterates): ' + h_inf.error + ' / ' + h_sup.error)

    def approach_set(self, k, set1, set2, dist, strict):
        set1[k], set2[k] = self.approach(set1[k], set2[k], dist=dist, strict=strict)

    def iterate_circle(self, modes_i):
        start = time.time()
        h_inf, h_sup = self.generate_2Hamiltonians(modes_i)
        h_inf, h_sup = self.approach(h_inf, h_sup, dist=self.DistSurf, strict=True)
        h_inf = self.renormalization_group(h_inf)
        h_sup = self.renormalization_group(h_sup)
        h_inf, h_sup = self.approach(h_inf, h_sup, dist=self.DistCircle, strict=True)
        if (h_inf.error == [0, 0]) and (h_sup.error == [0, 0]):
            print('starting circle')
            timestr = time.strftime("%Y%m%d_%H%M")
            hc_inf, hc_sup = self.approach(h_inf, h_sup, dist=self.DistSurf, strict=True)
            v1 = xp.zeros((self.J+1,) + self.dim * (2*self.L+1,), dtype=self.Precision)
            v2 = xp.zeros((self.J+1,) + self.dim * (2*self.L+1,), dtype=self.Precision)
            v1[0][self.dim * xp.index_exp[:self.ModesPerturb]] = 2.0 * xp.random.random(self.dim * (self.ModesPerturb,)) - 1.0
            v2[0][self.dim * xp.index_exp[:self.ModesPerturb]] = 2.0 * xp.random.random(self.dim * (self.ModesPerturb,)) - 1.0
            v1 = self.sym(v1)
            v2 = self.sym(v2)
            v2 = v2 - xp.vdot(v2, v1) * v1 / xp.vdot(v1, v1)
            v1 = self.Radius * v1 / xp.sqrt(xp.vdot(v1, v1))
            v2 = self.Radius * v2 / xp.sqrt(xp.vdot(v2, v2))
            circle_inf = []
            circle_sup = []
            for k_ in range(self.Nh+1):
                h_inf_ = copy.deepcopy(h_inf)
                h_sup_ = copy.deepcopy(h_sup)
                theta = self.Precision(k_) * 2.0 * xp.pi / self.Precision(self.Nh)
                h_inf_.f = h_inf.f + v1 * xp.cos(theta) + v2 * xp.sin(theta)
                h_sup_.f = h_sup.f + v1 * xp.cos(theta) + v2 * xp.sin(theta)
                circle_inf.append(h_inf_)
                circle_sup.append(h_sup_)
            pool = multiprocessing.Pool(self.NumCores)
            approach_circle = partial(self.approach_set, set1=circle_inf, set2=circle_sup, dist=self.DistSurf, strict=True)
            pool.imap(approach_circle, iterable=range(self.Nh+1))
            Coord = xp.zeros((self.Nh+1, 2, self.NumberOfIterations))
            for i_ in trange(self.NumberOfIterations):
                for k_ in range(self.Nh+1):
                    Coord[k_, :, i_] = [xp.vdot(circle_inf[k_].f - hc_inf.f, v1), xp.vdot(circle_inf[k_].f - hc_inf.f, v2)]
                if self.SaveData:
                    save_data('RG_circle', Coord / self.Radius ** 2, timestr, self.params, modes_i)
                if self.PlotResults:
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.plot(Coord[:, 0, i_] / self.Radius ** 2, Coord[:, 1, i_] / self.Radius ** 2, label='%d -th iterate' % i_)
                    ax.legend()
                    plt.pause(1e-17)
                renfunc = partial(self.renormalization_group)
                circle_inf = pool.imap(renfunc, iterable=circle_inf)
                circle_sup = pool.imap(renfunc, iterable=circle_sup)
                approach_circle = partial(self.approach_set, set1=circle_inf, set2=circle_sup, dist=self.DistSurf, strict=True)
                pool.imap(approach_circle, iterable=range(self.Nh+1))
                hc_inf = self.renormalization_group(hc_inf)
                hc_sup = self.renormalization_group(hc_sup)
                hc_inf, hc_sup = self.approach(hc_inf, hc_sup, dist=self.DistSurf, strict=True)
            end = time.time()
            print("Computation done in {} seconds".format(int(xp.rint(end-start))))
            plt.show()
        else:
            print('Warning (iterate_circle): ' + h_inf.error + ' / ' + h_sup.error)

    def compute_cr(self, epsilon, modes_i):
        k_inf_ = modes_i.KampInf.copy()
        k_sup_ = modes_i.KampSup.copy()
        k_inf_[0] = modes_i.KampInf[0] + epsilon * (modes_i.KampSup[0] - modes_i.KampInf[0])
        k_sup_[0] = k_inf_[0]
        modes_ = ModesInit(modes_i.K, k_inf_, k_sup_, modes_i.Omega)
        h_inf, h_sup = self.generate_2Hamiltonians(modes_)
        if self.converge(h_inf) and (not self.converge(h_sup)):
            h_inf, h_sup = self.approach(h_inf, h_sup, dist=self.TolCS)
            return 2.0 * xp.array([h_inf.f[modes_i.K[0]], h_inf.f[modes_i.K[1]]])
        else:
            return 2.0 * xp.array([h_inf.f[modes_i.K[0]], xp.nan])

    def critical_surface(self, modes_i):
        timestr = time.strftime("%Y%m%d_%H%M")
        epsilon_ = xp.linspace(0.0, 1.0, self.Ncs)
        pool = multiprocessing.Pool(self.NumCores)
        data = []
        compfun = partial(self.compute_cr, modes_i=modes_i)
        for result in tqdm(pool.imap(compfun, iterable=epsilon_), total=len(epsilon_)):
            data.append(result)
        data = xp.array(data).transpose()
        if self.SaveData:
            save_data('RG_critical_surface', data, timestr, self.params, modes_i)
        if self.PlotResults:
            fig = plt.figure()
            ax = fig.gca()
            ax.set_box_aspect(1)
            plt.plot(data[0, :], data[1, :], color='b', linewidth=2)
            ax.set_xlim(modes_i.KampInf[0], modes_i.KampSup[0])
            ax.set_ylim(modes_i.KampInf[1], modes_i.KampSup[1])
            plt.show()

    def converge_point(self, val1, val2, modes_i):
        k_amp_ = modes_i.KampSup.copy()
        k_amp_[0] = val1
        k_amp_[1] = val2
        h_ = self.generate_1Hamiltonian(modes_i.K, k_amp_, modes_i.Omega, symmetric=True)
        return [int(self.converge(h_)), h_.count], h_.error

    def converge_region(self, modes_i):
        timestr = time.strftime("%Y%m%d_%H%M")
        x_vec = xp.linspace(modes_i.KampInf[0], modes_i.KampSup[0], self.Ncs)
        y_vec = xp.linspace(modes_i.KampInf[1], modes_i.KampSup[1], self.Ncs)
        pool = multiprocessing.Pool(self.NumCores)
        data = []
        info = []
        for y_ in tqdm(y_vec):
            converge_point_ = partial(self.converge_point, val2=y_, modes_i=modes_i)
            for result_data, result_info in tqdm(pool.imap(converge_point_, iterable=x_vec), total=self.Ncs, leave=False):
                data.append(result_data)
                info.append(result_info)
            if self.SaveData:
                save_data('RG_converge_region', data, timestr, self.params, modes_i)
        if self.SaveData:
            save_data('RG_converge_region', xp.array(data).reshape((self.Ncs, self.Ncs, 2)), timestr, self.params, modes_i, info=xp.array(info).reshape((self.Ncs, self.Ncs, 2)))
        if self.PlotResults:
            fig = plt.figure()
            ax = fig.gca()
            ax.set_box_aspect(1)
            im = ax.pcolor(x_vec, y_vec, xp.array(data)[:, 0].reshape((self.Ncs, self.Ncs)).astype(int), cmap='Reds_r')
            ax.set_xlim(modes_i.KampInf[0], modes_i.KampSup[0])
            ax.set_ylim(modes_i.KampInf[1], modes_i.KampSup[1])
            fig.colorbar(im)
            plt.show()


def save_data(name, data, timestr, params, modes_i, info=[]):
    mdic = params.copy()
    mdic.update({'modes_i': modes_i})
    mdic.update({'data': data})
    mdic.update({'info': info})
    today = date.today()
    date_today = today.strftime(" %B %d, %Y\n")
    email = ' cristel.chandre@univ-amu.fr'
    mdic.update({'date': date_today, 'author': email})
    savemat(name + '_' + timestr + '.mat', mdic)
