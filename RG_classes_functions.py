import numpy as xp
from numpy import linalg as LA
import scipy.signal as sps
import copy

class Hamiltonian:
    def __init__(self, omega, f, error=[0, 0], count=0):
        self.Omega = omega
        self.f = f
        self.error = error
        self.count = count

class RG:
    def __init__(self, dict_param):
        for key in dict_param:
            setattr(self, key, dict_param[key])
        if self.Precision == 32:
            self.Precision = xp.float32
        elif self.Precision == 128:
            self.Precision = xp.float128
        else:
            self.Precision = xp.float64
        self.N = xp.asarray(self.N, dtype=int)
        self.dim = len(self.omega_0)
        self.omega_0 = xp.asarray(self.omega_0, dtype=self.Precision)
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
        self.Eigenvalue = xp.real(eigenval[xp.abs(eigenval) < 1])
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
        omega_0_nu_ = xp.repeat(xp.abs(self.omega_0_nu), self.J+1, axis=0) / xp.sqrt((self.omega_0 ** 2).sum())
        self.iminus = omega_0_nu_ > comp_im
        self.nu_mask = xp.index_exp[:self.J+1]
        self.N_nu_mask = xp.index_exp[:self.J+1]
        for it in range(self.dim):
            self.nu_mask += (self.nu[it][mask],)
            self.N_nu_mask += (N_nu[it][mask],)
        if self.CanonicalTransformation in ['Type2', 'Type3']:
            self.reshape_Je = (1,) + self.dim * (2*self.L+1,) + (1,) + self.dim * (1,)
            self.reshape_oa = (self.J+1,) + self.dim * (1,) + (self.J+1,) + self.dim * (1,)
            self.reshape_cs = (1,) + self.dim * (2*self.L+1,) + (1,) + self.dim * (2*self.L+1,)
            self.reshape_t = (self.J+1,) + self.dim * (2*self.L+1,) + (1,) + self.dim * (1,)
            self.reshape_av = (1,) + self.dim * (1,) + (self.J+1,) + self.dim * (1,)
            self.sum_dim = tuple(range(self.dim+1))
            reshape_Le = (self.J+1,) + self.dim * (1,) + (1,) + self.dim * (1,)
            self.Je_ = xp.arange(self.J+1, dtype=self.Precision).reshape(reshape_Le)
            self.omega_0_nu_e = self.omega_0_nu.reshape(self.reshape_Je)
            self.oa_vec = 2.0 * self.MaxCT * xp.random.rand(self.J+1) - self.MaxCT
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
        elif self.NormChoice == 'Analytic':
            return (xp.exp(xp.log(xp.abs(fun)) + self.NormAnalytic * xp.sum(xp.abs(self.nu), axis=0)).reshape(self.reshape_J)).max()

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
                sh_ *= self.MaxCT
                k_ = 2
                while (self.TolMax > self.norm(sh_) > self.TolLie) and (self.TolMax > self.norm(f_) > self.TolMin) and (k_ < self.MaxLie):
                    f_ += sh_
                    sh_t = xp.roll(sh_ * self.J_, -1, axis=0)
                    sh_o = omega_nu * sh_
                    sh_ = (ao2 * sh_t + self.conv_product(y_t, sh_o) - self.conv_product(y_o, sh_t)) / self.Precision(k_)
                    sh_ *= self.MaxCT
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

    def generate_1Hamiltonian(self, k_modes, k_amps, omega, symmetric=False):
        f_ = xp.zeros((self.J+1,) + self.dim * (2*self.L+1,), dtype=self.Precision)
        for (k_mode, k_amp) in zip(k_modes, k_amps):
            f_[k_mode] = k_amp
        if symmetric:
            f_ = self.sym(f_)
        f_[2][self.zero_] = 0.5
        return Hamiltonian(omega, f_)

    def generate_2Hamiltonians(self):
        h_inf = self.generate_1Hamiltonian(self.K, self.KampInf, self.Omega, symmetric=True)
        h_sup = self.generate_1Hamiltonian(self.K, self.KampSup, self.Omega, symmetric=True)
        if not self.converge(h_inf):
            h_inf.error = [4, 0]
        if self.converge(h_sup):
            h_sup.error = [-4, 0]
        else:
            h_sup.error = [0, 0]
        return h_inf, h_sup
