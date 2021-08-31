import numpy as xp
from numpy import linalg as LA
import matplotlib.pyplot as plt
import scipy.signal as sps
import copy
import itertools
import warnings
warnings.filterwarnings("ignore")
from RG_modules import compute_iterates, compute_surface, compute_region
from RG_dict import dict

def main():
    case = RG(dict)
    eval(case.Method + '(case)')
    plt.show()

class RG:
    def __repr__(self):
        return '{self.__class__.__name__}({self.DictParams})'.format(self=self)

    def __str__(self):
        return '{}D renormalization with N = {}'.format(self.dim, self.N.tolist())

    def __init__(self, dict_param):
        for key in dict_param:
            setattr(self, key, dict_param[key])
        self.DictParams = dict_param
        self.dim = len(self.omega0)
        self.zero_ = self.dim * (0,)
        self.one_ = self.dim * (1,)
        self.L_ = self.dim * (self.L,)
        self.nL_ = self.dim * (-self.L,)
        self.axis_dim = tuple(range(1, self.dim+1))
        self.r_jl = (self.J+1,) + self.dim * (2*self.L+1,)
        self.r_1l = (1,) + self.dim * (2*self.L+1,)
        self.r_j1 = (self.J+1,) + self.dim * (1,)
        self.conv_dim = xp.index_exp[:self.J+1] + self.dim * xp.index_exp[self.L:3*self.L+1]
        indx = self.dim * (xp.hstack((xp.arange(0, self.L+1), xp.arange(-self.L, 0))),)
        self.nu = xp.meshgrid(*indx, indexing='ij')
        eigenval, w_eig = LA.eig(self.N.transpose())
        self.Eigenvalue = xp.real(eigenval[xp.abs(eigenval) < 1])
        N_nu = xp.sign(self.Eigenvalue).astype(int) * xp.einsum('ij,j...->i...', self.N, self.nu)
        self.omega0_nu = xp.einsum('i,i...->...', self.omega0, self.nu).reshape(self.r_1l)
        self.omega_nu = xp.einsum('i,i...->...', self.Omega, self.nu).reshape(self.r_1l)
        mask = xp.prod(abs(N_nu) <= self.L, axis=0, dtype=bool)
        norm_nu = self.Precision(LA.norm(self.nu, axis=0)).reshape(self.r_1l)
        self.J_ = xp.arange(self.J+1, dtype=self.Precision).reshape(self.r_j1)
        if self.ChoiceIm == 'AKW1998':
            comp_im = self.Sigma * xp.repeat(norm_nu, self.J+1, axis=0) + self.Kappa * self.J_
        elif self.ChoiceIm =='K1999':
            comp_im = xp.maximum(self.Sigma * xp.repeat(norm_nu, self.J+1, axis=0), self.Kappa * self.J_)
        else:
            w_nu = xp.einsum('ij,j...->i...', w_eig, self.nu)
            norm_w_nu = LA.norm(w_nu, axis=0).reshape(self.r_1l)
            comp_im = self.Sigma * xp.repeat(norm_w_nu, self.J+1, axis=0) + self.Kappa * self.J_
        omega0_nu_ = xp.repeat(xp.abs(self.omega0_nu), self.J+1, axis=0) / LA.norm(self.omega0)
        self.iminus = omega0_nu_ > comp_im
        self.nu_mask = xp.index_exp[:self.J+1]
        self.N_nu_mask = xp.index_exp[:self.J+1]
        for _ in range(self.dim):
            self.nu_mask += (self.nu[_][mask],)
            self.N_nu_mask += (N_nu[_][mask],)
        self.norm = {
            'sum': lambda h: xp.abs(h).sum(),
            'max': lambda h: xp.abs(h).max(),
            'Euclidean': lambda h: xp.sqrt((xp.abs(h) ** 2).sum()),
            'Analytic': lambda h: xp.exp(xp.log(xp.abs(fun)) + self.NormAnalytic * xp.sum(xp.abs(self.nu), axis=0).reshape(self.r_1l)).max()
            }.get(self.NormChoice, lambda h: xp.abs(h).sum())

    def conv_product(self, fun1, fun2):
        fun1_ = xp.roll(fun1, self.L_, axis=self.axis_dim)
        fun2_ = xp.roll(fun2, self.L_, axis=self.axis_dim)
        fun3_ = sps.convolve(fun1_, fun2_, mode='full', method='auto')
        return xp.roll(fun3_[self.conv_dim], self.nL_, axis=self.axis_dim)

    def generate_y(self, h):
        y_ = xp.zeros_like(h.f)
        y_[0][self.iminus[0]] = h.f[0][self.iminus[0]] / self.omega0_nu[0][self.iminus[0]]
        for m in range(1, self.J+1):
            y_[m][self.iminus[m]] = (h.f[m][self.iminus[m]] - 2.0 * h.f[2][self.zero_] * self.omega_nu[0][self.iminus[m]] * y_[m-1][self.iminus[m]]) / self.omega0_nu[0][self.iminus[m]]
        return [y_, -h.f[1][self.zero_] / (2.0 * h.f[2][self.zero_]), xp.roll(y_ * self.J_, -1, axis=0), self.omega_nu * y_]

    def exp_ls(self, h, y, step):
        h_ = copy.deepcopy(h)
        h_.error = 0
        y_ = [y_elt * step for y_elt in y]
        f_d = [xp.roll(h_.f * self.J_, -1, axis=0), self.omega_nu * h_.f]
        sh_ = y_[1] * f_d[0] - self.omega0_nu * y_[0] + self.conv_product(y_[2], f_d[1]) - self.conv_product(y_[3], f_d[0])
        sh_[xp.abs(sh_) < self.TolMin**2] = 0.0
        h_.f += sh_
        k_ = 2
        while (self.TolMax > self.norm_int(sh_) > self.TolMinLie):
            sh_d = [xp.roll(sh_ * self.J_, -1, axis=0), self.omega_nu * sh_]
            sh_ = (y_[1] * sh_d[0] + self.conv_product(y_[2], sh_d[1]) - self.conv_product(y_[3], sh_d[0])) / self.Precision(k_)
            sh_[xp.abs(sh_) < self.TolMin**2] = 0.0
            h_.f += sh_
            k_ += 1
        if (self.norm(sh_) > self.TolMax):
                h_.error = 1
        return h_

    def exp_adapt(self, h, y, step):
        h_ = copy.deepcopy(h)
        if step < self.MinStep:
            h_.error = 5
            return self.exp_ls(h_, y, step)
        h1 = self.exp_ls(h_, y, step)
        h2 = self.exp_ls(self.exp_ls(h_, y, 0.5 * step), y, 0.5 * step)
        if self.norm_int(h1.f - h2.f) < self.AbsTol + self.RelTol * self.norm_int(h1.f):
            h_.f = 0.75 * h1.f + 0.25 * h2.f
            return h_
        else:
            return self.exp_adapt(self.exp_adapt(h_, y, 0.5 * step), y, 0.5 * step)

    def rg_map(self, h):
        h_ = copy.deepcopy(h)
        h_.error = 0
        omega_ = (self.N.transpose()).dot(h_.Omega)
        ren = (2.0 * LA.norm(omega_) / self.Eigenvalue * h_.f[2][self.zero_]) ** (2 - xp.arange(self.J+1, dtype=int)) / (2.0 * h_.f[2][self.zero_])
        h_.Omega = omega_ / LA.norm(omega_)
        self.omega_nu = xp.einsum('i,i...->...', h_.Omega, self.nu).reshape(self.r_1l)
        f_ = xp.zeros_like(h_.f)
        f_[self.nu_mask] = h_.f[self.N_nu_mask].copy()
        h_.f = f_ * ren.reshape(self.r_j1)
        k_ = 0
        iminus_f = xp.zeros_like(h_.f)
        iminus_f[self.iminus] = h_.f[self.iminus].copy()
        while (self.TolMax > self.norm(iminus_f) > self.TolMin) and (self.TolMax > self.norm_int(h_.f) > self.TolMin) and (k_ < self.MaxLie):
            y = self.generate_y(h_)
            if self.CanonicalTransformation == 'Lie':
                h_ = self.exp_ls(h_, y, 1.0)
            elif self.CanonicalTransformation == 'Lie_scaling':
                for _ in itertools.repeat(None, self.LieSteps):
                    h_ = self.exp_ls(h_, y, 1.0 / self.Precision(self.LieSteps))
            elif self.CanonicalTransformation == 'Lie_adaptive':
                    h_ = self.exp_adapt(h_, y, 1.0)
            iminus_f[self.iminus] = h_.f[self.iminus].copy()
            k_ += 1
            h_.f = self.sym(h_.f)
        if (self.norm(iminus_f) > self.TolMax):
            h_.error = 2
        elif (k_ > self.MaxLie):
            h_.error = -2
        return h_

    class Hamiltonian:
        def __repr__(self):
            return '{self.__class__.name__}({self.omega, self.f, self.error, self.count})'.format(self=self)

        def __str__(self):
            return 'a {self.__class__.name__} in action-angle variables'.format(self=self)

        def __init__(self, omega, f, error=0, count=0):
            self.Omega = omega
            self.f = f
            self.error = error
            self.count = count

    def generate_1Hamiltonian(self, ks, amps, omega, symmetric=False):
        f_ = xp.zeros(self.r_jl, dtype=self.Precision)
        for (k, amp) in zip(ks, amps):
            f_[k] = amp
        if symmetric:
            f_ = self.sym(f_)
        f_[2][self.zero_] = 0.5
        return self.Hamiltonian(omega, f_)

    def generate_2Hamiltonians(self, amps):
        h_inf = self.generate_1Hamiltonian(self.K, amps[0], self.Omega, symmetric=True)
        h_sup = self.generate_1Hamiltonian(self.K, amps[1], self.Omega, symmetric=True)
        if not self.converge(h_inf):
            h_inf.error = 4
        if self.converge(h_sup):
            h_sup.error = -4
        else:
            h_sup.error = 0
        return h_inf, h_sup

    def norm_int(self, fun):
        fun_ = fun.copy()
        fun_[xp.index_exp[:self.J+1] + self.zero_] = 0.0
        return self.norm(fun_)

    def sym(self, fun):
        fun_ = (fun + xp.roll(xp.flip(fun, axis=self.axis_dim), 1, axis=self.axis_dim).conj()) / 2.0
        fun_[0][self.zero_] = 0.0
        fun_[xp.abs(fun_) < self.TolMin**2] = 0.0
        return fun_

    def converge(self, h):
        h_ = copy.deepcopy(h)
        h_.error, it_conv = 0, 0
        while (self.TolMax > self.norm_int(h_.f) > self.TolMin) and (it_conv < self.MaxIterates):
            h_ = self.rg_map(h_)
            it_conv += 1
        if (self.norm_int(h_.f) < self.TolMin):
            h.count = - it_conv
            return True
        else:
            h.count = it_conv
            h.error = h_.error
            return False

    def approach(self, h, dist, strict=False):
        h_inf_, h_sup_ = copy.deepcopy((h[0], h[1]))
        h_inf_.error = 0
        h_mid_ = copy.deepcopy(h_inf_)
        while self.norm_int(h_inf_.f - h_sup_.f) > dist:
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
        if self.converge(h_sup_):
            h_sup_.error = 3
        else:
            h_sup_.error = 0
        return h_inf_, h_sup_

if __name__ == "__main__":
	main()
