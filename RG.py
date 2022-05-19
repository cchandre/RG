#
# BSD 2-Clause License
#
# Copyright (c) 2021, Cristel Chandre
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as xp
import scipy.sparse.linalg as sla
import scipy.linalg as la
import matplotlib.pyplot as plt
import scipy.signal as sps
import copy
import itertools
import warnings
warnings.filterwarnings("ignore")
from RG_modules import compute_iterates, compute_surface, compute_region, compute_line
from RG_dict import dict
import time

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
        self.vecjl = xp.prod(xp.asarray(self.r_jl))
        self.conv_dim = xp.index_exp[:self.J+1] + self.dim * xp.index_exp[self.L:3*self.L+1]
        indx = self.dim * (xp.hstack((xp.arange(0, self.L+1), xp.arange(-self.L, 0))),)
        self.nu = xp.meshgrid(*indx, indexing='ij')
        eigenval, w_eig = la.eig(self.N.transpose())
        self.Eigenvalue = xp.real(eigenval[xp.abs(eigenval) < 1])
        N_nu = xp.sign(self.Eigenvalue).astype(int) * xp.einsum('ij,j...->i...', self.N, self.nu)
        self.omega0_nu = xp.einsum('i,i...->...', self.omega0, self.nu).reshape(self.r_1l)
        self.omega_nu = xp.einsum('i,i...->...', self.Omega, self.nu).reshape(self.r_1l)
        mask = xp.prod(abs(N_nu) <= self.L, axis=0, dtype=bool)
        norm_nu = self.Precision(la.norm(self.nu, axis=0)).reshape(self.r_1l)
        self.J_ = xp.arange(self.J+1, dtype=self.Precision).reshape(self.r_j1)
        self.derivs = lambda f: [xp.roll(f * self.J_, -1, axis=0), self.omega_nu * f]
        if self.ChoiceIm == 'AKW1998':
            comp_im = self.Sigma * xp.repeat(norm_nu, self.J+1, axis=0) + self.Kappa * self.J_
        elif self.ChoiceIm =='K1999':
            comp_im = xp.maximum(self.Sigma * xp.repeat(norm_nu, self.J+1, axis=0), self.Kappa * self.J_)
        else:
            w_nu = xp.einsum('ij,j...->i...', w_eig, self.nu)
            norm_w_nu = la.norm(w_nu, axis=0).reshape(self.r_1l)
            comp_im = self.Sigma * xp.repeat(norm_w_nu, self.J+1, axis=0) + self.Kappa * self.J_
        omega0_nu_ = xp.repeat(xp.abs(self.omega0_nu), self.J+1, axis=0) / la.norm(self.omega0)
        self.iminus = omega0_nu_ > comp_im
        self.nu_mask = xp.index_exp[:self.J+1]
        self.N_nu_mask = xp.index_exp[:self.J+1]
        for _ in range(self.dim):
            self.nu_mask += (self.nu[_][mask],)
            self.N_nu_mask += (N_nu[_][mask],)
        self.norm = {
            'sum': lambda _: xp.abs(_).sum(),
            'max': lambda _: xp.abs(_).max(),
            'Euclidean': lambda _: xp.sqrt((xp.abs(_) ** 2).sum()),
            'Analytic': lambda _: xp.exp(xp.log(xp.abs(_)) + self.NormAnalytic * xp.sum(xp.abs(self.nu), axis=0).reshape(self.r_1l)).max()
            }.get(self.NormChoice, lambda _: xp.abs(_).sum())
        self.theta = {
                    1: 2.29e-16, 2: 2.58e-8, 3: 1.39e-5, 4: 3.40e-4, 5: 2.40e-3,
                    6: 9.07e-3, 7: 2.38e-2, 8: 5.00e-2, 9: 8.96e-2, 10: 1.44e-1,
                    11: 2.14e-1, 12: 3.00e-1, 13: 4.00e-1, 14: 5.14e-1, 15: 6.41e-1,
                    16: 7.81e-1, 17: 9.31e-1, 18: 1.09, 19: 1.26, 20: 1.44, 21: 1.62,
                    22: 1.82, 23: 2.01, 24: 2.22, 25: 2.43, 26: 2.64, 27: 2.86,
                    28: 3.08, 29: 3.31, 30: 3.54, 35: 4.7, 40: 6.0, 45: 7.2,
                    50: 8.5, 55: 9.9}

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
        return y_, -h.f[1][self.zero_] / (2.0 * h.f[2][self.zero_]), xp.roll(y_ * self.J_, -1, axis=0), self.omega_nu * y_, -self.omega0_nu * y_

    def liouville(self, y, f):
        f_ = self.derivs(f.reshape(self.r_jl))
        return y[1] * f_[0] + self.conv_product(y[2], f_[1]) - self.conv_product(y[3], f_[0])

    def pnorm(self, y, p=1):
        Ls = lambda f: self.liouville(y, f).flatten()
        Ls_H = lambda f: -self.liouville(y, f).flatten()
        A = sla.LinearOperator(shape=(self.vecjl, self.vecjl), matvec=Ls, rmatvec=Ls_H, dtype=self.Precision)**p
        return sla.onenormest(A)**(1/p)

    def expm_multiply(self, h, y, tol=2**-53):
        h_ = copy.deepcopy(h)
        h_.error = 0
        m_star, scale = self.compute_m_s(y)
        y_ = [item / scale for item in y]
        for s in range(scale):
            sh_ = y_[4] + self.liouville(y_, h_.f)
            h_.f += sh_
            c1 = self.norm_int(sh_)
            for m in range(2, m_star+1):
                sh_ = self.liouville(y_, sh_) / self.Precision(m)
                c2 = self.norm_int(sh_)
                h_.f += sh_
                if c1 + c2 <= tol * self.norm_int(h_.f):
                    break
                elif c1 + c2 >= self.TolMax:
                    h_.error = 1
                    break
                c1 = c2
        return h_

    def compute_p_max(self, m_max):
        sqrt_m_max = xp.sqrt(m_max)
        p_low, p_high = int(xp.floor(sqrt_m_max)), int(xp.ceil(sqrt_m_max + 1))
        return max(p for p in range(p_low, p_high +1) if p * (p - 1) <= m_max + 1)

    def compute_m_s(self, y_, n0=1, tol=2**-53, m_max=55, ell=2):
        best_m, best_s = None, None
        norm_ls = self.pnorm(y_)
        p_max = self.compute_p_max(m_max)
        if norm_ls <= 2 * ell * p_max * (p_max + 3) * self.theta[m_max] / float(n0 * m_max):
            for m, theta in self.theta.items():
                s = int(xp.ceil(norm_ls / theta))
                if best_m is None or m * s < best_m * best_s:
                    best_m, best_s = m, s
        else:
            norm_d = xp.array([self.pnorm(y_, p) for p in range(2, p_max + 2)])
            norm_alpha = xp.maximum(norm_d, xp.roll(norm_d, -1))
            for p in range(2, p_max + 1):
                for m in range(p * (p - 1) - 1, m_max + 1):
                    if m in self.theta:
                        s = int(xp.ceil(norm_alpha[p - 2] / self.theta[m]))
                        if best_m is None or m * s < best_m * best_s:
                            best_m, best_s = m, s
            best_s = max(best_s, 1)
        return best_m, best_s

    def expm_onestep(self, h, y, step=1, tol=2**-53):
        h_ = copy.deepcopy(h)
        h_.error = 0
        y_ = [item * step for item in y]
        sh_ = y_[4] + self.liouville(y_, h_.f)
        h_.f += sh_
        m = 2
        c1 = self.norm_int(sh_)
        while True:
            sh_ = self.liouville(y_, sh_) / self.Precision(m)
            c2 = self.norm_int(sh_)
            h_.f += sh_
            if c1 + c2 <= tol * self.norm_int(h_.f):
                break
            elif c1 + c2 >= self.TolMax:
                h_.error = 1
                break
            c1 = c2
            m += 1
        return h_

    def expm_adapt(self, h, y, step=1):
        h_ = copy.deepcopy(h)
        if step < self.MinStep:
            h_.error = 5
            return self.expm_onestep(h_, y, step)
        h1 = self.expm_onestep(h_, y, step)
        h2 = self.expm_onestep(self.expm_onestep(h_, y, 0.5 * step), y, 0.5 * step)
        if self.norm_int(h1.f - h2.f) < self.AbsTol + self.RelTol * self.norm_int(h1.f):
            h_.f = 0.75 * h1.f + 0.25 * h2.f
            return h_
        else:
            return self.expm_adapt(self.expm_adapt(h_, y, 0.5 * step), y, 0.5 * step)

    def rg_map(self, h):
        h_ = copy.deepcopy(h)
        h_.error = 0
        omega_ = (self.N.transpose()).dot(h_.Omega)
        ren = (2.0 * la.norm(omega_) / self.Eigenvalue * h_.f[2][self.zero_]) ** (2 - xp.arange(self.J+1, dtype=int)) / (2.0 * h_.f[2][self.zero_])
        h_.Omega = omega_ / la.norm(omega_)
        self.omega_nu = xp.einsum('i,i...->...', h_.Omega, self.nu).reshape(self.r_1l)
        f_ = xp.zeros_like(h_.f)
        f_[self.nu_mask] = h_.f[self.N_nu_mask].copy()
        h_.f = f_ * ren.reshape(self.r_j1)
        k_ = 0
        iminus_f = xp.zeros_like(h_.f)
        iminus_f[self.iminus] = h_.f[self.iminus].copy()
        while (self.TolMax > self.norm(iminus_f) > self.TolMin) and (self.TolMax > self.norm_int(h_.f) > self.TolMin) and (k_ < self.MaxLie):
            y = self.generate_y(h_)
            if self.CanonicalTransformation == 'expm_onestep':
                h_ = self.expm_onestep(h_, y)
            elif self.CanonicalTransformation == 'expm_adapt':
                h_ = self.expm_adapt(h_, y)
            elif self.CanonicalTransformation == 'expm_multiply':
                h_ = self.expm_multiply(h_, y)
            iminus_f[self.iminus] = h_.f[self.iminus].copy()
            k_ += 1
            h_.f = self.sym(h_.f)
        if (self.norm(iminus_f) > self.TolMax):
            h_.error = 2
        elif (k_ > self.MaxLie):
            h_.error = -2
        return h_

    def norm_int(self, fun):
        fun_ = fun.copy()
        fun_[xp.index_exp[:self.J+1] + self.zero_] = 0.0
        return self.norm(fun_)

    def sym(self, fun):
        fun_ = (fun + xp.roll(xp.flip(fun, axis=self.axis_dim), 1, axis=self.axis_dim).conj()) / 2.0
        fun_[0][self.zero_] = 0.0
        fun_[xp.abs(fun_) < self.TolMin**2] = 0.0
        return fun_

    class Hamiltonian:
        def __repr__(self):
            return '{self.__class__.name__}({self.omega, self.f, self.error, self.count})'.format(self=self)

        def __str__(self):
            return 'A {self.__class__.name__} in action-angle variables'.format(self=self)

        def __init__(self, omega, f, error=0, count=0):
            self.Omega = omega
            self.f = f
            self.error = error
            self.count = count

if __name__ == "__main__":
    main()
