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
import scipy.signal as sps
import copy
import warnings
from RG_modules import compute_iterates, compute_surface, compute_region, compute_line
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
        self.vecjl = xp.prod(xp.asarray(self.r_jl))
        self.conv_dim = xp.index_exp[:self.J+1] + self.dim * xp.index_exp[self.L:3*self.L+1]
        indx = self.dim * (xp.hstack((xp.arange(0, self.L+1), xp.arange(-self.L, 0))),)
        self.nu = xp.meshgrid(*indx, indexing='ij')
        eigenval, w_eig = la.eig(self.N.transpose())
        self.Eigenvalue = xp.real(eigenval[xp.abs(eigenval) < 1])
        N_nu = xp.sign(self.Eigenvalue).astype(int) * xp.einsum('ij,j...->i...', self.N, self.nu)
        self.omega0_nu = xp.einsum('i,i...->...', self.omega0, self.nu).reshape(self.r_1l)
        self.Omega_nu = xp.einsum('i,i...->...', self.Omega, self.nu).reshape(self.r_1l)
        mask = xp.prod(abs(N_nu) <= self.L, axis=0, dtype=bool)
        norm_nu = self.Precision(la.norm(self.nu, axis=0)).reshape(self.r_1l)
        self.J_ = xp.arange(self.J+1, dtype=self.Precision).reshape(self.r_j1)
        self.derivs = lambda f: [xp.roll(f * self.J_, -1, axis=0), self.Omega_nu * f]
        if self.ChoiceIm == 'AKW1998':
            comp_im = self.Sigma * xp.repeat(norm_nu, self.J+1, axis=0) + self.Kappa * self.J_
        elif self.ChoiceIm =='K1999':
            comp_im = xp.maximum(self.Sigma * xp.repeat(norm_nu, self.J+1, axis=0), self.Kappa * self.J_)
        else:
            w_nu = xp.einsum('ij,j...->i...', w_eig, self.nu)
            norm_w_nu = la.norm(w_nu, axis=0).reshape(self.r_1l)
            comp_im = self.Sigma * xp.repeat(norm_w_nu, self.J+1, axis=0) + self.Kappa * self.J_
        omega0_nu_ = xp.repeat(xp.abs(self.omega0_nu), self.J+1, axis=0) / la.norm(self.omega0)
        self.Iminus = omega0_nu_ > comp_im
        self.nu_mask = xp.index_exp[:self.J+1]
        self.N_nu_mask = xp.index_exp[:self.J+1]
        for _ in range(self.dim):
            self.nu_mask += (self.nu[_][mask],)
            self.N_nu_mask += (N_nu[_][mask],)
        self.norm = {
            'sum': lambda _: xp.abs(_).sum(),
            'max': lambda _: xp.abs(_).max(),
            'Euclidean': lambda _: xp.sqrt((xp.abs(_)**2).sum()),
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
        y = xp.zeros_like(h.f)
        y[0][self.Iminus[0]] = h.f[0][self.Iminus[0]] / self.omega0_nu[0][self.Iminus[0]]
        for m in range(1, self.J+1):
            y[m][self.Iminus[m]] = (h.f[m][self.Iminus[m]] - 2 * h.f[2][self.zero_] * self.Omega_nu[0][self.Iminus[m]] * y[m-1][self.Iminus[m]]) / self.omega0_nu[0][self.Iminus[m]]
        return y, -h.f[1][self.zero_] / (2 * h.f[2][self.zero_]), xp.roll(y * self.J_, -1, axis=0), self.Omega_nu * y, -self.omega0_nu * y

    def liouville(self, y, f):
        f_ = self.derivs(f.reshape(self.r_jl))
        return y[1] * f_[0] + self.conv_product(y[2], f_[1]) - self.conv_product(y[3], f_[0])

    def pnorm(self, y, p=1):
        Ls = lambda f: self.liouville(y, f).flatten()
        Ls_H = lambda f: -Ls(f)
        A = sla.LinearOperator(shape=(self.vecjl, self.vecjl), matvec=Ls, rmatvec=Ls_H, dtype=self.Precision)
        return sla.onenormest(A**p)**(1/p)

    def compute_p_max(self, m_max):
        sqrt_m_max = xp.sqrt(m_max)
        p_low, p_high = int(xp.floor(sqrt_m_max)), int(xp.ceil(sqrt_m_max + 1))
        return max(p for p in range(p_low, p_high +1) if p * (p - 1) <= m_max + 1)

    def compute_m_s(self, y, n0=1, tol=2**-53, m_max=55, ell=2):
        best_m, best_s = None, None
        norm_ls = self.pnorm(y)
        p_max = self.compute_p_max(m_max)
        if norm_ls < 2 * ell * p_max * (p_max + 3) * self.theta[m_max] / float(n0 * m_max):
            for m, theta in self.theta.items():
                s = int(xp.ceil(norm_ls / theta))
                if best_m is None or m * s < best_m * best_s:
                    best_m, best_s = m, s
        else:
            norm_d = xp.array([self.pnorm(y, p) for p in range(2, p_max + 2)])
            norm_alpha = xp.maximum(norm_d, xp.roll(norm_d, -1))
            for p in range(2, p_max + 1):
                for m in range(p * (p - 1) - 1, m_max + 1):
                    if m in self.theta:
                        s = int(xp.ceil(norm_alpha[p - 2] / self.theta[m]))
                        if best_m is None or m * s < best_m * best_s:
                            best_m, best_s = m, s
            best_s = max(best_s, 1)
        return best_m, best_s

    def expm_multiply(self, h, y, tol=2**-53):
        h_ = copy.deepcopy(h)
        h_.error = 0
        m_star, scale = self.compute_m_s(y)
        y_ = [item / scale for item in y]
        for s in range(scale):
            sh = y_[4] + self.liouville(y_, h_.f)
            h_.f += sh
            c1 = self.norm(sh)
            for m in range(2, m_star+1):
                sh = self.liouville(y_, sh) / self.Precision(m)
                c2 = self.norm(sh)
                h_.f += sh
                if c1 + c2 < tol * self.norm(h_.f):
                    break
                elif c1 + c2 > self.TolMax:
                    h_.error = 1
                    break
                c1 = c2
        return h_

    def expm_onestep(self, h, y, step=1, tol=2**-53):
        h_ = copy.deepcopy(h)
        h_.error = 0
        y_ = [item * step for item in y]
        sh = y_[4] + self.liouville(y_, h_.f)
        h_.f += sh
        m = 2
        c1 = self.norm(sh)
        while True:
            sh = self.liouville(y_, sh) / self.Precision(m)
            c2 = self.norm(sh)
            h_.f += sh
            if c1 + c2 < tol * self.norm(h_.f):
                break
            elif c1 + c2 > self.TolMax:
                h_.error = 1
                break
            c1 = c2
            m += 1
        return h_

    def expm_adapt(self, h, y, step=1.0):
        h_ = copy.deepcopy(h)
        if step < self.MinStep:
            h_.error = 4
            return self.expm_onestep(h_, y, step)
        h1 = self.expm_onestep(h_, y, step)
        h2 = self.expm_onestep(self.expm_onestep(h_, y, 0.5 * step), y, 0.5 * step)
        if self.norm(h1.f - h2.f) < self.AbsTol + self.RelTol * self.norm(h1.f):
            h_.f = 0.75 * h1.f + 0.25 * h2.f
            return h_
        else:
            return self.expm_adapt(self.expm_adapt(h_, y, 0.5 * step), y, 0.5 * step)

    def rg_map(self, h):
        h_ = copy.deepcopy(h)
        h_.error = 0
        Omega_ = (self.N.transpose()).dot(h_.Omega)
        ren = (2 * la.norm(Omega_) / self.Eigenvalue * h_.f[2][self.zero_])**(2 - xp.arange(self.J+1, dtype=int)) / (2 * h_.f[2][self.zero_])
        h_.Omega = Omega_ / la.norm(Omega_)
        self.Omega_nu = xp.einsum('i,i...->...', h_.Omega, self.nu).reshape(self.r_1l)
        f_ = xp.zeros_like(h_.f)
        f_[self.nu_mask] = h_.f[self.N_nu_mask]
        h_.f = f_ * ren.reshape(self.r_j1)
        k, Iminus_f = 0, xp.zeros_like(h_.f)
        Iminus_f[self.Iminus] = h_.f[self.Iminus]
        while self.TolMax > self.norm(Iminus_f) > self.TolMin and k < self.MaxLie:
            h_ = self.sym(eval(self.CanonicalTransformation)(h_, self.generate_y(h_)))
            Iminus_f[self.Iminus] = h_.f[self.Iminus]
            k += 1
        if self.norm(Iminus_f) > self.TolMax:
            h_.error = 2
        elif k >= self.MaxLie:
            warnings.warn('Maximum number of Lie transforms reached (MaxLie)')
            h_.error = -2
        return h_

    def norm_int(self, h):
        f = h.f.copy()
        f[xp.index_exp[:self.J+1] + self.zero_] = 0.0
        return self.norm(f)

    def sym(self, h, tol=2**-53):
        h_ = copy.deepcopy(h)
        h_.f = (h.f + xp.roll(xp.flip(h.f, axis=self.axis_dim), 1, axis=self.axis_dim).conj()) / 2
        h_.f[0][self.zero_] = 0.0
        h_.f[xp.abs(h_.f) < tol] = 0.0
        return h_

    def generate_1Hamiltonian(self, amps, symmetric=False):
        f_ = xp.zeros(self.r_jl, dtype=self.Precision)
        for k, amp in zip(self.K, amps):
            f_[k] = amp
        h = Hamiltonian(self.Omega, f_)
        h.f[2][self.zero_] = 0.5
        if symmetric:
            return self.sym(h)
        else:
            return h

    def generate_2Hamiltonians(self, amps):
        h_list = [self.generate_1Hamiltonian(amp, symmetric=True) for amp in amps]
        if not self.converge(h_list[0]):
            warnings.warn('Iterates of H\u2081 do not converge')
            h_list[0].error = 3
        if self.converge(h_list[1]):
            warnings.warn('Iterates of H\u2082 do not diverge')
            h_list[1].error = -3
        else:
            h_list[1].error = 0
        return h_list

    def converge(self, h):
        h_ = copy.deepcopy(h)
        k, h_.error = 0, 0
        while self.TolMax > self.norm_int(h_) > self.TolMin:
            h_ = self.rg_map(h_)
            k += 1
        if self.norm_int(h_) < self.TolMin:
            h.count = -k
            return True
        else:
            h.count = k
            h.error = h_.error
            return False

    def approach(self, h_list, reldist, display=False):
        h_list_ = copy.deepcopy(h_list)
        for h in h_list_:
            h.error = 0
        while self.norm_int(h_list_[0] - h_list_[1]) > reldist * self.norm(h_list_[0].f):
            h_mid = (h_list_[0] + h_list_[1]) * 0.5
            if self.converge(h_mid):
                h_list_[0] = copy.deepcopy(h_mid)
            else:
                h_list_[1] = copy.deepcopy(h_mid)
            if display:
                print('\033[90m               [{:.6f}   {:.6f}] \033[00m'.format(2 * h_list_[0].f[self.ModesK[0]], 2 * h_list_[1].f[self.ModesK[0]]))
        if display:
            print('\033[96m          Critical parameters = {} \033[00m'.format([2 * h_list_[0].f[k] for k in self.K]))
        h_list_[1].error = 0
        return h_list_

class Hamiltonian:
    def __repr__(self):
        return '{self.__class__.name__}({self.Omega, self.f, self.error, self.count})'.format(self=self)

    def __str__(self):
        return 'A {self.__class__.name__} in action-angle variables'.format(self=self)

    def __init__(self, Omega, f, error=0, count=0):
        self.Omega = Omega
        self.f = f
        self.error = error
        self.count = count

    def __add__(self, other):
        return Hamiltonian(self.Omega, self.f + other.f, error=self.error, count=self.count)

    def __sub__(self, other):
        return Hamiltonian(self.Omega, self.f - other.f, error=self.error, count=self.count)

    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return Hamiltonian(self.Omega, other * self.f, error=self.error, count=self.count)
        else:
            raise TypeError('multiplication only defined with a float or a int')

if __name__ == "__main__":
    main()
