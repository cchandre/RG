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
import multiprocess
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import copy
import time
from datetime import date
from scipy.io import savemat
from tqdm import tqdm

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.sans-serif': ['Palatino'],
    'font.size': 24,
    'axes.labelsize': 30,
    'figure.figsize': [8, 8],
    'image.cmap': 'bwr'})

def compute_iterates(case):
    print('\033[92m    {} -- iterates \033[00m'.format(case.__str__()))
    h_list = case.generate_2Hamiltonians((case.AmpInf, case.AmpSup))
    if h_list[0].error == 0 and h_list[1].error == 0:
        timestr = time.strftime("%Y%m%d_%H%M")
        plot_fun(case, h_list[1].f[0])
        k_, data = 0, []
        while k_ < case.Iterates and h_list[0].error == 0:
            k_ += 1
            start = time.time()
            h_list = case.approach(h_list, reldist=case.RelDist)
            h_list_ = [case.rg_map(h_list[0]), case.rg_map(h_list[1])]
            if k_ == 1:
                print('\033[96m          Critical parameter = {:.6f} \033[00m'.format(2.0 * h_list[0].f[case.ModesK[0]]))
            plot_fun(case, h_list_[0].f[0])
            mean2_p = 2 * h_list_[0].f[2][case.zero_]
            diff_p = case.norm(xp.abs(h_list[0].f) - xp.abs(h_list_[0].f))
            delta_p = case.norm(xp.abs(h_list_[0].f) - xp.abs(h_list_[1].f)) / case.norm(h_list[0].f - h_list[1].f)
            data.append([diff_p, delta_p, mean2_p])
            h_list = copy.deepcopy(h_list_)
            print('\033[96m          diff = {:.3e}    delta = {:.7f}   <f\u2082> = {:.7f}    (done in {:d} seconds) \033[00m'.format(diff_p, delta_p, mean2_p, int(time.time()-start)))
            plt.pause(0.5)
        save_data('iterates', data, timestr, case, info='diff     delta     <f\u2082>', display=True)

def compute_cr(epsilon, case):
    [amp_inf, amp_sup] = [case.AmpInf, case.AmpSup].copy()
    amp_inf[0] = case.AmpInf[0] + epsilon * (case.AmpSup[0] - case.AmpInf[0])
    amp_sup[0] = amp_inf[0].copy()
    h_list = case.generate_2Hamiltonians((amp_inf, amp_sup))
    if case.converge(h_list[0]) and (not case.converge(h_list[1])):
        h_list = case.approach(h_list, reldist=case.RelDist)
        return [2 * h_list[0].f[_] for _ in case.ModesK]
    else:
        return [2 * h_list[0].f[case.ModesK[0]], xp.nan]

def compute_line(case):
    print('\033[92m    {} -- line \033[00m'.format(case.__str__()))
    amps = tuple(coord * case.ModesLine * case.DirLine + (1 - case.ModesLine) * case.DirLine for coord in case.CoordLine)
    h_list = case.generate_2Hamiltonians(amps)
    if case.converge(h_list[0]) and (not case.converge(h_list[1])):
        h_list = case.approach(h_list, reldist=case.RelDist, display=True)
    return h_list

def compute_surface(case):
    print('\033[92m    {} -- surface \033[00m'.format(case.__str__()))
    timestr = time.strftime("%Y%m%d_%H%M")
    epsilon = xp.linspace(0, 1, case.Nxy, dtype=case.Precision)
    data = []
    if case.Parallelization[0]:
        if case.Parallelization[1] == 'all':
            num_cores = multiprocess.cpu_count()
        else:
            num_cores = min(multiprocess.cpu_count(), case.Parallelization[1])
        pool = multiprocess.Pool(num_cores)
        compfun = lambda _: compute_cr(_, case=case)
        for result in tqdm(pool.imap(compfun, iterable=epsilon)):
            data.append(result)
    else:
        for _ in tqdm(epsilon):
            result = compute_cr(_, case)
            data.append(result)
    data = xp.array(data).transpose()
    save_data('surface', data, timestr, case, display=True)
    if case.PlotResults:
        fig, ax = plt.subplots(1, 1)
        ax.set_box_aspect(1)
        ax.plot(data[0, :], data[1, :], color='b', linewidth=2)
        ax.set_xlim(case.AmpInf[0], case.AmpSup[0])
        ax.set_ylim(case.AmpInf[1], case.AmpSup[1])
        ax.set_xlabel('$\epsilon_1$')
        ax.set_ylabel('$\epsilon_2$')

def point(x, y, case):
    amp = case.AmpSup.copy()
    amp[0:2] = [x, y]
    h = case.generate_1Hamiltonian(amp, symmetric=True)
    return [int(case.converge(h)), h.count], h.error

def compute_region(case):
    print('\033[92m    {} -- region \033[00m'.format(case.__str__()))
    timestr = time.strftime("%Y%m%d_%H%M")
    x_vec, y_vec = (xp.linspace(case.AmpInf[_], case.AmpSup[_], case.Nxy, dtype=case.Precision) for _ in range(2))
    data, info = [], []
    if case.Parallelization[0]:
        if case.Parallelization[1] == 'all':
            num_cores = multiprocess.cpu_count()
        else:
            num_cores = min(multiprocess.cpu_count(), case.Parallelization[1])
        pool = multiprocess.Pool(num_cores)
        for y in tqdm(y_vec, desc='y'):
            point_ = lambda _: point(_, y=y, case=case)
            for result_data, result_info in tqdm(pool.imap(point_, iterable=x_vec), leave=False, desc='x', total=len(x_vec)):
                data.append(result_data)
                info.append(result_info)
            save_data('region', xp.array(data).reshape((-1, case.Nxy, 2)), timestr, case, xp.array(info).reshape((-1, case.Nxy)))
    else:
        for y in tqdm(y_vec, desc='y'):
            for x in tqdm(x_vec, leave=False, desc='x'):
                result_data, result_info = point(x, y, case)
                data.append(result_data)
                info.append(result_info)
            save_data('region', xp.array(data).reshape((-1, case.Nxy, 2)), timestr, case, xp.array(info).reshape((-1, case.Nxy)))
    save_data('region', xp.array(data).reshape((case.Nxy, case.Nxy, 2)), timestr, case, info=xp.array(info).reshape((case.Nxy, case.Nxy)), display=True)
    if case.PlotResults:
        divnorm = colors.TwoSlopeNorm(vmin=min(xp.array(data)[:, 1]), vcenter=0.0, vmax=max(xp.array(data)[:, 1]))
        fig, ax = plt.subplots(1, 1)
        ax.set_box_aspect(1)
        im = ax.pcolormesh(x_vec, y_vec, xp.array(data)[:, 1].reshape((case.Nxy, case.Nxy)).astype(int), norm=divnorm)
        ax.set_xlim(case.AmpInf[0], case.AmpSup[0])
        ax.set_ylim(case.AmpInf[1], case.AmpSup[1])
        ax.set_xlabel('$\epsilon_1$')
        ax.set_ylabel('$\epsilon_2$')
        fig.colorbar(im)

def save_data(name, data, timestr, case, info=[], display=False):
    if case.SaveData:
        mdic = case.DictParams.copy()
        del mdic['Precision']
        mdic.update({'data': data, 'info': info})
        date_today = date.today().strftime(" %B %d, %Y\n")
        mdic.update({'date': date_today, 'author': 'cristel.chandre@cnrs.fr'})
        name_file = type(case).__name__ + '_' + name + '_' + timestr + '.mat'
        savemat(name_file, mdic)
        if display:
            print('\033[90m        Results saved in {} \033[00m'.format(name_file))

def plot_fun(case, fun):
    if case.dim == 2 and case.PlotResults:
        fig, ax = plt.subplots(1,1)
        im = ax.imshow(xp.abs(xp.roll(fun, (case.L, case.L), axis=(0,1))).transpose(), origin='lower', extent=[-case.L, case.L, -case.L, case.L], norm=colors.LogNorm(vmin=case.TolMin, vmax=xp.abs(fun).max()), cmap='hot_r')
        fig.colorbar(im, orientation='vertical')
        ax.set_xlim(-case.L, case.L)
        ax.set_ylim(-case.L, case.L)
        ax.set_xlabel('$k_1$')
        ax.set_ylabel('$k_2$')
    elif case.dim == 3 and case.PlotResults:
        Y, Z = xp.meshgrid(xp.arange(-case.L, case.L+1), xp.arange(-case.L, case.L+1))
        X = xp.zeros_like(Y)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_box_aspect((5/3, 1/3, 1/3))
        norm_c = colors.LogNorm(vmin=case.TolMin, vmax=xp.abs(fun).max())
        for k_ in range(-case.L, case.L+1):
            A = xp.abs(xp.roll(fun[k_, :, :], (case.L,case.L), axis=(0,1)))
            ax.plot_surface(X + k_, Y, Z, rstride=1, cstride=1, facecolors=cm.hot(norm_c(A)), alpha=0.4, linewidth=0.0, shade=False)
        ax.set_xticks((-case.L,0,case.L))
        ax.set_yticks((-case.L,0,case.L))
        ax.set_zticks((-case.L,0,case.L))
        ax.set_xlim((-case.L-1/2, case.L+1/2))
        ax.set_ylim((-case.L-1/2, case.L+1/2))
        ax.set_zlim((-case.L-1/2, case.L+1/2))
        ax.view_init(elev=20, azim=120)
    plt.pause(0.5)
