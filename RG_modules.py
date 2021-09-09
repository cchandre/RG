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
    h_list = generate_2Hamiltonians(case, (case.AmpInf, case.AmpSup))
    if (h_list[0].error == 0) and (h_list[1].error == 0):
        timestr = time.strftime("%Y%m%d_%H%M")
        plot_fun(case, h_list[1].f[0])
        data, k_ = [], 0
        while (k_ < case.Iterates) and (h_list[0].error == 0) and (h_list[1].error == 0):
            k_ += 1
            start = time.time()
            h_list = approach(case, h_list, dist=case.DistSurf, strict=True)
            h_list_ = case.rg_map(h_list[0]), case.rg_map(h_list[1])
            if k_ == 1:
                print('\033[96m          Critical parameter = {:.6f} \033[00m'.format(2.0 * h_list[0].f[case.ModesK[0]]))
            plot_fun(case, h_list_[0].f[0])
            mean2_p = 2.0 * h_list_[0].f[2][case.zero_]
            diff_p = case.norm(xp.abs(h_list[0].f) - xp.abs(h_list_[0].f))
            delta_p = case.norm(xp.abs(h_list_[0].f) - xp.abs(h_list_[1].f)) / case.norm(h_list[0].f - h_list[1].f)
            data.append([diff_p, delta_p, mean2_p])
            h_list = copy.deepcopy(h_list_)
            print('\033[96m          diff = {:.3e}    delta = {:.7f}   <f2> = {:.7f}    (done in {:d} seconds) \033[00m'.format(diff_p, delta_p, mean2_p, int(xp.rint(time.time()-start))))
            plt.pause(0.5)
        save_data('iterates', data, timestr, case, info='diff     delta     <f2>', display=True)

def compute_cr(epsilon, case):
	[amp_inf_, amp_sup_] = [case.AmpInf, case.AmpSup].copy()
	amp_inf_[0] = case.AmpInf[0] + epsilon * (case.AmpSup[0] - case.AmpInf[0])
	amp_sup_[0] = amp_inf_[0].copy()
	h_list = generate_2Hamiltonians(case, (amp_inf_, amp_sup_))
	if converge(case, h_list[0]) and (not converge(case, h_list[1])):
		h_list = approach(case, h_list, dist=case.DistSurf)
		return [2.0 * h_list[0].f[_] for _ in case.ModesK]
	else:
		return [2.0 * h_list[0].f[case.ModesK[0]], xp.nan]

def compute_line(case):
    print('\033[92m    {} -- line \033[00m'.format(case.__str__()))
    amps = [case.CoordLine[_] * case.ModesLine * case.DirLine + (1 - case.ModesLine) * case.DirLine for _ in range(2)]
    h_list = generate_2Hamiltonians(case, amps)
    if converge(case, h_list[0]) and (not converge(case, h_list[1])):
        h_list = approach(case, h_list, dist=case.DistSurf, display=True)
    return h_list

def compute_surface(case):
    print('\033[92m    {} -- surface \033[00m'.format(case.__str__()))
    timestr = time.strftime("%Y%m%d_%H%M")
    epsilon = xp.linspace(0.0, 1.0, case.Nxy, dtype=case.Precision)
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
	h = generate_1Hamiltonian(case, amp, symmetric=True)
	return [int(converge(case, h)), h.count], h.error

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
            for result_data, result_info in tqdm(pool.imap(point_, iterable=x_vec), leave=False, desc='x'):
                data.append(result_data)
                info.append(result_info)
            save_data('region', data, timestr, case, info)
    else:
        for y in tqdm(y_vec, desc='y'):
            for x in tqdm(x_vec, leave=False, desc='x'):
                result_data, result_info = point(x, y, case)
                data.append(result_data)
                info.append(result_info)
            save_data('region', data, timestr, case, info)
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

def generate_1Hamiltonian(case, amps, symmetric=False):
    f_ = xp.zeros(case.r_jl, dtype=case.Precision)
    for (k, amp) in zip(case.K, amps):
        f_[k] = amp
    if symmetric:
        f_ = case.sym(f_)
    f_[2][case.zero_] = 0.5
    return case.Hamiltonian(case.Omega, f_)

def generate_2Hamiltonians(case, amps):
    h_list = [generate_1Hamiltonian(case, amps[_], symmetric=True) for _ in range(2)]
    if not converge(case, h_list[0]):
        h_list[0].error = 4
    if converge(case, h_list[1]):
        h_list[1].error = -4
    else:
        h_list[1].error = 0
    return h_list

def converge(case, h):
    h_ = copy.deepcopy(h)
    h_.error, it_conv = 0, 0
    while (case.TolMax > case.norm_int(h_.f) > case.TolMin) and (it_conv < case.MaxIterates):
        h_ = case.rg_map(h_)
        it_conv += 1
    if (case.norm_int(h_.f) < case.TolMin):
        h.count = - it_conv
        return True
    else:
        h.count = it_conv
        h.error = h_.error
        return False

def approach(case, h_list, dist, strict=False, display=False):
    h_list_ = copy.deepcopy(h_list)
    h_list_[0].error = 0
    h_mid = copy.deepcopy(h_list_[0])
    while case.norm_int(h_list_[0].f - h_list_[1].f) > dist:
        h_mid.f = (h_list_[0].f + h_list_[1].f) / 2.0
        if converge(case, h_mid):
            h_list_[0].f = h_mid.f.copy()
        else:
            h_list_[1].f = h_mid.f.copy()
        if display:
            print('\033[90m               [{:.6f}   {:.6f}] \033[00m'.format(2.0 * h_list_[0].f[case.ModesK[0]], 2.0 * h_list_[1].f[case.ModesK[0]]))
    if display:
        print('\033[96m          Critical parameters = {} \033[00m'.format([2.0 * h_list_[0].f[case.K[_]] for _ in range(len(case.K))]))
    if strict:
        h_mid.f = (h_list_[0].f + h_list_[1].f) / 2.0
        delta_ = dist / case.norm_int(h_list_[0].f - h_list_[1].f)
        for _ in range(2):
            h_list_[_].f = h_mid.f + (h_list_[_].f - h_mid.f) * delta_
    if converge(case, h_list_[1]):
        h_list_[1].error = 3
    else:
        h_list_[1].error = 0
    return h_list_

def save_data(name, data, timestr, case, info=[], display=False):
    if case.SaveData:
        mdic = case.DictParams.copy()
        del mdic['Precision']
        mdic.update({'data': data, 'info': info})
        date_today = date.today().strftime(" %B %d, %Y\n")
        mdic.update({'date': date_today, 'author': 'cristel.chandre@univ-amu.fr'})
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
