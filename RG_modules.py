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
    h_inf, h_sup = case.generate_2Hamiltonians((case.AmpInf, case.AmpSup))
    if (h_inf.error == 0) and (h_sup.error == 0):
        timestr = time.strftime("%Y%m%d_%H%M")
        plot_fun(h_sup.f[0], case)
        data, k_ = [], 0
        while (k_ < case.Iterates) and (h_inf.error == 0) and (h_sup.error == 0):
            k_ += 1
            start = time.time()
            h_inf, h_sup = case.approach((h_inf, h_sup), dist=case.DistSurf, strict=True)
            h_inf_, h_sup_ = case.rg_map(h_inf), case.rg_map(h_sup)
            if k_ == 1:
                print('\033[96m          Critical parameter = {} \033[00m'.format(2.0 * h_inf.f[case.K[0]]))
            plot_fun(h_inf_.f[0], case)
            mean2_p = 2.0 * h_inf.f[2][case.zero_]
            diff_p = case.norm(xp.abs(h_inf.f) - xp.abs(h_inf_.f))
            delta_p = case.norm(xp.abs(h_inf_.f) - xp.abs(h_sup_.f)) / case.norm(h_inf.f - h_sup.f)
            data.append([diff_p, delta_p, mean2_p])
            h_inf, h_sup = copy.deepcopy((h_inf_, h_sup_))
            print('\033[96m          diff = {:.3e}    delta = {:.7f}   <f2> = {:.7f}    (done in {:d} seconds) \033[00m'.format(diff_p, delta_p, mean2_p, int(xp.rint(time.time()-start))))
            save_data('iterates', data, timestr, case, info='diff     delta     <f2>')
            plt.pause(0.5)

def compute_cr(epsilon, case):
	[amp_inf_, amp_sup_] = [case.AmpInf, case.AmpSup].copy()
	amp_inf_[0] = case.AmpInf[0] + epsilon * (case.AmpSup[0] - case.AmpInf[0])
	amp_sup_[0] = amp_inf_[0].copy()
	case_ = copy.deepcopy(case)
	h_inf, h_sup = case_.generate_2Hamiltonians((amp_inf_, amp_sup_))
	if case.converge(h_inf) and (not case.converge(h_sup)):
		h_inf, h_sup = case.approach((h_inf, h_sup), dist=case.DistSurf)
		return [2.0 * h_inf.f[case.K[0]], 2.0 * h_inf.f[case.K[1]]]
	else:
		return [2.0 * h_inf.f[case.K[0]], xp.nan]

def compute_surface(case):
    print('\033[92m    {} -- surface \033[00m'.format(case.__str__()))
    timestr = time.strftime("%Y%m%d_%H%M")
    epsilon_ = xp.linspace(0.0, 1.0, case.Nxy, dtype=case.Precision)
    data = []
    if case.Parallelization[0]:
        if case.Parallelization[1] == 'all':
            num_cores = multiprocess.cpu_count()
        else:
            num_cores = min(multiprocess.cpu_count(), case.Parallelization[1])
        pool = multiprocess.Pool(num_cores)
        compfun = lambda epsilon: compute_cr(epsilon, case=case)
        for result in tqdm(pool.imap(compfun, iterable=epsilon_)):
            data.append(result)
    else:
        for epsilon in tqdm(epsilon_):
            result = compute_cr(epsilon, case)
            data.append(result)
    data = xp.array(data).transpose()
    save_data('surface', data, timestr, case)
    if case.PlotResults:
        fig, ax = plt.subplots(1, 1)
        ax.set_box_aspect(1)
        ax.plot(data[0, :], data[1, :], color='b', linewidth=2)
        ax.set_xlim(case.AmpInf[0], case.AmpSup[0])
        ax.set_ylim(case.AmpInf[1], case.AmpSup[1])
        ax.set_xlabel('$\epsilon_1$')
        ax.set_ylabel('$\epsilon_2$')

def point(x, y, case):
	amp_ = case.AmpSup.copy()
	amp_[0:2] = [x, y]
	h_ = case.generate_1Hamiltonian(case.K, amp_, case.Omega, symmetric=True)
	return [int(case.converge(h_)), h_.count], h_.error

def compute_region(case):
    print('\033[92m    {} -- region \033[00m'.format(case.__str__()))
    timestr = time.strftime("%Y%m%d_%H%M")
    x_vec = xp.linspace(case.AmpInf[0], case.AmpSup[0], case.Nxy, dtype=case.Precision)
    y_vec = xp.linspace(case.AmpInf[1], case.AmpSup[1], case.Nxy, dtype=case.Precision)
    data, info = [], []
    if case.Parallelization[0]:
        if case.Parallelization[1] == 'all':
            num_cores = multiprocess.cpu_count()
        else:
            num_cores = min(multiprocess.cpu_count(), case.Parallelization[1])
        pool = multiprocess.Pool(num_cores)
        for y_ in tqdm(y_vec, desc='y'):
            point_ = lambda x_: point(x_, y=y_, case=case)
            for result_data, result_info in tqdm(pool.imap(point_, iterable=x_vec), leave=False, desc='x'):
                data.append(result_data)
                info.append(result_info)
            save_data('region', data, timestr, case, info)
    else:
        for y_ in tqdm(y_vec, desc='y'):
            for x_ in tqdm(x_vec, leave=False, desc='x'):
                result_data, result_info = point(x_, y_, case)
                data.append(result_data)
                info.append(result_info)
            save_data('region', data, timestr, case, info)
    save_data('region', xp.array(data).reshape((case.Nxy, case.Nxy, 2)), timestr, case, info=xp.array(info).reshape((case.Nxy, case.Nxy)))
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

def save_data(name, data, timestr, case, info=[]):
    if case.SaveData:
        mdic = case.DictParams.copy()
        mdic.update({'data': data, 'info': info})
        date_today = date.today().strftime(" %B %d, %Y\n")
        mdic.update({'date': date_today, 'author': 'cristel.chandre@univ-amu.fr'})
        name_file = type(case).__name__ + '_' + name + '_' + timestr + '.mat'
        savemat(name_file, mdic)
        print('\033[90m        Results saved in {} \033[00m'.format(name_file))

def plot_fun(fun, case):
    if case.dim == 2 and case.PlotResults:
        fig, ax = plt.subplots(1,1)
        color_map = 'hot_r'
        im = ax.imshow(xp.abs(xp.roll(fun, (case.L, case.L), axis=(0,1))).transpose(), origin='lower', extent=[-case.L, case.L, -case.L, case.L], norm=colors.LogNorm(vmin=case.TolMin, vmax=xp.abs(fun).max()), cmap=color_map)
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
