import numpy as xp
import tkinter as tk
from tkinter import ttk
import time
from itertools import chain
import multiprocessing
import RG_classes_functions as RG
import matplotlib.pyplot as plt
import copy
from functools import partial
from tqdm import tqdm, trange
import warnings
warnings.filterwarnings("ignore")

color_bg = '#9BB7D4'
font = 'Garamond 14 bold'
font_color = '#000080'
NumCores = multiprocessing.cpu_count()

def main():
	version_rgapp = '0.3'
	date_rgapp = time.strftime("%Y / %m / %d")

	rg_app = tk.Tk()
	rg_app.title("Renormalization Group for Hamiltonians")
	window_x = 680
	window_y = 400
	screen_width = rg_app.winfo_screenwidth()
	screen_height = rg_app.winfo_screenheight()
	position_x = (screen_width // 2) - (window_x // 2)
	position_y = (screen_height // 2) - (window_y // 2)
	geo = "{}x{}+{}+{}".format(window_x, window_y, position_x, position_y)
	rg_app.geometry(geo)
	rg_app.resizable(False, False)
	rg_app.configure(bg=color_bg)
	style = ttk.Style()
	style.theme_use('clam')
	style.configure('.', background=color_bg)
	style.configure('.', foreground=font_color)

	tab_parent = ttk.Notebook(rg_app)
	tab_param = ttk.Frame(tab_parent)
	tab_advanced = ttk.Frame(tab_parent)
	tab_run = ttk.Frame(tab_parent)
	tab_about = ttk.Frame(tab_parent)
	tab_parent.add(tab_run, text="Main")
	tab_parent.add(tab_param, text="Parameters")
	tab_parent.add(tab_advanced, text="Advanced")
	tab_parent.add(tab_about, text="About")
	tab_parent.pack(expand=1, fill='both')

	mp_names = 'L', 'J', 'Sigma', 'Kappa', 'NumberOfIterations', 'Ncs', 'TolCS'
	mp_types = 'Int', 'Int', 'Double', 'Double', 'Int', 'Int', 'Double'
	mp_values = 5, 5, 0.4, 0.1, 10, 100, 1e-7
	mp_positions = (1, 0), (2, 0), (4,0), (5, 0), (8, 0), (7, 5), (8, 5)

	mpc_names = 'SaveData', 'PlotResults'
	mpc_types = 'Bool', 'Bool'
	mpc_values = True, False
	mpc_positions = (7, 3), (6, 3)

	tol_names = 'TolMin', 'TolMax', 'TolLie', 'MaxIter', 'MaxLie', 'DistSurf'
	tol_types = 'Double', 'Double', 'Double', 'Int', 'Int', 'Double'
	tol_values = 1e-9, '{:1.0e}'.format(1e+9), 1e-11, 5000, 5000, 1e-7
	tol_positions = (1, 3), (2, 3), (5, 3), (3, 3), (6, 3), (8, 3)

	adv_names = 'ChoiceIm', 'CanonicalTransformation', 'NormChoice', 'Precision'
	adv_types = 'Char', 'Char', 'Char', 'Int'
	adv_values = 'AK2000', 'Lie', 'sum', 64
	adv_menus = ('AK2000', 'K1999', 'AKP1998'), ('Lie', 'Type2', 'Type3'), ('sum', 'max', 'Euclidian', 'Analytic'), (32, 64, 128)
	adv_positions = (1, 0), (5, 0), (3, 0), (8, 0)
	adv_commands = None, None, None, None

	adv2_names = 'MaxA', 'DistCircle', 'Radius', 'ModesPerturb', 'Nh'
	adv2_types = 'Double', 'Double', 'Double', 'Int', 'Int'
	adv2_values = 0.2, 1e-5, 1e-5, 3, 10
	adv2_positions = (5, 3), (1, 3), (2, 3), (3, 3), (4, 3)

	val_names = 'N', 'omega_0', 'Omega', 'K', 'KampInf', 'KampSup'
	val_types = 'Char', 'Char', 'Char', 'Char', 'Char', 'Char'
	val_positions = (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0)
	val_values = '[[1, 1], [1, 0]]', '[-0.618033988749895, 1.0]', '[1.0, 0.0]', '((0, 1, 0), (0, 1, 1))', '[0.0, 0.0]', '[0.04, 0.04]'

	run_etiqs = 'Iterates', 'Circle Iterates', 'Critical Surface', 'Converge Region'
	run_positions = (1, 3), (2, 3), (3, 3), (4, 3)
	run_method = tk.StringVar()
	run_method.set(run_etiqs[0])
	for (run_etiq, run_position) in zip(run_etiqs, run_positions):
		b_method = tk.Radiobutton(tab_run, variable=run_method, text=run_etiq, value=run_etiq, width=15, anchor='w', bg=color_bg)
		b_method.grid(row=run_position[0], column=run_position[1], sticky='w')

	case_options = ('GoldenMean', 'SpiralMean', 'TauMean', 'OMean', 'EtaMean')

	mp_params = definevar(tab_param, mp_types, mp_values)
	mp_par = makeform(tab_param, mp_params, mp_names, mp_positions, (14, 4))

	tk.Label(tab_param, width=10, text=None, bg=color_bg).grid(row=0, column=2)

	mpc_params = definevar(tab_run, mpc_types, mpc_values)
	mpc_par = makechecks(tab_run, mpc_params, mpc_names, mpc_positions)

	tol_params = definevar(tab_param, tol_types, tol_values)
	tol_par = makeform(tab_param, tol_params, tol_names, tol_positions, (8, 6))

	adv_params = definevar(tab_advanced, adv_types, adv_values)
	adv_par = makemenus(tab_advanced, adv_params, adv_names, adv_menus, adv_positions)

	tk.Label(tab_advanced, width=10, text=None, bg=color_bg).grid(row=0, column=2)

	adv2_params = definevar(tab_advanced, adv2_types, adv2_values)
	adv2_par = makeform(tab_advanced, adv2_params, adv2_names, adv2_positions, (12, 8))

	choice_ml = tk.Label(tab_run, width=20, text='Choose method:', anchor='w', bg=color_bg, pady=5, font=font, fg=font_color)
	choice_ml.grid(row=0, column=3, padx=5)
	case_lab = tk.Label(tab_run, width=20, text='Choose frequency vector:', anchor='w', bg=color_bg, pady=5, font=font, fg=font_color)
	case_var = tk.StringVar(tab_run, value=case_options[0])
	case_menu = tk.OptionMenu(tab_run, case_var, *case_options, command= lambda x: define_case(x, val_params))
	case_menu.grid(row=1, column=0, padx=5, sticky='w')
	case_lab.grid(row=0, column=0, padx=5)

	val_params = definevar(tab_run, val_types, val_values)
	val_par = makeform(tab_run, val_params, val_names, val_positions, (7, 20))

	tk.Label(tab_run, width=5, text=None, bg=color_bg).grid(row=0, column=2)

	run_button = tk.Button(tab_run, text='Run', highlightbackground=color_bg, width=18,\
	 	command= lambda : rg_run(run_method, case_var, mp_names, mp_par, mpc_names, mpc_par, adv_names, adv_par, adv2_names, adv2_par, tol_names, tol_par, val_params)).grid(row=8, column=3, sticky='w')
	#tk.Button(tab_run, text='Quit', command=rg_app.quit, bg=color_bg).grid(row=8, column=4)

	tk.Label(tab_about, width=10, text=None, bg=color_bg).grid(row=0, column=2)

	errorcode = tk.Text(tab_about, height=9, width=35, pady=10, bg=color_bg, font=font, fg=font_color)
	errorcode.insert(tk.INSERT, "ERROR CODES\n\n")
	errorcode.insert(tk.INSERT, "    k-th Lie transform diverging: [1, k]\n")
	errorcode.insert(tk.INSERT, "    k-th Lie transform not converging: [-1, k]\n")
	errorcode.insert(tk.INSERT, "    I- iterations diverging: [2, 0]\n")
	errorcode.insert(tk.INSERT, "    I- iterations not converging: [-2, 0]\n")
	errorcode.insert(tk.INSERT, "    below (approach): [3, 0]\n")
	errorcode.insert(tk.INSERT, "    above (generate_2Hamiltonians): [4, 0]\n")
	errorcode.insert(tk.END, "    below (generate_2Hamiltonians): [-4, 0]")
	errorcode.config(state=tk.DISABLED)
	errorcode.grid(row=1, column=0, sticky='n')

	author = tk.Text(tab_about, height=4, width=35, pady=10, bg=color_bg, font=font, fg=font_color)
	author.insert(tk.INSERT, "AUTHOR\n\n")
	author.insert(tk.INSERT, "     Cristel Chandre (I2M, CNRS)\n")
	author.insert(tk.END, "     cristel.chandre@univ-amu.fr ")
	author.config(state=tk.DISABLED)
	author.grid(row=2, column=0, sticky='s')

	version = tk.Text(tab_about, height=1, width=29, pady=10, padx=10, bg=color_bg, font=font, fg=font_color)
	version.insert(tk.INSERT, "VERSION    ")
	version.insert(tk.INSERT, version_rgapp )
	version.insert(tk.END, "   (" + date_rgapp + ")")
	version.config(state=tk.DISABLED)
	version.grid(row=2, column=2, sticky='s')

	rg_app.mainloop()

def definevar(root, types, values):
	paramlist = []
	for (type, value) in zip(types, values):
		if type == 'Double':
			tempvar = tk.DoubleVar(root, value=value)
		elif type == 'Int':
			tempvar = tk.IntVar(root, value=value)
		elif type == 'Char':
			tempvar = tk.StringVar(root, value=value)
		elif type == 'Bool':
			tempvar = tk.BooleanVar(root, value=value)
		paramlist.append(tempvar)
	return paramlist

def makeform(root, fields, names, positions, width):
	entries = []
	for (field, name, position) in zip(fields, names, positions):
		lab = tk.Label(root, width=width[0], text=name, anchor='e', bg=color_bg, font=font, fg=font_color)
		ent = tk.Entry(root, width=width[1], textvariable=field, bg=color_bg)
		lab.grid(row=position[0], column=position[1], pady=5)
		ent.grid(row=position[0], column=position[1]+1, pady=5)
		entries.append((field, ent))
	return entries

def makemenus(root, fields, names, menus, positions):
	entries = []
	for (field, name, menu, position) in zip(fields, names, menus, positions):
		lab = tk.Label(root, width=18, text=name, anchor='e', bg=color_bg, font=font, fg=font_color)
		men = tk.OptionMenu(root, field, * menu)
		lab.grid(row=position[0], column=position[1], pady=5, sticky='e')
		men.grid(row=position[0], column=position[1]+1, pady=5, padx=5, sticky='e')
		entries.append((field, men))
	return entries

def makechecks(root, fields, names, positions):
	entries = []
	for (field, name, position) in zip(fields, names, positions):
		chec = tk.Checkbutton(root, text=name, variable=field, onvalue=True, offvalue=False, bg=color_bg, font=font, fg=font_color)
		chec.grid(row=position[0], column=position[1], sticky='w')
		entries.append((field, chec))
	return entries

def rg_run(run_method, case_var, mp_names, mp_par, mpc_names, mpc_par, adv_names, adv_par, adv2_names, adv2_par, tol_names, tol_par, val_params):
	params = dict()
	for (name, entry) in chain(zip(mp_names, mp_par), zip(mpc_names, mpc_par),\
	 zip(adv_names, adv_par), zip(adv2_names, adv2_par), zip(tol_names, tol_par)):
		params[name] = entry[0].get()
	N = list(eval(val_params[0].get()))
	omega_0 = list(eval(val_params[1].get()))
	Omega = list(eval(val_params[2].get()))
	K = list(eval(val_params[3].get()))
	KampInf = list(eval(val_params[4].get()))
	KampSup = list(eval(val_params[5].get()))
	case_init = RG.CaseInit(N, omega_0, Omega, K, KampInf, KampSup)
	case_study = RG.RG(case_init, params)
	if run_method.get() == 'Iterates':
		iterates(case_study, case_init)
	elif run_method.get() == 'Circle Iterates':
		case_study.iterate_circle(case_init)
	elif run_method.get() == 'Critical Surface':
		critical_surface(case_study, case_init)
	elif run_method.get() == 'Converge Region':
		converge_region(case_study, case_init)

def define_case(case, val_params):
	if case == 'GoldenMean':
	    N = [[1, 1], [1, 0]]
	    Eigenvalues = [-0.618033988749895, 1.618033988749895]
	    omega_0 = (Eigenvalues[0], 1.0)
	    Omega = [1.0, 0.0]
	    K = ((0, 1, 0), (0, 1, 1))
	    KampInf = [0.02, 0.02]
	    KampSup = [0.04, 0.04]
	elif case == 'SpiralMean':
	    N = [[0, 0, 1], [1, 0, 0], [0, 1, -1]]
	    sigma = 1.3247179572447460259
	    #Eigenvalues = [1.0 / sigma]
	    omega_0 = (sigma**2, sigma, 1.0)
	    Omega = [1.0, 1.0, -1.0]
	    K = ((0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
	    KampInf = [0.034, 0.089, 0.1]
	    KampSup = [0.036, 0.091, 0.1]
	elif case == 'TauMean':
	    N = [[0, 1, -1], [1, -1, 1], [0, -1, 2]]
	    Tau = 0.445041867912629
	    Tau2 = 1.801937735804839
	    Tau3 = -1.246979603717467
	    #Eigenvalues = [Tau, Tau2, Tau3]
	    omega_0 = (1.0, Tau, 1.0 - Tau - Tau**2)
	    Omega = [1.0, 1.0, -1.0]
	    K = ((0, 0, -1, 1), (0, 1, -1, -1), (0, 0, 0, 1))
	    KampInf = [0.0, 0.0, 0.0]
	    KampSup = [0.1, 0.1, 0.1]
	elif case == 'OMean':
	    N = [[0, 0, 1], [1, 0, -1], [0, 1, 0]]
	    o_val = 0.682327803828019
	    #Eigenvalues = [o_val]
	    omega_0 = (1.0, o_val, o_val**2)
	    Omega = [1.0, 1.0, 1.0]
	    K = ((0, 1, -1, -1), (0, 0, 1, -1), (0, 1, -1, 0))
	    KampInf = [0.0, 0.0, 0.0]
	    KampSup = [0.1, 0.1, 0.1]
	elif case == 'EtaMean':
	    N = [[-1, 1, 0], [1, 1, 1], [0, 1, 0]]
	    Eta = -0.347296355333861
	    Eta2 = -1.532088886237956
	    Eta3 = 1.879385241571816
	    Eigenvalues = [Eta, Eta2, Eta3]
	    omega_0 = (Eta **2 - Eta - 1.0, Eta, 1.0)
	    Omega = [1.0, -1.0, 1.0]
	    K = ((0, 1, 1, 1), (0, -1, 1, 0), (0, 0, 1, 0))
	    KampInf = [0.0, 0.0, 0.0]
	    KampSup = [0.1, 0.1, 0.1]
	val_params[0].set(str(N))
	val_params[1].set(str(omega_0))
	val_params[2].set(str(Omega))
	val_params[3].set(str(K))
	val_params[4].set(str(KampInf))
	val_params[5].set(str(KampSup))

def iterates(case, case_init):
	h_inf, h_sup = case.generate_2Hamiltonians(case_init)
	if (h_inf.error == [0, 0]) and (h_sup.error == [0, 0]):
		timestr = time.strftime("%Y%m%d_%H%M")
		if case.PlotResults:
			case.plotf(h_sup.f[0])
		start = time.time()
		data = []
		k_ = 0
		while (k_ < case.NumberOfIterations) and (h_inf.error == [0, 0]) and (h_sup.error == [0, 0]):
			k_ += 1
			start_k = time.time()
			h_inf, h_sup = case.approach(h_inf, h_sup, dist=case.DistSurf, strict=True)
			h_inf_ = case.renormalization_group(h_inf)
			h_sup_ = case.renormalization_group(h_sup)
			if k_ == 1:
				print('Critical parameter = {}'.format(2.0 * h_inf.f[case_init.K[0]]))
			if case.PlotResults:
				case.plotf(h_inf_.f[0])
			mean2_p = 2.0 * h_inf.f[2][case.zero_]
			diff_p = case.norm(xp.abs(h_inf.f) - xp.abs(h_inf_.f))
			delta_p = case.norm(xp.abs(h_inf_.f) - xp.abs(h_sup_.f)) / case.norm(h_inf.f - h_sup.f)
			data.append([diff_p, delta_p, mean2_p])
			h_inf = copy.deepcopy(h_inf_)
			h_sup = copy.deepcopy(h_sup_)
			end_k = time.time()
			print("diff = %.3e    delta = %.7f   <f2> = %.7f    (done in %d seconds)" % \
					(diff_p, delta_p, mean2_p, int(xp.rint(end_k-start_k))))
			if case.SaveData:
				info = 'diff     delta     <f2>'
				save_data('RG_iterates', data, timestr, case.params, case_init, info=info)
		if (k_ < case.NumberOfIterations):
			print('Warning (iterates): ' + h_inf.error + ' / ' + h_sup.error)
		end = time.time()
		print("Computation done in {} seconds".format(int(xp.rint(end-start))))
		plt.show()
	else:
		print('Warning (iterates): ' + h_inf.error + ' / ' + h_sup.error)

def compute_cr(epsilon, case, case_init):
	k_inf_ = case_init.KampInf.copy()
	k_sup_ = case_init.KampSup.copy()
	k_inf_[0] = case_init.KampInf[0] + epsilon * (case_init.KampSup[0] - case_init.KampInf[0])
	k_sup_[0] = k_inf_[0]
	case_ = copy.deepcopy(case_init)
	case_.KampInf = k_inf_
	case_.KampSup = k_sup_
	h_inf, h_sup = case.generate_2Hamiltonians(case_)
	if case.converge(h_inf) and (not case.converge(h_sup)):
		h_inf, h_sup = case.approach(h_inf, h_sup, dist=case.TolCS)
		return 2.0 * xp.array([h_inf.f[case_init.K[0]], h_inf.f[case_init.K[1]]])
	else:
		return 2.0 * xp.array([h_inf.f[case_init.K[0]], xp.nan])

def critical_surface(case, case_init):
	timestr = time.strftime("%Y%m%d_%H%M")
	epsilon_ = xp.linspace(0.0, 1.0, case.Ncs)
	pool = multiprocessing.Pool(NumCores)
	data = []
	compfun = partial(compute_cr, case=case, case_init=case_init)
	for result in tqdm(pool.imap(compfun, iterable=epsilon_), total=len(epsilon_)):
		data.append(result)
	data = xp.array(data).transpose()
	if case.SaveData:
		save_data('RG_critical_surface', data, timestr, case.params, case_init)
	if case.PlotResults:
		fig = plt.figure()
		ax = fig.gca()
		ax.set_box_aspect(1)
		plt.plot(data[0, :], data[1, :], color='b', linewidth=2)
		ax.set_xlim(case_init.KampInf[0], case_init.KampSup[0])
		ax.set_ylim(case_init.KampInf[1], case_init.KampSup[1])
		plt.show()

def converge_point(val1, val2, case, case_init):
	k_amp_ = case_init.KampSup.copy()
	k_amp_[0] = val1
	k_amp_[1] = val2
	h_ = case.generate_1Hamiltonian(case_init.K, k_amp_, case_init.Omega, symmetric=True)
	return [int(case.converge(h_)), h_.count], h_.error

def converge_region(case, case_init):
	timestr = time.strftime("%Y%m%d_%H%M")
	x_vec = xp.linspace(case_init.KampInf[0], case_init.KampSup[0], case.Ncs)
	y_vec = xp.linspace(case_init.KampInf[1], case_init.KampSup[1], case.Ncs)
	pool = multiprocessing.Pool(NumCores)
	data = []
	info = []
	for y_ in tqdm(y_vec):
		converge_point_ = partial(converge_point, val2=y_, case=case, case_init=case_init)
		for result_data, result_info in tqdm(pool.imap(converge_point_, iterable=x_vec), total=case.Ncs, leave=False):
			data.append(result_data)
			info.append(result_info)
		if case.SaveData:
			save_data('RG_converge_region', data, timestr, case.params, case_init)
	if case.SaveData:
		save_data('RG_converge_region', xp.array(data).reshape((case.Ncs, case.Ncs, 2)), timestr, case.params, case_init, info=xp.array(info).reshape((case.Ncs, case.Ncs, 2)))
	if case.PlotResults:
		fig = plt.figure()
		ax = fig.gca()
		ax.set_box_aspect(1)
		im = ax.pcolor(x_vec, y_vec, xp.array(data)[:, 0].reshape((case.Ncs, case.Ncs)).astype(int), cmap='Reds_r')
		ax.set_xlim(case_init.KampInf[0], case_init.KampSup[0])
		ax.set_ylim(case_init.KampInf[1], case_init.KampSup[1])
		fig.colorbar(im)
		plt.show()

def approach_set(k, set1, set2, case, dist, strict):
	set1[k], set2[k] = case.approach(set1[k], set2[k], dist=dist, strict=strict)

def iterate_circle(case, case_init):
	start = time.time()
	h_inf, h_sup = case.generate_2Hamiltonians(case_init)
	h_inf, h_sup = case.approach(h_inf, h_sup, dist=case.DistSurf, strict=True)
	h_inf = case.renormalization_group(h_inf)
	h_sup = case.renormalization_group(h_sup)
	h_inf, h_sup = case.approach(h_inf, h_sup, dist=case.DistCircle, strict=True)
	if (h_inf.error == [0, 0]) and (h_sup.error == [0, 0]):
		print('starting circle')
		timestr = time.strftime("%Y%m%d_%H%M")
		hc_inf, hc_sup = case.approach(h_inf, h_sup, dist=case.DistSurf, strict=True)
		v1 = xp.zeros((case.J+1,) + case.dim * (2*case.L+1,), dtype=case.Precision)
		v2 = xp.zeros((case.J+1,) + case.dim * (2*case.L+1,), dtype=case.Precision)
		v1[0][case.dim * xp.index_exp[:case.ModesPerturb]] = 2.0 * xp.random.random(case.dim * (case.ModesPerturb,)) - 1.0
		v2[0][case.dim * xp.index_exp[:case.ModesPerturb]] = 2.0 * xp.random.random(case.dim * (case.ModesPerturb,)) - 1.0
		v1 = case.sym(v1)
		v2 = case.sym(v2)
		v2 = v2 - xp.vdot(v2, v1) * v1 / xp.vdot(v1, v1)
		v1 = case.Radius * v1 / xp.sqrt(xp.vdot(v1, v1))
		v2 = case.Radius * v2 / xp.sqrt(xp.vdot(v2, v2))
		circle_inf = []
		circle_sup = []
		for k_ in range(case.Nh+1):
			h_inf_ = copy.deepcopy(h_inf)
			h_sup_ = copy.deepcopy(h_sup)
			theta = case.Precision(k_) * 2.0 * xp.pi / case.Precision(case.Nh)
			h_inf_.f = h_inf.f + v1 * xp.cos(theta) + v2 * xp.sin(theta)
			h_sup_.f = h_sup.f + v1 * xp.cos(theta) + v2 * xp.sin(theta)
			circle_inf.append(h_inf_)
			circle_sup.append(h_sup_)
		pool = multiprocessing.Pool(NumCores)
		approach_circle = partial(approach_set, set1=circle_inf, set2=circle_sup, case=case, dist=case.DistSurf, strict=True)
		pool.imap(approach_circle, iterable=range(case.Nh+1))
		Coord = xp.zeros((case.Nh+1, 2, case.NumberOfIterations))
		for i_ in trange(case.NumberOfIterations):
			for k_ in range(case.Nh+1):
				Coord[k_, :, i_] = [xp.vdot(circle_inf[k_].f - hc_inf.f, v1), xp.vdot(circle_inf[k_].f - hc_inf.f, v2)]
			if case.SaveData:
				save_data('RG_circle', Coord / case.Radius ** 2, timestr, case.params, case_init)
			if case.PlotResults:
				fig = plt.figure()
				ax = fig.add_subplot(111)
				ax.plot(Coord[:, 0, i_] / case.Radius ** 2, Coord[:, 1, i_] / case.Radius ** 2, label='%d -th iterate' % i_)
				ax.legend()
				plt.pause(1e-17)
			renfunc = partial(case.renormalization_group)
			circle_inf = pool.imap(renfunc, iterable=circle_inf)
			circle_sup = pool.imap(renfunc, iterable=circle_sup)
			approach_circle = partial(approach_set, set1=circle_inf, set2=circle_sup, case=case, dist=case.DistSurf, strict=True)
			pool.imap(approach_circle, iterable=range(case.Nh+1))
			hc_inf = case.renormalization_group(hc_inf)
			hc_sup = case.renormalization_group(hc_sup)
			hc_inf, hc_sup = case.approach(hc_inf, hc_sup, dist=case.DistSurf, strict=True)
		end = time.time()
		print("Computation done in {} seconds".format(int(xp.rint(end-start))))
		plt.show()
	else:
		print('Warning (iterate_circle): ' + h_inf.error + ' / ' + h_sup.error)

def save_data(name, data, timestr, params, case_init, info=[]):
    mdic = params.copy()
    mdic.update({'case': case_init})
    mdic.update({'data': data})
    mdic.update({'info': info})
    today = date.today()
    date_today = today.strftime(" %B %d, %Y\n")
    email = ' cristel.chandre@univ-amu.fr'
    mdic.update({'date': date_today, 'author': email})
    savemat(name + '_' + timestr + '.mat', mdic)

if __name__ == "__main__":
	main()
