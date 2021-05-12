import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
import numpy as xp
from itertools import chain
import multiprocess
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import copy
import time
from datetime import date
from scipy.io import savemat
import RG_classes_functions as RG

color_bg = '#9BB7D4'
font = 'Garamond 14 bold'
font_color = '#000080'
NumCores = multiprocess.cpu_count()

version_rgapp = '1.1'
date_rgapp = time.strftime("%Y / %m / %d")

def main():
	rg_app = tk.Tk()
	style = ttk.Style()
	style.theme_use('clam')
	style.configure('.', background=color_bg)
	style.configure('.', foreground=font_color)
	rg_app.title("Renormalization Group for Hamiltonians")
	window_x = 680
	window_y = 450
	screen_width = rg_app.winfo_screenwidth()
	screen_height = rg_app.winfo_screenheight()
	position_x = (screen_width // 2) - (window_x // 2)
	position_y = (screen_height // 2) - (window_y // 2)
	rg_app.geometry("{}x{}+{}+{}".format(window_x, window_y, position_x, position_y))
	rg_app.resizable(False, False)
	rg_app.configure(bg=color_bg)

	tab_parent = ttk.Notebook(rg_app)
	tab_main = ttk.Frame(tab_parent)
	tab_params = ttk.Frame(tab_parent)
	tab_options = ttk.Frame(tab_parent)
	tab_about = ttk.Frame(tab_parent)
	tab_parent.add(tab_main, text="Main")
	tab_parent.add(tab_params, text="RG Parameters")
	tab_parent.add(tab_options, text="Options")
	tab_parent.add(tab_about, text="About")
	tab_parent.pack(expand=1, fill='both')

	case_names = 'N', 'omega0', 'Omega', 'K', 'KampInf', 'KampSup'
	case_types = 'Char', 'Char', 'Char', 'Char', 'Char', 'Char'
	case_positions = (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0)
	case_values = '[[1, 1], [1, 0]]', '[-0.618033988749895, 1.0]', '[1.0, 0.0]', '((0, 1, 0), (0, 1, 1))', '[0.0, 0.0]', '[0.04, 0.04]'
	case_options = ('GoldenMean', 'SilverMean', 'BronzeMean', 'SpiralMean', 'TauMean', 'OMean', 'EtaMean')

	param_rg_names = 'L', 'J', 'Sigma', 'Kappa', 'TolMin', 'TolMax', 'TolMinLie', 'MaxIter', 'MaxOA', 'NormAnalytic'
	param_rg_types = 'Int', 'Int', 'Double', 'Double', 'Double', 'Double', 'Double', 'Int', 'Double', 'Double'
	param_rg_values = 5, 5, 0.4, 0.1, 1e-8, '{:1.0e}'.format(1e+1), 1e-10, 1000, 0.2, 1.0
	param_rg_positions = (1, 0), (2, 0), (4,0), (5, 0), (1, 2), (2, 2), (6, 2), (3, 2), (7, 0), (8, 0)

	menu_rg_names = 'ChoiceIm', 'CanonicalTransformation', 'NormChoice', 'Precision'
	menu_rg_types = 'Char', 'Char', 'Char', 'Int'
	menu_rg_values = 'AK2000', 'Lie', 'sum', 64
	menu_rg_menus = ('AK2000', 'K1999', 'AKP1998'), ('Lie', 'Type2', 'Type3'), ('sum', 'max', 'Euclidean', 'Analytic'), (32, 64, 128)
	menu_rg_positions = (1, 4), (5, 4), (3, 4), (7, 4)
	menu_rg_commands = None, None, None, None

	output_names = 'SaveData', 'PlotResults'
	output_types = 'Bool', 'Bool'
	output_values = False, False
	output_positions = (6, 3), (7, 3)

	option_names = 'DistSurf', 'DistCircle', 'Radius', 'ModesPerturb', 'Nh', 'Ncs', 'TolCS', 'NumberOfIterations'
	option_types = 'Double', 'Double', 'Double', 'Int', 'Int', 'Int', 'Double', 'Int'
	option_values = 1e-7, 1e-5, 1e-5, 3, 10, 100, 1e-7, 10
	option_positions = (1, 0), (2, 3), (4, 3), (6, 3), (8, 3), (5, 0), (7, 0), (3, 0)

	case_vars = definevar(tab_main, case_types, case_values)
	makeforms(tab_main, case_vars, case_names, case_positions, (8, 20))
	tk.Label(tab_main, width=20, text='Choose frequency vector:', anchor='w', bg=color_bg, pady=5, font=font, fg=font_color).grid(row=0, column=0, padx=5)
	choice_case = tk.StringVar(tab_main, value=case_options[0])
	tk.OptionMenu(tab_main, choice_case, *case_options, command= lambda x: define_case(x, case_vars)).grid(row=0, column=1, padx=5, sticky='w')
	tk.Label(tab_main, width=10, text=None, bg=color_bg).grid(row=0, column=2)
	tk.Label(tab_main, width=20, text='Choose method:', anchor='w', bg=color_bg, pady=5, font=font, fg=font_color).grid(row=0, column=3, padx=5)
	run_etiqs = 'Iterates', 'Circle Iterates', 'Critical Surface', 'Converge Region'
	run_positions = (1, 3), (2, 3), (3, 3), (4, 3)
	run_method = tk.StringVar()
	run_method.set(run_etiqs[0])
	for (run_etiq, run_position) in zip(run_etiqs, run_positions):
		tk.Radiobutton(tab_main, variable=run_method, text=run_etiq, value=run_etiq, width=15, anchor='w', bg=color_bg).grid(row=run_position[0], column=run_position[1], sticky='w')
	output_vars = definevar(tab_main, output_types, output_values)
	makechecks(tab_main, output_vars, output_names, output_positions)

	param_rg_vars = definevar(tab_params, param_rg_types, param_rg_values)
	makeforms(tab_params, param_rg_vars, param_rg_names, param_rg_positions, (13, 5))
	tk.Label(tab_params, width=10, text=None, bg=color_bg).grid(row=0, column=2)
	menu_rg_vars = definevar(tab_params, menu_rg_types, menu_rg_values)
	makemenus(tab_params, menu_rg_vars, menu_rg_names, menu_rg_menus, menu_rg_positions, 25)

	option_vars = definevar(tab_options, option_types, option_values)
	makeforms(tab_options, option_vars, option_names, option_positions, (14, 8))
	tk.Label(tab_options, width=10, text=None, bg=color_bg).grid(row=0, column=2)

	parameters = [case_names + param_rg_names + menu_rg_names, case_vars + param_rg_vars + menu_rg_vars]
	options = [output_names + option_names, output_vars + option_vars]
	tabs = [tab_main, tab_params, tab_options, tab_about]

	run_button = tk.Button(tab_main, text='Run', highlightbackground=color_bg, width=15,\
	 	command= lambda : rgrun(run_method, parameters, options, tabs)).grid(row=8, column=3, sticky='w')
	tk.Label(tab_main, width=10, text=None, bg=color_bg).grid(row=9, column=0)

	tk.Label(tab_about, width=10, text=None, bg=color_bg).grid(row=0, column=2)
	errorcode = tk.Text(tab_about, height=9, width=35, pady=10, bg=color_bg, font=font, fg=font_color)
	errorcode.insert(tk.INSERT, "ERROR CODES\n\n")
	errorcode.insert(tk.INSERT, "    k-th Lie transform diverging: [1, k]\n")
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
	dict_types = {
		'Double': lambda value : tk.DoubleVar(root, value=value),
		'Int': lambda value : tk.IntVar(root, value=value),
		'Char': lambda value : tk.StringVar(root, value=value),
		'Bool': lambda value : tk.BooleanVar(root, value=value)
	}
	paramlist = []
	for (type, value) in zip(types, values):
		paramlist.append(dict_types.get(type)(value))
	return paramlist

def makeforms(root, fields, names, positions, width):
	for (field, name, position) in zip(fields, names, positions):
		lab = tk.Label(root, width=width[0], text=name, anchor='e', bg=color_bg, font=font, fg=font_color)
		ent = tk.Entry(root, width=width[1], textvariable=field, bg=color_bg)
		lab.grid(row=position[0], column=position[1], pady=5)
		ent.grid(row=position[0], column=position[1]+1)

def makemenus(root, fields, names, menus, positions, width):
	for (field, name, menu, position) in zip(fields, names, menus, positions):
		lab = tk.Label(root, width=width, text=name, anchor='e', bg=color_bg, font=font, fg=font_color)
		men = tk.OptionMenu(root, field, * menu)
		lab.grid(row=position[0], column=position[1], pady=5, sticky='e')
		men.grid(row=position[0]+1, column=position[1], pady=5, sticky='e')

def makechecks(root, fields, names, positions):
	for (field, name, position) in zip(fields, names, positions):
		chec = tk.Checkbutton(root, text=name, variable=field, onvalue=True, offvalue=False, bg=color_bg, font=font, fg=font_color)
		chec.grid(row=position[0], column=position[1], sticky='w')

def rgrun(run_method, parameters, options, tabs):
	dict_param = dict()
	for (name, var) in zip(parameters[0], parameters[1]):
		dict_param[name] = var.get()
	for (name, var) in zip(options[0], options[1]):
		dict_param[name] = var.get()
	for key in ['N', 'omega0', 'Omega', 'K', 'KampInf', 'KampSup']:
		dict_param[key] = list(eval(dict_param[key]))
	case_study = RG.RG(dict_param)
	if run_method.get() == 'Iterates':
		iterates(case_study, tabs)
	elif run_method.get() == 'Circle Iterates':
		iterate_circle(case_study,tabs)
	elif run_method.get() == 'Critical Surface':
		critical_surface(case_study, tabs)
	elif run_method.get() == 'Converge Region':
		converge_region(case_study, tabs)

def define_case(case_option, params):
	if case_option == 'GoldenMean':
	    dict_params = {
			'N': [[1, 1], [1, 0]],
	    	'omega0': [-0.618033988749895, 1.0],
	    	'Omega': [1.0, 0.0],
	    	'K': ((0, 1, 0), (0, 1, 1)),
	    	'KampInf': [0.0, 0.0],
	    	'KampSup': [0.35, 0.12]}
	elif case_option == 'SilverMean':
		dict_params = {
			'N': [[2, 1], [1, 0]],
			'omega0': [-0.414213562373095, 1.0],
			'Omega': [1.0, 0.0],
			'K': ((0, 1, 0), (0, 1, 1)),
			'KampInf': [0.0, 0.0],
			'KampSup': [0.12, 0.225]}
	elif case_option == 'BronzeMean':
		dict_params = {
			'N': [[3, 1], [1, 0]],
			'omega0': [-0.302775637731995, 1.0],
			'Omega': [1.0, 0.0],
			'K': ((0, 1, 0), (0, 1, 1)),
			'KampInf': [0.0, 0.0],
			'KampSup': [0.06, 0.2]}
	elif case_option == 'SpiralMean':
		sigma = 1.3247179572447460259
		dict_params = {
	    	'N': [[0, 0, 1], [1, 0, 0], [0, 1, -1]],
	    	'omega0': [sigma**2, sigma, 1.0],
	    	'Omega': [1.0, 1.0, -1.0],
	    	'K': ((0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
	    	'KampInf': [0.034, 0.089, 0.1],
	    	'KampSup': [0.036, 0.091, 0.1]}
	elif case_option == 'TauMean':
		Tau = 0.445041867912629
		dict_params = {
	    	'N': [[0, 1, -1], [1, -1, 1], [0, -1, 2]],
	    	'omega0': [1.0, Tau, 1.0 - Tau - Tau**2],
	    	'Omega': [1.0, 1.0, -1.0],
	    	'K': ((0, 0, -1, 1), (0, 1, -1, -1), (0, 0, 0, 1)),
	    	'KampInf': [0.0, 0.0, 0.0],
	    	'KampSup': [0.001, 0.007, 0.01]}
	elif case_option == 'OMean':
		o_val = 0.682327803828019
		dict_params = {
	    	'N': [[0, 0, 1], [1, 0, -1], [0, 1, 0]],
	    	'omega0': [1.0, o_val, o_val**2],
	    	'Omega': [1.0, 1.0, 1.0],
	    	'K': ((0, 1, -1, -1), (0, 0, 1, -1), (0, 1, -1, 0)),
	    	'KampInf': [0.0, 0.0, 0.0],
	    	'KampSup': [0.1, 0.1, 0.1]}
	elif case_option == 'EtaMean':
		Eta = -0.347296355333861
		dict_params = {
	    	'N': [[-1, 1, 0], [1, 1, 1], [0, 1, 0]],
	    	'omega0': [Eta **2 - Eta - 1.0, Eta, 1.0],
	    	'Omega': [1.0, -1.0, 1.0],
	    	'K': ((0, 1, 1, 1), (0, -1, 1, 0), (0, 0, 1, 0)),
	    	'KampInf': [0.0, 0.0, 0.0],
	    	'KampSup': [0.1, 0.1, 0.1]}
		for i, key in enumerate(dict_params):
			params[i].set(key)

def iterates(case, tabs):
	h_inf, h_sup = case.generate_2Hamiltonians()
	if (h_inf.error == [0, 0]) and (h_sup.error == [0, 0]):
		timestr = time.strftime("%Y%m%d_%H%M")
		plotf(h_sup.f[0], case)
		data = []
		progress = ttk.Progressbar(tabs[0], orient=tk.HORIZONTAL, length=500, maximum=case.NumberOfIterations, mode='indeterminate')
		progress.grid(row=10, column=0, columnspan=4, sticky='s')
		progress['value'] = 0
		text_output = scrolledtext.ScrolledText(tabs[0], wrap = tk.WORD, width = 85, height = 6)
		text_output.grid(row=11, column=0, columnspan=4)
		k_ = 0
		while (k_ < case.NumberOfIterations) and (h_inf.error == [0, 0]) and (h_sup.error == [0, 0]):
			k_ += 1
			start = time.time()
			h_inf, h_sup = case.approach(h_inf, h_sup, dist=case.DistSurf, strict=True)
			h_inf_ = case.renormalization_group(h_inf)
			h_sup_ = case.renormalization_group(h_sup)
			if k_ == 1:
				text_output.insert(tk.END, 'Critical parameter = {}'.format(2.0 * h_inf.f[case.K[0]]))
			plotf(h_inf_.f[0], case)
			mean2_p = 2.0 * h_inf.f[2][case.zero_]
			diff_p = case.norm(xp.abs(h_inf.f) - xp.abs(h_inf_.f))
			delta_p = case.norm(xp.abs(h_inf_.f) - xp.abs(h_sup_.f)) / case.norm(h_inf.f - h_sup.f)
			data.append([diff_p, delta_p, mean2_p])
			h_inf = copy.deepcopy(h_inf_)
			h_sup = copy.deepcopy(h_sup_)
			text_output.insert(tk.END, '\n diff = %.3e    delta = %.7f   <f2> = %.7f    (done in %d seconds)' % \
					(diff_p, delta_p, mean2_p, int(xp.rint(time.time()-start))))
			text_output.yview_moveto(1)
			progress['value'] += 1
			progress.update()
			save_data('RG_iterates', data, timestr, case, info='diff     delta     <f2>')
		plt.show()
		progress.destroy()

def compute_cr(epsilon, case):
	k_inf_ = case.KampInf.copy()
	k_sup_ = case.KampSup.copy()
	k_inf_[0] = case.KampInf[0] + epsilon * (case.KampSup[0] - case.KampInf[0])
	k_sup_[0] = k_inf_[0]
	case_ = copy.deepcopy(case)
	case_.KampInf = k_inf_
	case_.KampSup = k_sup_
	h_inf, h_sup = case_.generate_2Hamiltonians()
	if case.converge(h_inf) and (not case.converge(h_sup)):
		h_inf, h_sup = case.approach(h_inf, h_sup, dist=case.TolCS)
		return 2.0 * [h_inf.f[case.K[0]], h_inf.f[case.K[1]]]
	else:
		return 2.0 * [h_inf.f[case.K[0]], xp.nan]

def critical_surface(case, tabs):
	timestr = time.strftime("%Y%m%d_%H%M")
	epsilon_ = xp.linspace(0.0, 1.0, case.Ncs)
	pool = multiprocess.Pool(NumCores)
	data = []
	compfun = lambda epsilon: compute_cr(epsilon, case=case)
	progress = ttk.Progressbar(tabs[0], orient=tk.HORIZONTAL, length=500, maximum=len(epsilon_), mode='determinate')
	progress.grid(row=10, column=0, columnspan=4, sticky='s')
	progress['value'] = 0
	for result in pool.imap(compfun, iterable=epsilon_):
		progress['value'] += 1
		progress.update()
		data.append(result)
	progress.destroy()
	data = xp.array(data).transpose()
	save_data('RG_critical_surface', data, timestr, case)
	if case.PlotResults:
		fig = plt.figure()
		ax = fig.gca()
		ax.set_box_aspect(1)
		plt.plot(data[0, :], data[1, :], color='b', linewidth=2)
		ax.set_xlim(case.KampInf[0], case.KampSup[0])
		ax.set_ylim(case.KampInf[1], case.KampSup[1])
		plt.show()

def converge_point(val1, val2, case):
	k_amp_ = case.KampSup.copy()
	k_amp_[0] = val1
	k_amp_[1] = val2
	h_ = case.generate_1Hamiltonian(case.K, k_amp_, case.Omega, symmetric=True)
	return [int(case.converge(h_)), h_.count], h_.error

def converge_region(case, tabs):
	timestr = time.strftime("%Y%m%d_%H%M")
	x_vec = xp.linspace(case.KampInf[0], case.KampSup[0], case.Ncs)
	y_vec = xp.linspace(case.KampInf[1], case.KampSup[1], case.Ncs)
	pool = multiprocess.Pool(NumCores)
	data = []
	info = []
	progress1 = ttk.Progressbar(tabs[0], orient=tk.HORIZONTAL, length=500, maximum=case.Ncs, mode='determinate')
	progress1.grid(row=10, column=0, columnspan=4, sticky='s')
	progress1['value'] = 0
	for y_ in y_vec:
		converge_point_ = lambda val1: converge_point(val1, val2=y_, case=case)
		progress2 = ttk.Progressbar(tabs[0], orient=tk.HORIZONTAL, length=500, maximum=case.Ncs, mode='determinate')
		progress2.grid(row=11, column=0, columnspan=4, sticky='s')
		progress2['value'] = 0
		for result_data, result_info in pool.imap(converge_point_, iterable=x_vec):
			data.append(result_data)
			info.append(result_info)
			progress2['value'] += 1
			progress2.update()
		save_data('RG_converge_region', data, timestr, case, info)
		progress2.destroy()
		progress1['value'] += 1
		progress1.update()
	progress1.destroy()
	save_data('RG_converge_region', xp.array(data).reshape((case.Ncs, case.Ncs, 2)), timestr, case, info=xp.array(info).reshape((case.Ncs, case.Ncs, 2)))
	if case.PlotResults:
		fig = plt.figure()
		ax = fig.gca()
		ax.set_box_aspect(1)
		im = ax.pcolormesh(x_vec, y_vec, xp.array(data)[:, 0].reshape((case.Ncs, case.Ncs)).astype(int), cmap='Reds_r', shading='nearest')
		ax.set_xlim(case.KampInf[0], case.KampSup[0])
		ax.set_ylim(case.KampInf[1], case.KampSup[1])
		fig.colorbar(im)
		plt.show()

def approach_set(k, set1, set2, case, dist, strict):
	set1[k], set2[k] = case.approach(set1[k], set2[k], dist=dist, strict=strict)

def iterate_circle(case, tabs):
	h_inf, h_sup = case.generate_2Hamiltonians()
	h_inf, h_sup = case.approach(h_inf, h_sup, dist=case.DistSurf, strict=True)
	h_inf = case.renormalization_group(h_inf)
	h_sup = case.renormalization_group(h_sup)
	h_inf, h_sup = case.approach(h_inf, h_sup, dist=case.DistCircle, strict=True)
	if (h_inf.error == [0, 0]) and (h_sup.error == [0, 0]):
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
		pool = multiprocess.Pool(NumCores)
		approach_circle = lambda k: approach_set(k, set1=circle_inf, set2=circle_sup, case=case, dist=case.DistSurf, strict=True)
		pool.imap(approach_circle, iterable=range(case.Nh+1))
		Coord = xp.zeros((case.Nh+1, 2, case.NumberOfIterations))
		progress = ttk.Progressbar(tabs[0], orient=tk.HORIZONTAL, length=500, maximum=case.NumberOfIterations, mode='determinate')
		progress.grid(row=10, column=0, columnspan=4, sticky='s')
		progress['value'] = 0
		for i_ in trange(case.NumberOfIterations):
			for k_ in range(case.Nh+1):
				Coord[k_, :, i_] = [xp.vdot(circle_inf[k_].f - hc_inf.f, v1), xp.vdot(circle_inf[k_].f - hc_inf.f, v2)]
			save_data('RG_circle', Coord / case.Radius ** 2, timestr, case)
			if case.PlotResults:
				fig = plt.figure()
				ax = fig.add_subplot(111)
				ax.plot(Coord[:, 0, i_] / case.Radius ** 2, Coord[:, 1, i_] / case.Radius ** 2, label='%d -th iterate' % i_)
				ax.legend()
				plt.pause(1e-17)
			circle_inf = pool.imap(case.renormalization_group, iterable=circle_inf)
			circle_sup = pool.imap(case.renormalization_group, iterable=circle_sup)
			approach_circle = lambda k: approach_set(k, set1=circle_inf, set2=circle_sup, case=case, dist=case.DistSurf, strict=True)
			pool.imap(approach_circle, iterable=range(case.Nh+1))
			hc_inf = case.renormalization_group(hc_inf)
			hc_sup = case.renormalization_group(hc_sup)
			hc_inf, hc_sup = case.approach(hc_inf, hc_sup, dist=case.DistSurf, strict=True)
			progress['value'] += 1
			progress.update()
		plt.show()
		progress.destroy()

def save_data(name, data, timestr, case, info=[]):
	if case.SaveData:
		mdic = case.DictParams.copy()
		mdic.update({'data': data, 'info': info})
		date_today = date.today().strftime(" %B %d, %Y\n")
		mdic.update({'date': date_today, 'author': 'cristel.chandre@univ-amu.fr'})
		savemat(name + '_' + timestr + '.mat', mdic)

def plotf(fun, case):
	plt.rcParams.update({'font.size': 22})
	if case.dim == 2 and case.PlotResults:
		fig, ax = plt.subplots(1,1)
		ax.set_xlim(-case.L, case.L)
		ax.set_ylim(-case.L, case.L)
		color_map = 'hot_r'
		im = ax.imshow(xp.abs(xp.roll(fun, (case.L, case.L), axis=(0,1))).transpose(), origin='lower', extent=[-case.L, case.L, -case.L, case.L], \
						norm=colors.LogNorm(vmin=case.TolMin, vmax=xp.abs(fun).max()), cmap=color_map)
		fig.colorbar(im, orientation='vertical')
	elif case.dim == 3 and case.PlotResults:
		Y, Z = xp.meshgrid(xp.arange(-case.L, case.L+1), xp.arange(-case.L, case.L+1))
		X = xp.zeros_like(Y)
		fig = plt.figure()
		ax = fig.gca(projection='3d')
		ax.set_box_aspect((5/3, 1/3, 1/3))
		norm_c = colors.LogNorm(vmin=case.TolMin, vmax=xp.abs(fun).max())
		for k_ in range(-case.L, case.L+1):
			A = xp.abs(xp.roll(fun[k_, :, :], (case.L,case.L), axis=(0,1)))
			ax.plot_surface(X + k_, Y, Z, rstride=1, cstride=1, facecolors=cm.hot(norm_c(A)), alpha=0.4, linewidth=0.0, \
							shade=False)
		ax.set_xticks((-case.L,0,case.L))
		ax.set_yticks((-case.L,0,case.L))
		ax.set_zticks((-case.L,0,case.L))
		ax.set_xlim((-case.L-1/2, case.L+1/2))
		ax.set_ylim((-case.L-1/2, case.L+1/2))
		ax.set_zlim((-case.L-1/2, case.L+1/2))
		ax.view_init(elev=20, azim=120)
	plt.pause(1e-17)

if __name__ == "__main__":
	main()
