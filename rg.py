import tkinter as tk
from tkinter import ttk
import time
from itertools import chain

version_rgapp = "0.1"
date_rgapp = time.strftime("%Y / %m / %d")

rg_app = tk.Tk()
rg_app.title("Renormalization Group for Hamiltonians")

window_x = 700
window_y = 400
screen_width = rg_app.winfo_screenwidth()
screen_height = rg_app.winfo_screenheight()
position_x = (screen_width // 2) - (window_x // 2)
position_y = (screen_height // 2) - (window_y // 2)
geo = "{}x{}+{}+{}".format(window_x, window_y, position_x, position_y)
rg_app.geometry(geo)
rg_app.configure(bg='#BFC9CA')

style = ttk.Style()
style.theme_use('clam')
style.configure("BW.TLabel", foreground="black", background="white")

tab_parent = ttk.Notebook(rg_app)
tab_param = ttk.Frame(tab_parent)
tab_advanced = ttk.Frame(tab_parent)
tab_run = ttk.Frame(tab_parent)
tab_about = ttk.Frame(tab_parent)

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

def makeform(root, fields, names, positions):
	entries = []
	for (field, name, position) in zip(fields, names, positions):
		lab = tk.Label(root, width=10, text=name, anchor='e')
		ent = tk.Entry(root, width=15, textvariable=field)
		lab.grid(row=position[0], column=position[1], pady=5)
		ent.grid(row=position[0], column=position[1]+1, pady=5, padx=15)
		entries.append((field, ent))
	return entries

def makemenus(root, fields, names, menus, positions):
	entries = []
	for (field, name, menu, position) in zip(fields, names, menus, positions):
		lab = tk.Label(root, width=18, text=name, anchor='e')
		men = tk.OptionMenu(root, field, * menu)
		lab.grid(row=position[0], column=position[1], pady=10)
		men.grid(row=position[0], column=position[1]+1, pady=10, padx=10)
		entries.append((field, men))
	return entries

def makechecks(root, fields, names, positions):
	entries = []
	for (field, name, position) in zip(fields, names, positions):
		chec = tk.Checkbutton(root, text=name, variable=field, onvalue=True, offvalue=False)
		chec.grid(row=position[0], column=position[1])
		entries.append((field, chec))
	return entries

def runcommand():
	for (name, entry) in chain(zip(mp_names, mp_par), zip(mpc_names, mpc_par),\
	 zip(adv_names, adv_par), zip(adv2_names, adv2_par)):
		globals()[name] = entry[0].get()
	print(Sigma)
	import RGfunctions

def define_case(case):
	if case == 'GoldenMean':
	    N = [(1, 1), (1, 0)]
	    Eigenvalues = [-0.618033988749895, 1.618033988749895]
	    Omega0 = (Eigenvalues[0], 1.0)
	    Omega = [1.0, 0.0]
	    K = ((0, 1, 0), (0, 1, 1))
	    KampInf = [0.02, 0.02]
	    KampSup = [0.04, 0.04]
	elif case == 'SpiralMean':
	    N = [(0, 0, 1), (1, 0, 0), (0, 1, -1)]
	    sigma = 1.3247179572447460259
	    Eigenvalues = [1.0 / sigma]
	    Omega0 = (sigma**2, sigma, 1.0)
	    Omega = [1.0, 1.0, -1.0]
	    K = ((0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
	    KampInf = [0.034, 0.089, 0.1]
	    KampSup = [0.036, 0.091, 0.1]
	elif case == 'TauMean':
	    N = [(0, 1, -1), (1, -1, 1), (0, -1, 2)]
	    Tau = 0.445041867912629
	    Tau2 = 1.801937735804839
	    Tau3 = -1.246979603717467
	    Eigenvalues = [Tau, Tau2, Tau3]
	    Omega0 = (1.0, Tau, 1.0 - Tau - Tau**2)
	    Omega = [1.0, 1.0, -1.0]
	    K = ((0, 0, -1, 1), (0, 1, -1, -1), (0, 0, 0, 1))
	    KampInf = [0.0, 0.0, 0.0]
	    KampSup = [0.1, 0.1, 0.1]
	elif case == 'OMean':
	    N = [(0, 0, 1), (1, 0, -1), (0, 1, 0)]
	    o_val = 0.682327803828019
	    Eigenvalues = [o_val]
	    Omega0 = (1.0, o_val, o_val**2)
	    FixedOmega = False
	    Omega = [1.0, 1.0, 1.0]
	    K = ((0, 1, -1, -1), (0, 0, 1, -1), (0, 1, -1, 0))
	    KampInf = [0.0, 0.0, 0.0]
	    KampSup = [0.1, 0.1, 0.1]
	elif case == 'EtaMean':
	    N = [(-1, 1, 0), (1, 1, 1), (0, 1, 0)]
	    Eta = -0.347296355333861
	    Eta2 = -1.532088886237956
	    Eta3 = 1.879385241571816
	    Eigenvalues = [Eta, Eta2, Eta3]
	    Omega0 = (Eta **2 - Eta - 1.0, Eta, 1.0)
	    Omega = [1.0, -1.0, 1.0]
	    K = ((0, 1, 1, 1), (0, -1, 1, 0), (0, 0, 1, 0))
	    KampInf = [0.0, 0.0, 0.0]
	    KampSup = [0.1, 0.1, 0.1]

tab_parent.add(tab_param, text="Main Parameters")
tab_parent.add(tab_advanced, text="Advanced Parameters")
tab_parent.add(tab_run, text="Run")
tab_parent.add(tab_about, text="About")

tab_parent.pack(expand=1, fill='both')

mp_names = 'L', 'J', 'Sigma', 'Kappa'
mp_types = 'Int', 'Int', 'Double', 'Double'
mp_values = 5, 5, 0.4, 0.1
mp_positions = (1, 0), (2, 0), (4,0), (5, 0)

mpc_names = 'SaveData', 'PlotResults'
mpc_types = 'Bool', 'Bool'
mpc_values = True, False
mpc_positions = (8, 0), (8, 1)

tol_names = 'TolMin', 'TolMax', 'TolLie', 'MaxIter', 'MaxLie', 'DistSurf'
tol_types = 'Double', 'Double', 'Double', 'Int', 'Int', 'Double'
tol_values = 1e-9, 1e+9, 1e-11, 5000, 5000, 1e-7
tol_positions = (1, 3), (2, 3), (5, 3), (3, 3), (6, 3), (8, 3)

adv_names = 'ChoiceIm', 'CanonicalTransformation', 'NormChoice', 'Precision'
adv_types = 'Char', 'Char', 'Char', 'Int'
adv_values = 'AK2000', 'Lie', 'sum', 64
adv_menus = ('AK2000', 'K1999', 'AKP1998'), ('Lie', 'Type2', 'Type3'), ('sum', 'max', 'Euclidian', 'Analytic'), (32, 64, 128)
adv_positions = (1, 0), (5, 0), (3, 0), (9, 0)
adv_commands = None, None, None, None

adv2_names = 'MaxA', 'DistCircle', 'Radius', 'ModesPerturb', 'Nh', 'Ncs', 'TolCS'
adv2_types = 'Double', 'Double', 'Double', 'Int', 'Int', 'Int', 'Double'
adv2_values = 0.2, 1e-5, 1e-5, 3, 10, 100, 1e-7
adv2_positions = (7, 0), (1, 3), (2, 3), (3, 3), (4, 3), (7, 3), (8, 3)

run_etiqs = 'Iterates', 'Circle', 'Critical Surface', 'Converge Region'
run_vals = 1, 2, 3, 4
run_positions = (1, 3), (3, 3), (5, 3), (7, 3)
var_run = tk.IntVar()
var_run.set(run_vals[0])
for (run_etiq, run_val, run_position) in zip(run_etiqs, run_vals, run_positions):
	b_method = tk.Radiobutton(tab_run, variable=var_run, text=run_etiq, value=run_val, width=15, anchor='w')
	b_method.grid(row=run_position[0], column=run_position[1], sticky='w')


case_options = ('GoldenMean', 'SpiralMean', 'TauMean', 'OMean', 'EtaMean')

mp_params = definevar(tab_param, mp_types, mp_values)
mp_par = makeform(tab_param, mp_params, mp_names, mp_positions)

mpc_params = definevar(tab_param, mpc_types, mpc_values)
mpc_par = makechecks(tab_param, mpc_params, mpc_names, mpc_positions)

tol_params = definevar(tab_param, tol_types, tol_values)
tol_par = makeform(tab_param, tol_params, tol_names, tol_positions)

adv_params = definevar(tab_advanced, adv_types, adv_values)
adv_par = makemenus(tab_advanced, adv_params, adv_names, adv_menus, adv_positions)

adv2_params = definevar(tab_advanced, adv2_types, adv2_values)
adv2_par = makeform(tab_advanced, adv2_params, adv2_names, adv2_positions)

case_lab = tk.Label(tab_run, width=20, text='Choose frequency vector', anchor='w')
case_var = tk.StringVar(tab_run, value=case_options[0])
case_menu = tk.OptionMenu(tab_run, case_var, *case_options, command=define_case)
case_menu.grid(row=1, column=0, padx=10)
case_lab.grid(row=0, column=0, padx=10)


tk.Button(tab_run, text='Run', command=runcommand).grid(row=11, column=4)
tk.Button(tab_run, text='Quit', command=rg_app.quit).grid(row=11, column=5)

errorcode = tk.Text(tab_about, height=9, width=48, pady=10)
errorcode.insert(tk.INSERT, "ERROR CODES\n\n")
errorcode.insert(tk.INSERT, "    k-th Lie transform diverging: [1, k]\n")
errorcode.insert(tk.INSERT, "    k-th Lie transform not converging: [-1, k]\n")
errorcode.insert(tk.INSERT, "    I- iterations diverging: [2, 0]\n")
errorcode.insert(tk.INSERT, "    I- iterations not converging: [-2, 0]\n")
errorcode.insert(tk.INSERT, "    below (approach): [3, 0]\n")
errorcode.insert(tk.INSERT, "    above (generate_2Hamiltonians): [4, 0]\n")
errorcode.insert(tk.END, "    below (generate_2Hamiltonians): [-4, 0]")
errorcode.grid(row=1, column=0)

author = tk.Text(tab_about, height=4, width=33, pady=10)
author.insert(tk.INSERT, "AUTHOR\n\n")
author.insert(tk.INSERT, "     Cristel Chandre (I2M, CNRS)\n")
author.insert(tk.END, "     cristel.chandre@univ-amu.fr ")
author.grid(row=2, column=0)

version = tk.Text(tab_about, height=1, width=37, pady=10)
version.insert(tk.INSERT, "VERSION    ")
version.insert(tk.INSERT, version_rgapp )
version.insert(tk.END, "   (" + date_rgapp + ")")
version.grid(row=2, column=2)

rg_app.mainloop()
