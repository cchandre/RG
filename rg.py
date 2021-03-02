import tkinter as tk
from tkinter import ttk

rg_app = tk.Tk()
rg_app.title("Renormalization Group for Hamiltonians")

window_x = 800
window_y = 600
screen_width = rg_app.winfo_screenwidth()
screen_height = rg_app.winfo_screenheight()
position_x = (screen_width // 2) - (window_x // 2)
position_y = (screen_height // 2) - (window_y // 2)
geo = "{}x{}+{}+{}".format(window_x, window_y, position_x, position_y)
rg_app.geometry(geo)

s = ttk.Style()
s.theme_use('clam')
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
		lab = tk.Label(root, width=20, text=name, anchor='e')
		men = ttk.Combobox(root, textvariable=field)
		men['values'] = menu
		men.state(["readonly"])
		lab.grid(row=position[0], column=position[1], pady=10)
		men.grid(row=position[0], column=position[1]+1, pady=10, padx=10)
		entries.append((field, men))
	return entries

def checks(root, fields, names, positions):
	entries = []
	for (field, name, position) in zip(fields, names, positions):
		chec = tk.Checkbutton(root, text=name, variable=field, onvalue=True, offvalue=False)
		chec.grid(row=position[0], column=position[1])
		entries.append((field, chec))
	return entries

def fetch(entries):
	for entry in entries:
		field = entry[0]
		text = entry[1].get()
	print(field, text)
	return field, text


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
mpc_values = False, False
mpc_positions = (7, 0), (8, 0)

tol_names = 'TolMin', 'TolMax', 'TolLie', 'MaxIter', 'MaxLie', 'DistSurf'
tol_types = 'Double', 'Double', 'Double', 'Int', 'Int', 'Double'
tol_values = 1e-9, 1e+9, 1e-11, 5000, 5000, 1e-7
tol_positions = (1, 3), (2, 3), (5, 3), (3, 3), (6, 3), (8, 3)

adv_names = 'ChoiceIm', 'CanonicalTransformation', 'NormChoice', 'Precision'
adv_types = 'Char', 'Char', 'Char', 'Int'
adv_values = 'AK2000', 'Lie', 'sum', 64
adv_menus = ('AK2000', 'K1999', 'AKP1998'), ('Lie', 'Type2', 'Type3'), ('sum', 'max', 'Euclidian', 'Analytic'), (32, 64, 128)
adv_positions = (1, 0), (5, 0), (3, 0), (9, 0)

adv2_names = 'MaxA', 'DistCircle', 'Radius', 'ModesPerturb', 'Nh', 'Ncs', 'Kindx', 'TolCS'
adv2_types = 'Double', 'Double', 'Double', 'Int', 'Int', 'Int', 'Int', 'Double'
adv2_values = 0.2, 1e-5, 1e-5, 3, 10, 100, (0, 1), 1e-7
adv2_positions = (7, 0), (1, 3), (2, 3), (3, 3), (4, 3), (7, 3), (8, 3), (9, 3)

mp_params = definevar(tab_param, mp_types, mp_values)
mp_par = makeform(tab_param, mp_params, mp_names, mp_positions)
rg_app.bind('<Return>', (lambda event, e=mp_par: fetch(e)))

tol_params = definevar(tab_param, tol_types, tol_values)
tol_par = makeform(tab_param, tol_params, tol_names, tol_positions)
rg_app.bind('<Return>', (lambda event, e=tol_par: fetch(e)))

adv_params = definevar(tab_advanced, adv_types, adv_values)
adv_par = makemenus(tab_advanced, adv_params, adv_names, adv_menus, adv_positions)
rg_app.bind('<Return>', (lambda event, e=adv_par: fetch(e)))


tk.Button(tab_run, text='Run', command=rg_app.quit).grid(row=11, column=4)
tk.Button(tab_run, text='Quit', command=rg_app.quit).grid(row=11, column=5)

rg_app.mainloop()
