########################################################################################################################
##                              Dictionary of parameters: https://github.com/cchandre/RG                              ##
########################################################################################################################

import numpy as xp

#Method = 'iterates'
Iterates = 10

Method = 'region'
#Method = 'surface'
Nxy = 16
DistSurf = 1e-7

N = [[1, 1], [1, 0]]
omega0 = [(xp.sqrt(5)-1)/2, -1]
Omega = [1, 0]
K = ((0, 1, 0), (0, 1, 1))
AmpInf = [0, 0]
AmpSup = [0.04, 0.04]

# N = [[0, 0, 1], [1, 0, 0], [0, 1, -1]]
# sigma = 1.3247179572447460259
# omega0 = [sigma**2, sigma, 1.0]
# Omega = [1, 1, -1]
# K = ((0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
# AmpInf = [0.2, 0.03, 0.1]
# AmpSup = [0.25, 0.05, 0.1]

L = 5
J = 5

ChoiceIm = 'AK2000'
Sigma = 0.6
Kappa = 0.1

CanonicalTransformation = 'Lie'
LieSteps = 4
MinStep = 0.05
AbsTol = 1e-2
RelTol = 1e-3

TolMax = 1e+10
TolMin = 1e-8
TolMinLie = 1e-10
MaxLie = 500
MaxIterates = 100

NormChoice = 'sum'
NormAnalytic = 1
Precision = 64

SaveData = False
PlotResults = True
Parallelization = (True, 4)

########################################################################################################################
##                                                DO NOT EDIT BELOW                                                   ##
########################################################################################################################
Precision = {32: xp.float32, 64: xp.float64, 128: xp.float128}.get(Precision, xp.float64)
dict = {'Method': 'compute_' + Method}
dict.update({
        'Iterates': Iterates,
		'Nxy': Nxy,
		'DistSurf': DistSurf,
		'N': xp.asarray(N, dtype=int),
		'omega0': xp.asarray(omega0, dtype=Precision),
		'Omega': xp.asarray(Omega, dtype=Precision),
		'K': K,
		'AmpInf': AmpInf,
		'AmpSup': AmpSup,
		'L': L,
		'J': J,
		'ChoiceIm': ChoiceIm,
		'Sigma': Sigma,
		'Kappa': Kappa,
		'CanonicalTransformation': CanonicalTransformation,
		'LieSteps': LieSteps,
		'MinStep': MinStep,
		'AbsTol': AbsTol,
		'RelTol': RelTol,
		'TolMax': TolMax,
		'TolMin': TolMin,
		'TolMinLie': TolMinLie,
		'MaxLie': MaxLie,
		'MaxIterates': MaxIterates,
		'NormChoice': NormChoice,
		'NormAnalytic': NormAnalytic,
		'Precision': Precision,
		'SaveData': SaveData,
		'PlotResults': PlotResults,
		'Parallelization': Parallelization})
########################################################################################################################
