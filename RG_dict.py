########################################################################################################################
##                                   Definition of the parameters for RG                                              ##
########################################################################################################################
##                                                                                                                    ##
##   Method: 'iterates', 'critical_surface', 'converge_region'; choice of method                                      ##
##   Iterates: integer; number of iterates to compute in iterates()                                                   ##
##   Nxy: integer; number of points in the (x,y) figures for 'critical_surface' and 'converge_region'                 ##
##   DistSurf: float; distance of approach for the computation of critical values                                     ##
##                                                                                                                    ##
##   N: nxn integer matrix with determinant ±1                                                                        ##
##   omega0: array of n floats; frequency vector of the invariant torus; should be an eigenvector of N.transpose      ##
##   Omega: array of n floats; vector of the perturation in action                                                    ##
##   K: tuples of integers; wavevectors of the perturbation                                                           ##
##   AmpInf: array of floats; minimal amplitudes of the perturbation                                                  ##
##   AmpSup: array of floats; maximum amplitudes of the perturbation                                                  ##
##                                                                                                                    ##
##   L: integer; truncation in Fourier series (angles)                                                                ##
##   J: integer; truncation in Taylor series  (actions)                                                               ##
##                                                                                                                    ##
##   ChoiceIm: 'AK2000', 'K1999', 'AKW1998'; definition of I-                                                         ##
##   Sigma: float; definition of I-                                                                                   ##
##   Kappa: float; definition of I-                                                                                   ##
##                                                                                                                    ##
##   CanonicalTransformation: 'Lie', 'Lie_scaling', 'Lie_adaptive'; method to compute the canonical Lie transforms    ##
##   LieSteps: integer; number of steps in the scaling and squaring procedure to compute exponentials                 ##
##   MinStep: float; minimum value of the steps in the adaptive approach to compute exponentials                      ##
##   AbsTol: float; absolute tolerance for the adaptive approach to compute exponentials                              ##
##   RelTol: float; relative tolerance for the adaptive approach to compute exponentials                              ##
##                                                                                                                    ##
##   TolMax: float; value of Hamiltonian norm for divergence                                                          ##
##   TolMin: float; value of Hamiltonian norm for convergence                                                         ##
##   TolMinLie: float; value of norm for convergence of Lie transforms                                                ##
##   MaxLie: integer; maximum number of elements in Taylor series of exponentials                                     ##
##   MaxIterates: integer; maximum number of iterates for convergence/divergence                                      ##
##                                                                                                                    ##
##   Precision: 32, 64 or 128; precision of calculations (default=64)                                                 ##
##   NormChoice: 'sum', 'max', 'Euclidean', 'Analytic'; choice of Hamiltonian norm                                    ##
##   NormAnalytic: float; parameter of norm 'Analytic'                                                                ##
##                                                                                                                    ##
##   SaveData: boolean; if True, the results are saved in a .mat file                                                 ##
##   PlotResults: boolean; if True, the results are plotted right after the computation                               ##
##   Parallelization: 2d array [boolean, int]; True for parallelization, int is the number of processors to be used   ##
##                                                                                                                    ##
########################################################################################################################
import numpy as xp

Method = 'iterates'
Iterates = 10

#Method = 'converge_region'
#Method = 'critical_surface'
Nxy = 8
DistSurf = 1e-7

# N = [[1, 1], [1, 0]]
# omega0 = [(xp.sqrt(5)-1)/2, -1]
# Omega = [1, 0]
# K = ((0, 1, 0), (0, 1, 1))
# AmpInf = [0, 0]
# AmpSup = [0.04, 0.04]

N = [[0, 0, 1], [1, 0, 0], [0, 1, -1]]
sigma = 1.3247179572447460259
omega0 = [sigma**2, sigma, 1.0]
Omega = [1, 1, -1]
K = ((0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
AmpInf = [0.2, 0.03, 0.1]
AmpSup = [0.25, 0.05, 0.1]

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
Parallelization = [True, 4]

########################################################################################################################
##                                                DO NOT EDIT BELOW                                                   ##
########################################################################################################################
Precision = {32: xp.float32, 64: xp.float64, 128: xp.float128}.get(Precision, xp.float64)
dict = {'Method': Method}
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
