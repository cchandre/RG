J = 5
L = 5

TolLie = 1e-16
TolMin = 1e-15
TolMax = 1e+10
Dist_Surf = 1e-9

Dist_Circle = 1e-5 # distance of the circle from the surface
Radius = 1e-5 #radius of the circle
Modes_Perturb = 3 #number of modes in the perturbation
Nh = 10 #points on the circle

N_cs = 10 # number of points on the critical surface
K_indx = (0, 1) # indices of K for which the critical surface is computed
TolCS = 1e-6

Number_of_Iterations = 2

Save_Data = False

MaxLie = 50000
MaxIter = 50000

#norm_choice = 'max'
norm_choice = 'sum'
#norm_choice = 'Euclidian'
#norm_choice = 1.0

Sigma = 0.4
Kappa = 0.1

precision = 64 #@param ["32", "64", "128"] {type:"raw"}

case = 'GoldenMean'
#case = 'SpiralMean'
#case = 'OMean'
#case = 'TauMean'
#case = 'EtaMean'

if case == 'GoldenMean':
    N = [[1, 1],[1, 0]]
    eigenvalues = [-0.618033988749895, 1.618033988749895]
    omega_0 = [eigenvalues[0], 1.0]

    #fixed_Omega = True
    #Omega = [eigenvalues[1], 1.0]

    fixed_Omega = False
    Omega = [1.0, 0.0]

    K = ((0, 1, 0), (0, 1, 1))
    K_amp_inf = [0.0, 0.08]
    K_amp_sup = [0.004, 0.11]

elif case == 'SpiralMean':
    N = [[0, 0, 1],[1, 0, 0],[0, 1, -1]]
    sigma = 1.3247179572447460259
    eigenvalues = [1.0 / sigma]
    omega_0 = [sigma**2, sigma, 1.0]

    fixed_Omega = False
    Omega = [1.0, 1.0, -1.0]

    K = ((0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
    K_amp_inf = [0.005, 0.055, 0.1]
    K_amp_sup = [0.15, 0.11, 0.1]
    #K = ((0, 0, 1, -1), (0, 1, -1, 0), (0, 1, -1, -1))

elif case == 'TauMean':
    N = [[0, 1, -1],[1, -1, 1],[0, -1, 2]]
    Tau = 0.445041867912629
    Tau2 = 1.801937735804839
    Tau3 = -1.246979603717467
    eigenvalues = [Tau, Tau2]
    omega_0 = [1.0, Tau, 1.0 - Tau - Tau**2]

    fixed_Omega = False
    Omega = [1.0, 1.0, -1.0]

    #fixed_Omega = True
    #Omega = [1.0, Tau2, 1.0 - Tau2 - Tau2**2]
    #Omega = [1.0, Tau3, 1.0 - Tau3 - Tau3**2]

    K = ((0, 0, -1, 1), (0, 1, -1, -1), (0, 0, 0, 1))
    K_amp_inf = [0.0, 0.0, 0.0]
    K_amp_sup = [0.1, 0.1, 0.1]

elif case == 'OMean':
    N = [[0, 0, 1],[1, 0, -1],[0, 1, 0]]
    o_val = 0.682327803828019
    eigenvalues = [o_val, 0.0]
    omega_0 = [1.0, o_val, o_val**2]

    fixed_Omega = False
    Omega = [1.0, 1.0, 1.0]

    #K = ((0, 1, -1, -1), (0, 0, 1, -1), (0, 1, -1, 0))
    K_amp_inf = [0.0, 0.0, 0.0]
    K_amp_sup = [0.1, 0.1, 0.1]

elif case == 'EtaMean':
    N = [[-1, 1, 0],[1, 1, 1],[0, 1, 0]]
    Eta = -0.347296355333861
    Eta2 = -1.532088886237956
    Eta3 = 1.879385241571816
    eigenvalues = [Eta, Eta3]
    omega_0 = [Eta **2 - Eta - 1.0, Eta, 1.0]

    fixed_Omega = False
    Omega = [1.0, -1.0, 1.0]

    #fixed_Omega = True
    #Omega = [Eta2 **2 - Eta2 - 1.0, Eta2, 1.0]
    #Omega = [Eta3 **2 - Eta3 - 1.0, Eta3, 1.0]

    #K = ((0, 1, 1, 1), (0, -1, 1, 0), (0, 0, 1, 0))
    K_amp_inf = [0.0, 0.0, 0.0]
    K_amp_sup = [0.1, 0.1, 0.1]
