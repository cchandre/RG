J = 5
L = 5

Sigma = 0.4
Kappa = 0.1

#ChoiceIm = 'AKP1998' # [J J Abad, H Koch, P Wittwer, Nonlinearity 11, 1185 (1998)]
#ChoiceIm = 'K1999'   # [H Koch, ETDS 19, 475 (1999)]
ChoiceIm = 'AK2000'  # [J J Abad, H Koch, CMP 212, 371 (2000)]

CanonicalTransformation = 'Lie'
#CanonicalTransformation = 'Type2'
#CanonicalTransformation = 'Type3'
MaxA = 0.2

TolLie = 1e-12
TolMin = 1e-10
TolMax = 1e+6
MaxLie = 50000
MaxIter = 5000
DistSurf = 1e-8

## Parameters used in iterate_circle
DistCircle = 1e-5 # distance of the circle from the surface
Radius = 1e-5 #radius of the circle
ModesPerturb = 3 #number of modes in the perturbation
Nh = 10 #points on the circle

## Parameters used in critical_surface and converge_region
Ncs = 200 # number of points on the critical surface or converge region
Kindx = (0, 1) # indices of K for which the critical surface / converge region is computed
TolCS = 1e-7

NumberOfIterations = 20

SaveData = True
PlotResults = False

#NormChoice = 'max'
NormChoice = 'sum'
#NormChoice = 'Euclidian'
#NormChoice = 1.0

Precision = 64

Case = 'GoldenMean'
#Case = 'SpiralMean'
#Case = 'OMean'
#Case = 'TauMean'
#Case = 'EtaMean'

if Case == 'GoldenMean':
    N = [(1, 1), (1, 0)]
    Eigenvalues = [-0.618033988749895, 1.618033988749895]
    Omega0 = (Eigenvalues[0], 1.0)

    #FixedOmega = True
    #Omega = (Eigenvalues[1], 1.0)

    FixedOmega = False
    Omega = [1.0, 0.0]

    K = ((0, 1, 0), (0, 1, 1))
    KampInf = [0.02, 0.02]
    KampSup = [0.04, 0.04]

elif Case == 'SpiralMean':
    N = [(0, 0, 1), (1, 0, 0), (0, 1, -1)]
    sigma = 1.3247179572447460259
    Eigenvalues = [1.0 / sigma]
    Omega0 = (sigma**2, sigma, 1.0)

    FixedOmega = False
    Omega = [1.0, 1.0, -1.0]

    K = ((0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
    KampInf = [0.034, 0.089, 0.1]
    KampSup = [0.036, 0.091, 0.1]
    #K = ((0, 0, 1, -1), (0, 1, -1, 0), (0, 1, -1, -1))

elif Case == 'TauMean':
    N = [(0, 1, -1), (1, -1, 1), (0, -1, 2)]
    Tau = 0.445041867912629
    Tau2 = 1.801937735804839
    Tau3 = -1.246979603717467
    Eigenvalues = [Tau, Tau2, Tau3]
    Omega0 = (1.0, Tau, 1.0 - Tau - Tau**2)

    FixedOmega = False
    Omega = [1.0, 1.0, -1.0]

    #FixedOmega = True
    #Omega = [1.0, Tau2, 1.0 - Tau2 - Tau2**2]
    #Omega = [1.0, Tau3, 1.0 - Tau3 - Tau3**2]

    K = ((0, 0, -1, 1), (0, 1, -1, -1), (0, 0, 0, 1))
    KampInf = [0.0, 0.0, 0.0]
    KampSup = [0.1, 0.1, 0.1]

elif Case == 'OMean':
    N = [(0, 0, 1), (1, 0, -1), (0, 1, 0)]
    o_val = 0.682327803828019
    Eigenvalues = [o_val]
    Omega0 = (1.0, o_val, o_val**2)

    FixedOmega = False
    Omega = [1.0, 1.0, 1.0]

    K = ((0, 1, -1, -1), (0, 0, 1, -1), (0, 1, -1, 0))
    KampInf = [0.0, 0.0, 0.0]
    KampSup = [0.1, 0.1, 0.1]

elif Case == 'EtaMean':
    N = [(-1, 1, 0), (1, 1, 1), (0, 1, 0)]
    Eta = -0.347296355333861
    Eta2 = -1.532088886237956
    Eta3 = 1.879385241571816
    Eigenvalues = [Eta, Eta2, Eta3]
    Omega0 = (Eta **2 - Eta - 1.0, Eta, 1.0)

    FixedOmega = False
    Omega = [1.0, -1.0, 1.0]

    #FixedOmega = True
    #Omega = [Eta2 **2 - Eta2 - 1.0, Eta2, 1.0]
    #Omega = [Eta3 **2 - Eta3 - 1.0, Eta3, 1.0]

    K = ((0, 1, 1, 1), (0, -1, 1, 0), (0, 0, 1, 0))
    KampInf = [0.0, 0.0, 0.0]
    KampSup = [0.1, 0.1, 0.1]
