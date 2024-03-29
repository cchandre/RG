# Renormalization group (RG) for the break-up of invariant tori in Hamiltonian flows

- [`RG_dict.py`](https://github.com/cchandre/RG/blob/main/RG_dict.py): to be edited to change the parameters of the RG computation (see below for a dictionary of parameters)

- [`RG.py`](https://github.com/cchandre/RG/blob/main/RG.py): contains the RG classes and main functions defining the RG map

- [`RG_modules.py`](https://github.com/cchandre/RG/blob/main/RG_modules.py): contains the methods to execute the RG map

Once [`RG_dict.py`](https://github.com/cchandre/RG/blob/main/RG_dict.py) has been edited with the relevant parameters, run the file as 
```sh
python3 RG.py
```
or 
```sh
nohup python3 -u RG.py &>RG.out < /dev/null &
```
The list of Python packages and their version are specified in [`requirements.txt`](https://github.com/cchandre/RG/blob/main/requirements.txt)

___
##  Parameter dictionary

- *Method*: string; 'iterates', 'surface', 'region', 'line'; choice of method
  - 'iterates': starting from two Hamiltonians *H*<sub>1</sub> and *H*<sub>2</sub> defined with the modes *K* and amplitudes *AmpInf* and *AmpSup* respectively, the method first refines *H*<sub>1</sub> and *H*<sub>2</sub> close to the critical surface using a dichotomy procedure. Second, it iterates these two Hamiltonians by iterating and refining with the renormalization map
  - 'line': for the family of Hamiltonians defined by the modes *K* in the direction *DirLine* with *ModesLine*=1, determines the critical threshold 
  - 'surface': computes the critical surface in the plane of Fourier modes defined by *K* and *ModesLine* (the two modes *K* with *ModesLine*=1)
  - 'region': in the plane of Fourier modes defined by *K* and *ModesLine* (the two modes *K* with *ModesLine*=1) with amplitudes in the range defined by *AmpInf* and *AmpSup*, determines the number of iterations for the Hamiltonian to converge (negative integers) or diverge (positive integers)
- *Iterates*: integer; number of iterates to compute for *Method*='iterates'
- *Nxy*: integer; number of points along each direction for *Method*='surface' or 'region'
- *RelDist*: float; relative distance of approach for the computation of critical values
####
- *N*: *n*x*n* integer matrix with determinant ±1
- *omega0*: array of *n* floats; frequency vector **&omega;** of the invariant torus; should be an eigenvector of <sup>&dagger;</sup>*N* (transposed matrix of *N*)
- *Omega*: array of *n* floats; vector **&Omega;** of the perturation in action
- *K*: 2-dimensional tuple of integers; wavevectors (j,**&nu;**)=(j,k<sub>1</sub>,...,k<sub>n</sub>) of the perturbation 
- *AmpInf*: array of *len(K)* floats; minimal amplitudes of the perturbation 
- *AmpSup*: array of *len(K)* floats; maximum amplitudes of the perturbation
- *CoordLine*: 1d array of floats; min and max values of the amplitudes of the potential used in *Method*='line'   
- *ModesLine*: tuple of 0 and 1 of length *len(K)*; specify which modes are being varied (1 for a varied mode)     
- *DirLine*: 1d array of floats; direction of the one-parameter family used in *Method*='line' 
####
- *L*: integer; truncation in Fourier series (angles) 
- *J*: integer; truncation in Taylor series  (actions) 
####
- *ChoiceIm*: string; 'AK2000', 'K1999', 'AKW1998'; definition of *I<sup>-</sup>* 
- *Sigma*: float; definition of *I<sup>-</sup>*
- *Kappa*: float; definition of *I<sup>-</sup>*
####
- *CanonicalTransformation*: string; 'expm_onestep', 'expm_adapt', 'expm_multiply'; method to compute the canonical Lie transforms 
  - 'expm_onestep': compute the exponential of the Liouville operator in one single step
  - 'expm_adapt': use an adaptative step-size method to compute the exponential of the Liouville operator with *AbsTol* and *RelTol* as tolerance parameters, and *MinStep* as the minimum value of the step to be used
  - 'expm_multiply': use the algorithm developed in [A.H. Al-Mohy, N.J. Higham, SIAM Journal on Scientific Computing 33, 488 (2011)]( http://eprints.ma.man.ac.uk/1591/) to compute the exponential of the Liouville operator
- *MinStep*: float; minimum value of the steps in the adaptive procedure to compute exponentials (for 'expm_adapt')
- *AbsTol*: float; absolute tolerance for the adaptive procedure to compute exponentials (for 'expm_adapt')
- *RelTol*: float; relative tolerance for the adaptive procedure to compute exponentials (for 'expm_adapt')
- *MaxLie*: integer; maximum number of Lie transforms to be performed to eliminate the non-resonant part of the perturbation 
####
- *TolMax*: float; value of the norm of the Hamiltonian for divergence
- *TolMin*: float; value of the norm of the Hamiltonian for convergence 
####
- *Precision*: integer; 32, 64 or 128; precision of calculations (default=64)
- *NormChoice*: string; 'sum', 'max', 'Euclidean', 'Analytic'; choice of Hamiltonian norm 
- *NormAnalytic*: float; parameter of norm 'Analytic'
####
- *SaveData*: boolean; if True, the results are saved in a `.mat` file 
- *PlotResults*: boolean; if True, the results are plotted right after the computation
- *Parallelization*: tuple (boolean, int); True for parallelization, int is the number of cores to be used (all of them: int='all')
####
---
## Error codes
- `0`: all transformations have been properly computed (no error)
- `1`: one of the Lie transforms is not accurately computed
- `2`: the series of canonical transformations to eliminate the non-resonant part of the Hamiltonian is diverging
- `-2`: the series of canonical transformations to eliminate the non-resonant part of the Hamiltonian does not converge or diverge (*MaxLie* iterations reached)
- `3`: the iterates of the RG map on the initially generated Hamiltonian *H*<sub>1</sub> diverge (*H*<sub>1</sub> is above the critical surface)
- `-3`: the iterates of the RG map on the initially generated Hamiltonian *H*<sub>2</sub> converge (*H*<sub>2</sub> is below the critical surface)
- `4`: the step in the adaptive step-size computation of the Lie transform ('expm_adapt') is below the minimum defined step size (*MinStep*)
---

References: 
- C. Chandre, H.R. Jauslin, *Renormalization-group analysis for the transition to chaos in Hamiltonian systems*, [Physics Reports](https://doi.org/10.1016/S0370-1573(01)00094-1) 365, 1 (2002)
```bibtex
@article{chandre2002,
         title = {Renormalization-group analysis for the transition to chaos in Hamiltonian systems},
         author = {C. Chandre and H.R. Jauslin},
         journal = {Physics Reports},
         volume = {365},
         number = {1},
         pages = {1-64},
         year = {2002},
         issn = {0370-1573},
         doi = {https://doi.org/10.1016/S0370-1573(01)00094-1},
         url = {https://www.sciencedirect.com/science/article/pii/S0370157301000941}, 
}
```
- A.P Bustamante, C. Chandre, *Numerical computation of critical surfaces for the breakup of invariant tori in Hamiltonian systems*, [SIAM Journal on Applied Dynamical Systems](https://doi.org/10.1137/21M1448501) 22, 483 (2023)
```bibtex
@article{bustamante2023,
      title = {Numerical computation of critical surfaces for the breakup of invariant tori in Hamiltonian systems}, 
      author = {Adrian P. Bustamante and Cristel Chandre},
      journal = {SIAM Journal on Applied Dynamical Systems},
      volume = {22},
      number = {1},
      pages = {483-500},
      year = {2023},
      issn = {1536-0040},
      doi = {https://doi.org/10.1137/21M1448501},
}
```
For more information: <cristel.chandre@cnrs.fr>
