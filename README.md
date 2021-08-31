# Renormalization group (RG) for the break-up of invariant tori in Hamiltonian flows

- [`RG_dict.py`](https://github.com/cchandre/RG/blob/main/RG_dict.py): to be edited to change the parameters of the RG computation (see below for a dictionary of parameters)

- [`RG.py`](https://github.com/cchandre/RG/blob/main/RG.py): contains the RG classes and main functions defining the RG map

- [`RG_modules.py`](https://github.com/cchandre/RG/blob/main/RG_modules.py): contains the methods to execute the RG map

Once [`RG_dict.py`](https://github.com/cchandre/RG/blob/main/RG_dict.py) has been edited with the relevant parameters, run the file as 
```sh
python3.8 RG.py
```

___
##  Parameter dictionary

- *Method*: 'iterates', 'surface', 'region'; choice of method
- *Iterates*: integer; number of iterates to compute in `compute_iterates()`
- *Nxy*: integer; number of points in the (*x*,*y*) figures for `compute_surface()` and `compute_region()`
- *DistSurf*: float; distance of approach for the computation of critical values
####
- *N*: *n*x*n* integer matrix with determinant ±1
- *omega0*: array of *n* floats; frequency vector of the invariant torus; should be an eigenvector of `N.transpose()` 
- *Omega*: array of *n* floats; vector of the perturation in action
- *K*: tuples of integers; wavevectors of the perturbation 
- *AmpInf*: array of floats; minimal amplitudes of the perturbation 
- *AmpSup*: array of floats; maximum amplitudes of the perturbation
####
- *L*: integer; truncation in Fourier series (angles) 
- *J*: integer; truncation in Taylor series  (actions) 
####
- *ChoiceIm*: 'AK2000', 'K1999', 'AKW1998'; definition of *I-* 
- *Sigma*: float; definition of *I-*
- *Kappa*: float; definition of *I-*
####
- *CanonicalTransformation*: 'Lie', 'Lie_scaling', 'Lie_adaptive'; method to compute the canonical Lie transforms 
- *LieSteps*: integer; number of steps in the scaling and squaring procedure to compute exponentials 
- *MinStep*: float; minimum value of the steps in the adaptive procedure to compute exponentials 
- *AbsTol*: float; absolute tolerance for the adaptive procedure to compute exponentials 
- *RelTol*: float; relative tolerance for the adaptive procedure to compute exponentials
####
- *TolMax*: float; value of Hamiltonian norm for divergence
- *TolMin*: float; value of Hamiltonian norm for convergence 
- *TolMinLie*: float; value of norm for convergence of Lie transforms 
- *MaxLie*: integer; maximum number of elements in Taylor series of exponentials
- *MaxIterates*: integer; maximum number of iterates for convergence/divergence 
####
- *Precision*: 32, 64 or 128; precision of calculations (default=64)
- *NormChoice*: 'sum', 'max', 'Euclidean', 'Analytic'; choice of Hamiltonian norm 
- *NormAnalytic*: float; parameter of norm 'Analytic'
####
- *SaveData*: boolean; if True, the results are saved in a `.mat` file 
- *PlotResults*: boolean; if True, the results are plotted right after the computation
- *Parallelization*: tuple (boolean, int); True for parallelization, int is the number of cores to be used (all of them: int='all')
####
---

Reference: C. Chandre, H.R. Jauslin, *Renormalization-group analysis for the transition to chaos in Hamiltonian systems*, [Physics Reports](https://doi.org/10.1016/S0370-1573(01)00094-1) 365, 1 (2002)

```bibtex
@article{CHANDRE20021,
  title = {Renormalization-group analysis for the transition to chaos in Hamiltonian systems},
  journal = {Physics Reports},
  volume = {365},
  number = {1},
  pages = {1-64},
  year = {2002},
  issn = {0370-1573},
  doi = {https://doi.org/10.1016/S0370-1573(01)00094-1},
  url = {https://www.sciencedirect.com/science/article/pii/S0370157301000941},
  author = {C. Chandre and H.R. Jauslin},
}
```
For more information: <cristel.chandre@univ-amu.fr>
