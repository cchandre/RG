# Renormalization group (RG) for the break-up of invariant tori in Hamiltonian flows

- [`RG_dict.py`](https://github.com/cchandre/RG/blob/main/RG_dict.py): to be edited to change the parameters of the RG computation (see below for a dictionary of parameters)

- [`RG.py`](https://github.com/cchandre/RG/blob/main/RG.py): contains the RG classes and main functions defining the RG map

- [`RG_modules.py`](https://github.com/cchandre/RG/blob/main/RG_modules.py): contains the methods to execute the RG map

Once [`RG_dict.py`](https://github.com/cchandre/RG/blob/main/RG_dict.py) has been edited with the relevant parameters, run the file as 
```sh
python3 RG.py
```

___
##  Parameter dictionary

- *Method*: 'iterates', 'surface', 'region', 'line'; choice of method
- *Iterates*: integer; number of iterates to compute in `compute_iterates()`
- *Nxy*: integer; number of points in the (*x*,*y*) figures for `compute_surface()` and `compute_region()`
- *DistSurf*: float; distance of approach for the computation of critical values
####
- *N*: *n*x*n* integer matrix with determinant ±1
- *omega0*: array of *n* floats; frequency vector of the invariant torus; should be an eigenvector of `N.transpose()` 
- *Omega*: array of *n* floats; vector of the perturation in action
- *K*: 2-dimensional tuple of integers; wavevectors (j,k_1,...k_n) of the perturbation 
- *AmpInf*: array of *len(K)* floats; minimal amplitudes of the perturbation 
- *AmpSup*: array of *len(K)* floats; maximum amplitudes of the perturbation
- *CoordLine*: 1d array of floats; min and max values of the amplitudes of the potential used in `compute_line()`   
- *ModesLine*: tuple of 0 and 1; specify which modes are being varied (1 for a varied mode)     
- *DirLine*: 1d array of floats; direction of the one-parameter family used in `compute_line()` 
####
- *L*: integer; truncation in Fourier series (angles) 
- *J*: integer; truncation in Taylor series  (actions) 
####
- *ChoiceIm*: 'AK2000', 'K1999', 'AKW1998'; definition of *I-* 
- *Sigma*: float; definition of *I-*
- *Kappa*: float; definition of *I-*
####
- *CanonicalTransformation*: 'Lie', 'Lie_scaling', 'Lie_adaptive'; method to compute the canonical Lie transforms 
- *LieSteps*: integer; number of steps in the scaling and squaring procedure to compute exponentials (for 'Lie_scaling')
- *MinStep*: float; minimum value of the steps in the adaptive procedure to compute exponentials (for 'Lie_adaptive')
- *AbsTol*: float; absolute tolerance for the adaptive procedure to compute exponentials (for 'Lie_adaptive')
- *RelTol*: float; relative tolerance for the adaptive procedure to compute exponentials (for 'Lie_adaptive')
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

References: 
- C. Chandre, H.R. Jauslin, *Renormalization-group analysis for the transition to chaos in Hamiltonian systems*, [Physics Reports](https://doi.org/10.1016/S0370-1573(01)00094-1) 365, 1 (2002)
```bibtex
@article{chandre2002,
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
- A.P Bustamante, C. Chandre, *Numerical computation of critical surfaces for the breakup of invariant tori in Hamiltonian systems*, [arXiv:2109.12235](https://arxiv.org/abs/2109.12235)
```bibtex
@misc{bustamante2021,
      title={Numerical computation of critical surfaces for the breakup of invariant tori in Hamiltonian systems}, 
      author={Adrian P. Bustamante and Cristel Chandre},
      year={2021},
      eprint={2109.12235},
      archivePrefix={arXiv},
      primaryClass={math.DS}
}
```
For more information: <cristel.chandre@univ-amu.fr>
