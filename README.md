# <p align="center">Python-Library-for-Optimization</p>
<p align="center">
  <img src="https://img.shields.io/badge/conda-4.12.0-orange" width="10%" height="10%">
  <img src="https://img.shields.io/badge/Python-3.9.12-green" width="10%" height="10%">
  <img src="https://img.shields.io/badge/MIT_License-blue" width="8%" height="8%">
</p>

## A Python package integrating around ten unconstrained optimization algorithms, inclusive of 2D/3D visualizations for comparative analysis, and incorporated matrix operations.

This project, a component of the Numerical Analysis & Optimization course at [ENSIAS](https://fr.wikipedia.org/wiki/%C3%89cole_nationale_sup%C3%A9rieure_d%27informatique_et_d%27analyse_des_syst%C3%A8mes), [Mohammed V University](https://en.wikipedia.org/wiki/Mohammed_V_University) arose from Professor [M. Naoum](http://ensias.um5.ac.ma/professor/m-mohamed-naoum)'s suggestion to create a Python package encapsulating algorithms practiced during laboratory sessions. It focuses on about ten unconstrained optimization algorithms, offering 2D and 3D visualizations, enabling a performance comparison in computational efficiency and accuracy. Additionally, the project encompasses matrix operations such as inversion, decomposition, and solving linear systems, all integrated within the package containing the lab-derived algorithms.

<p align="center">
  <img src="https://github.com/Heyyassinesedjari/Python-Library-for-Optimization/assets/94799575/15197d66-cc2a-4149-8e21-db247ea31d25" width="50%" height="50%">
</p>
<p align="center">
  GIF Source: https://www.nsf.gov/news/mmg/mmg_disp.jsp?med_id=78950&from=](https://aria42.com/blog/2014/12/understanding-lbfgs
</p>

Every part of this project is sample code which shows how to do the following:
- Implementation of one-variable optimization algorithms, encompassing fixed_step, accelerated_step, exhaustive_search, dichotomous_search, interval_halving, Fibonacci, golden_section, Armijo_backward, and Armijo_forward.
- Incorporation of various one-variable optimization algorithms, including gradient descent, Gradient conjugate, Newton, quasi_Newton_dfp, stochastic gradient descent.
- Visualization of each iteration step for all algorithms in 2D, contour, and 3D formats.
- Conducting a comparative analysis focusing on their runtime and accuracy metrics.
- Implementation of matrix operations such as inversion (e.g., Gauss-Jordan), decomposition (e.g., LU, Choleski), and solutions for linear systems (e.g., Gauss-Jordan, LU, Choleski).


  ## High-level functional explanation

 
<p align="center">
  Simple quadratic objectif function for isllustration purpose <br>
  
  <img src="https://github.com/Heyyassinesedjari/Python-Library-for-Optimization/assets/94799575/3bb5cf17-e95b-4fde-8880-ad04c533cf02" width="50%" height="50%">
  <img src="https://github.com/Heyyassinesedjari/Python-Library-for-Optimization/assets/94799575/cd451cdd-2009-4702-8924-e85b58754189" width="50%" height="50%"><br>
</p>

<p align="center">
  Parameter Initialization:<br>

  <img src="https://github.com/Heyyassinesedjari/Python-Library-for-Optimization/assets/94799575/b0c201d3-8a75-42c9-97df-f989a124f8a8" width="50%" height="50%">
  <img src="https://github.com/Heyyassinesedjari/Python-Library-for-Optimization/assets/94799575/1ca2e87e-b91a-4d43-acee-41f5f7dae9c6" width="50%" height="50%"><br>
</p>

### One Var Algorithms
#### Fixed Step
<p align="center">
  Api call<br>
  <img src="https://github.com/Heyyassinesedjari/Python-Library-for-Optimization/assets/94799575/c72cd089-fbb6-462f-9dd6-b5ddaf323732" width="50%" height="50%"> <br>
  visulize each step<br>
  <img src="https://github.com/Heyyassinesedjari/Python-Library-for-Optimization/assets/94799575/5815e93a-3c54-4b70-a331-3ec13983e829" width="50%" height="50%">
</p>

#### Accelerated Step
<p align="center">
  Api call<br>
  <img src="https://github.com/Heyyassinesedjari/Python-Library-for-Optimization/assets/94799575/7ddf4122-b877-4808-a415-ac373630e535" width="50%" height="50%"> <br>
  visulize each step<br>
  <img src="https://github.com/Heyyassinesedjari/Python-Library-for-Optimization/assets/94799575/237ae937-764b-4205-a50b-05ca82463c77" width="50%" height="50%">
</p>


