## Gaussian Process Python Package

This is an on-going project aiming to develop a comprehensive Gaussian Process
Python package, which facilitates scikit-learn style of training and exploiting
a Gaussian Process model.

Currently, this package supports:

-Training a Gaussian Process interpolator:

 - Various trend type: 'Const', 'Linear', 'Quadratic', 'Custom';
 - Various kernel type: 'Gaussian', 'Matern-3_2', 'Matern-5_2', 'Cubic';
 - Efficient model training: implemented Adjoint method to accelerate
 global optimization;
 - Automatically draw realizations from the posterior distribution of the trained
 Gaussian Process model;
 -Integrated with Scikit-Learn to perform cross-validation,
 feature transformation, etc.;
 - Implemented fast approximation of leave-one-out cross-validation error;
