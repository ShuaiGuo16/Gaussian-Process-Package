## Gaussian Process Python Package

This is an on-going project aiming to develop a comprehensive Gaussian Process
Python package, which facilitates scikit-learn style of training and exploiting
a Gaussian Process model.

## Training a Gaussian Process interpolator:
 - Supported trends: 'Const', 'Linear', 'Quadratic', 'Custom';
 - Supported kernels: 'Gaussian', 'Matern-3_2', 'Matern-5_2', 'Cubic';
 - Efficient model training: implemented Adjoint method to accelerate
 global optimization;
 - Automatically draw realizations from the posterior distribution of the trained
 Gaussian Process model;
 - Integrated with Scikit-Learn to perform cross-validation,
 feature transformation, etc.;
 - Implemented fast approximation of leave-one-out cross-validation error;
 - **Active Learning**:
    - 'EPE' --> maximum expected prediction error learning;
    - 'U' --> minimum classification error learning;

## Training a Gaussian Process Regressor:
 - Supported trends: 'Const', 'Linear', 'Quadratic', 'Custom';
 - Supported kernels: 'Gaussian', 'Matern-3_2', 'Matern-5_2', 'Cubic';
 - Automatical estimation of noise variance;
 - Posterior sampling;
 - Integration with Scikit-Learn;
