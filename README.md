## Gaussian Process Python Package

This project aims to develop a comprehensive Gaussian Process
Python package, which facilitates scikit-learn style of training and exploiting
a Gaussian Process model.

## Gaussian Process Class

The folder **GaussianProcess** contains the code to train and exploit various types of Gaussian Process models. Specifically, a user can choose the following functionalities. 

### GPInterpolator Class:

This class deals with using Gaussian Process model to interpolate functions.
 - Supported trends: 'Const', 'Linear', 'Quadratic', 'Custom';
 - Supported kernels: 'Gaussian', 'Matern-3_2', 'Matern-5_2', 'Cubic';
 - Efficient model training: implemented Adjoint method to accelerate
 global optimization (Multi-start approach);
 - Predict-only mode: user can manually specify model parameters, thus eliminating the need to re-train the model; 
 - Automatically draw realizations from the posterior distribution of the trained
 Gaussian Process model;
 - Integrated with Scikit-Learn to perform cross-validation,
 feature transformation, etc.;
 - Implemented fast approximation of leave-one-out cross-validation error;
 - **Active Learning**:
    - 'EPE' --> maximum expected prediction error learning;
    - 'U' --> minimum classification error learning;

### GPRegressor Class:

This class deals with using Gaussian Process model to approximate functions using noisy observations.
 - Supported trends: 'Const', 'Linear', 'Quadratic', 'Custom';
 - Supported kernels: 'Gaussian', 'Matern-3_2', 'Matern-5_2', 'Cubic';
 - Predict-only mode: user can manually specify model parameters, thus eliminating the need to re-train the model;
 - Automatical estimation of noise variance;
 - Posterior sampling;
 - Integration with Scikit-Learn;

### GEGP Class:

This class deals with training and exploiting gradient-enhanced Gaussian Process model.
 - Supported trends: 'Const';
 - Supported kernels: 'Gaussian';
 - User can feed gradients of output to improve the model accuracy;
 - Predict gradients: analytically approximate the output gradients at test locations;
 - Predict-only mode: user can manually specify model parameters, thus eliminating the need to re-train the model;
 - Integration with Scikit-Learn;
 
## Gaussian Process Tutorials

In addition to the core code, this project also provides a total of 6 tutorials to help user understand how to use the current package to train/predict with Gaussian Process models.

### Tutorial 1: Gaussian Process Model for Interpolation

A walk-through of the functionalities of the developed package related to training and exploiting a Gaussian Process model for interpolation purposes.

### Tutorial 2: Gaussian Process Model for Regression

A walk-through of the functionalities of the developed package related to training and exploiting a Gaussian Process model for regression purposes.

### Tutorial 3: Gaussian Process Model with Active Learning

Train a Gaussian Process model using an active learning scheme based on maximizing the *expected prediction error*.

### Tutorial 4: Gaussian Process Model for Stability Margin Approximation

How to use active learning to make GP model particularly accurate in the vicinity of the stability margin.

### Tutorial 5: Gradient-Enhanced Gaussian Process Model

A walk-through of the functionalities of the developed package related to training and exploiting a gradient-enhanced Gaussian Process model for interpolation purposes.

### Tutorial 6: Gaussian Process Model with Multi-fidelity Learning

Train a multi-fidelity Gaussian Process model to aggregate training data with different fidelities.