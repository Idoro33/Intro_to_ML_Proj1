import numpy as np

def LeastSquares(X,y):
  theta = (np.linalg.inv( X.T @ X)) @ X.T @ y;
  '''
    Calculates the Least squares solution to the problem X*theta=y using the least squares method
    :param X: numpy input matrix, size [N,m+1] (feature 0 is a column of 1 for bias)
    :param y: numpy input vector, size [N]
    :return theta = (Xt*X)^(-1) * Xt * y: numpy output vector, size [m+1]
    N is the number of samples and m is the number of features=28
  '''
  return ((np.linalg.inv( X.T @ X)) @ X.T @ y)

def classification_accuracy(model,X,s):
  s_hat_cl = model.predict(X)
  correct = np.sum (s-s_hat_cl == 0)
  
  '''
    calculate the accuracy for the classification problem
    :param model: the classification model class
    :param X: numpy input matrix, size [N,m]
    :param s: numpy input vector of ground truth labels, size [N]
    :return: accuracy of the model = (correct classifications)/(total classifications) type float
    N is the number of samples and m is the number of features=28
  '''
  return correct/len(s)

def linear_regression_coeff_submission():
  '''
    copy the values from your notebook into a list in here. make sure the values
    seperated by commas
    :return: list of coefficiants for the linear regression problem. length 28
  '''
  return [-0.04038233,  0.03042779, -0.01722717,  0.02534981, -0.01113784, -0.03020349,
  0.06334342, -0.00999021,  0.02151976, -0.02063133,  0.01890982, -0.00796484,
  0.09140924,  0.13211249,  0.78794593,  0.03345455,  0.02035103,  0.01574533,
  0.02644344,  0.00290931,  0.0267272,   0.02121786, -0.00691334, -0.02140188,
 -0.05292936,  0.02261744, -0.01023402, -0.02289688]


def linear_regression_intrcpt_submission():
  '''
    copy the intercept value from your notebook into here.
    :return: the intercept value. type float
  '''
  return -2.5200652098372697e-16

def classification_coeff_submission():
  '''
    copy the values from your notebook into a list in here. make sure the values
    seperated by commas
    :return: list of list of coefficiants for the classification problem.  length 28
  '''
  return [[-1.74979938e-01,-2.88150208e-01,4.11795364e-01,-4.33224333e-04,
  -9.57478769e-02,-7.61864252e-01,8.43717116e-02,1.58639351e-02,
  -5.31541935e-03,5.48801426e-01,-7.52879584e-01,-1.31654019e-01,
   2.04930850e-01,1.08465193e+00,2.40775747e+00,-3.81185019e-01,
  -3.20052587e-02,-1.67543785e-01,-9.26047416e-03,-1.72494593e-01,
  -3.40704476e-02,1.02158329e-02,-2.54742439e-01,-2.21422019e-01,
  -4.20189539e-01,5.55609126e-02,1.63955753e-01,3.05440024e-01]]

def classification_intrcpt_submission():
  '''
    copy the intercept value from your notebook into here.
    :return: list with the intercept value. length 1
  '''
  return [0.14190316]

def classification_classes_submission():
  '''
    copy the classes values from your notebook into a list in here. make sure the values
    seperated by commas
    :return: list of classes for the classification problem. length 2.
  '''
  return [0,1]