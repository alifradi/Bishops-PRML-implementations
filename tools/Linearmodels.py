import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from typing import Union

class LinearRegression_LS():    
    def fit(self, Phi_X, T)-> Union[np.ndarray, float]:
        """ fitting traditional L. Reg with least square

        Args:
            Phi_X (np.ndarray): Matrix of 2D applied basis functions on sampled training data with n augmented dimensions
            T (numpy array): vector of target observation from training data

        Returns:
            Union[np.ndarray, float]: coefficients of contribution to each augmented dimention to the target prediction, estimated variance using training data (basis functions applied to training data), computed Least Squared W* and training targets 
        """
        w          = np.linalg.pinv(Phi_X) @ T
        var_w      = np.mean(np.square(Phi_X@w-T))
        self.w     = w 
        self.var_w = var_w
        return w, var_w
    
    def predict(self, Phi_X)-> np.ndarray:
        """predict using Least Squared fit

        Args:
            Phi_X (np.ndarray): Matrix of 2D applied basis functions on sampled test data with n augmented dimensions

        Returns:
            np.ndarray: predicted targets
        """
        T = Phi_X @ self.w
        return T



class RidgeRegression():
    def __init__(self,alpha):
        """ Constructor for Ridge L.Reg

        Args:
            alpha (float): parameter to penalize SE function 
        """
        self.alpha = alpha
    def fit(self, Phi_X, T)-> Union[np.ndarray, float]:
        """fitting L. Reg with Ridge penality |t - X @ w| + alpha * |w|_2^2

        Args:
            Phi_X (np.ndarray): Matrix of 2D applied basis functions on sampled training data with n augmented dimensions
            T (numpy array): vector of target observation from training data

        Returns:
            Union[np.ndarray, float]: coefficients of contribution to each augmented dimention to the target prediction, estimated variance using training data (basis functions applied to training data), computed Least Squared W* and training targets 
        """
        w          = np.linalg.inv(self.alpha*np.eye(np.size(Phi_X,1))+Phi_X.T@Phi_X)@Phi_X.T@T
        var_w      = np.mean(np.square(Phi_X@w - T))
        self.var_w = var_w
        self.w     = w
        return w, var_w
    
    def predict(self, Phi_X)-> np.ndarray:
        """predict using Least Squared fit with Ridge penality

        Args:
            Phi_X (np.ndarray): Matrix of 2D applied basis functions on sampled test data with n augmented dimensions

        Returns:
            np.ndarray: predicted targets
        """
        T = Phi_X @ self.w
        return T



class BayesianLinearRegression:
    def __init__(self, alpha, beta):
        """ Initiate Bayesian L.Reg parameters

        Args:
            alpha (float): prior precision over w p(w|0,alpha.I)
            beta (float): model-likelihood precision t ~ W.T @ Phi(x) + eps, eps ~ Norm(0, 1/beta)
        """
        self.alpha = alpha
        self.beta = beta
        self.m0 = None
        self.S0 = None

    def Prior(self, Phi_X) -> tuple:
        """ Returns prior's parameters

        Args:
            Phi_X (np.ndarray): transformed x 

        Returns:
            tuple: m0, S0 such that p(w) = Norm(m0, S0) or p(w|0,alpha.I)
        """
        if self.m0 is not None and self.S0 is not None:
            return self.m0, self.S0   
        else:
            # Set the prior mean and covariance as per standalone code
            return np.array([0] * Phi_X.shape[1]), np.eye(Phi_X.shape[1]) * self.alpha

    def fit(self, Phi_X_train, T_train):
        """ Compute posterior parameters of p(w | T_train)

        Args:
            Phi_X_train (np.ndarray): transformed training data with Phi applied
            T_train (np.ndarray): training target vector
        """
        # Compute priors
        m0, S0 = self.Prior(Phi_X_train)
        
        # Compute posterior p(w|T_train)
        SN = np.linalg.inv(np.linalg.inv(S0) + self.beta * Phi_X_train.T @ Phi_X_train)
        mN = SN @ (np.linalg.inv(S0) @ m0 + self.beta * Phi_X_train.T @ T_train)
        
        # Store posterior mean and covariance
        self.m0 = mN
        self.S0 = SN

    def predict(self, Phi_X_test, n_samples) -> tuple:
        """Uses sampled weights to predict on testing data

        Args:
            Phi_X_test (np.ndarray): transformed test data with Phi applied 
            n_samples (int): number of samples for each weight associated to each feature

        Returns:
            tuple: 1- n_samples of predictions using the sampled weights and transformed test data
            tuple: 2- mean of predicted values over sampled weights
            tuple: 3- standard deviation of predicted values over sampled weights
        """
        # Sample weights from the posterior distribution
        w_sample = np.random.multivariate_normal(mean=self.m0, cov=self.S0, size=n_samples)
        
        
        # Predict using sampled weights
        y_hat = Phi_X_test @ w_sample.T
        y_std = np.sqrt(1/self.beta + np.einsum("ik,kj,ij->i",Phi_X_test,self.S0,Phi_X_test))
        return y_hat, y_hat.mean(axis=1), y_std









  
    
