import numpy as np
from typing import Union

class LinearRegression_LS():    
    def fit(self, Phi_X, T)-> Union[np.ndarray, float]:
        """_summary_

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
