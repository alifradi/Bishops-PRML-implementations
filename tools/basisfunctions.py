import numpy as np

class features():
    def __init__(self, type, degree):
        """Pick type of basis functions
        Args:
            type (string): Polynomial, or Gaussian or Sigmoidal
            degree (Int): number of features in a matrix including bais
        """
        self.type   = type
        self.degree = degree

    def Polynomial(self, x)-> np.ndarray:
        """returns a matrix of polynomial features 

        Args:
            x (numpy array): sampled data

        Returns:
            np.ndarray: Phi matrix generated from polynomial features
        """
        Phi = [[1]*x.shape[0]]
        for i in range(1, self.degree):
            Phi.append(x**i)
        return np.array(Phi).transpose()
    
    def Gaussian(self,x,var=0.1)-> np.ndarray:
        """returns a matrix of gaussian features 

        Args:
            x (numpy array): sampled data
            var (float, optional): variance of gaussian radial basis functions. Defaults to 0.1.

        Returns:
            np.ndarray: Phi matrix generated from gaussian features
        """
        means = np.linspace(-1,1, self.degree)
        Phi = [[1]*x.shape[0]]
        for mu in means:
            Phi.append(np.exp(-0.5*(x-mu)**2/var))
        return np.array(Phi).transpose()
    
    def Sigmoid(self,x,coef=10)-> np.ndarray:
        """returns a matrix of gaussian features 

        Args:
            x (numpy array): sampled data
            coef (float, optional): coefficient of sigmoidal basis functions. Defaults to 10.

        Returns:
            np.ndarray: Phi matrix generated from sigmoidal features
        """
        means = np.linspace(-1,1, self.degree)
        Phi = [[1]*x.shape[0]]
        for mu in means:
            Phi.append(np.tanh((x-mu)*coef*0.5)*0.5+0.5)
        return np.array(Phi).transpose()
    
    def fit(self,*args):

        if self.type == "Polynomial":
            if len(args) != 1:
                raise ValueError("Polynomial feature requires data as numpy array, expected 1 arg but found " + str(len(args))+" arguments")
            return self.Polynomial(args[0])
        
        if self.type == "Gaussian":
            if len(args) != 2:
                raise ValueError("Gaussian feature requires data as numpy array and variance basis functions as int, expected 2 args but found " + str(len(args))+" arguments")
            return self.Gaussian(args[0], args[1])
        
        if self.type == "Sigmoidal":
            if len(args) != 2:
                raise ValueError("Sigmoidal feature requires data as numpy array and degree of polynomial basis functions as int, expected 2 args but found " + str(len(args))+" arguments")
            return self.Sigmoid(args[0], args[1])
        

