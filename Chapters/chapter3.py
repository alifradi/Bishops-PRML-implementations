import sys 
import os 
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)


from tools.basisfunctions import features
from tools.Linearmodels import LinearRegression_LS, RidgeRegression
import matplotlib.pyplot as plt
import numpy as np


# Basis Functions

x = np.linspace(-1,1,5)
Phi = features("Polynomial",4).fit(x)
Phi

x = np.linspace(-1,1,5)
PhiG = features("Gaussian",4).fit(x,0.1)
PhiG


x = np.linspace(-1,1,5)
PhiS = features("Sigmoidal",4).fit(x,10)
PhiS


#%%
def plot_columns_with_latex(matrix, x, title):
    """Generate plots for basis functions

    Args:
        matrix (2D np array): Phi
        x (np array): data
        title (str): main title
    """
    num_columns = matrix.shape[1]
    colors = plt.cm.viridis(np.linspace(0, 1, num_columns))
    plt.figure(figsize=(12, 6)) 
    for i in range(num_columns): 
        plt.plot(x, matrix[:, i], color=colors[i]) 
        plt.title(title, fontsize=14)
        plt.legend(loc="best", fontsize=12)
    plt.show() 


# Least Square Linear Regression

# %%
from tools.Linearmodels import LinearRegression_LS
# train
x_train = np.linspace(-4,5,100)
np.random.shuffle(x_train)
y_train = np.sin(x_train)+np.random.normal(0,0.2,x_train.shape)
Phi_X_train = features("Sigmoidal",10).fit(x_train,2)
leastSquare_linReg = LinearRegression_LS()
leastSquare_linReg.fit(Phi_X_train, y_train)
# Testing
x_test = np.linspace(-4,5,30)
Phi_X_test = features("Sigmoidal",10).fit(x_test,2)
y_predicted = leastSquare_linReg.predict(Phi_X_test)
y_test = np.sin(x_test)
# plotting
plt.scatter(x_train, y_train, facecolor="none", edgecolor="b", s=50, label="training data")
plt.plot(x_test, y_test, label="$\sin(2\pi x)$")
plt.plot(x_test, y_predicted, label="prediction")
plt.fill_between(
    x_test, y_predicted - leastSquare_linReg.var_w, y_predicted + leastSquare_linReg.var_w,
    color="orange", alpha=0.5, label="std.")
plt.legend()
plt.show()





# Least Square Linear Regression with Rodge penality

#%%
from tools.Linearmodels import RidgeRegression
# train
x_train = np.linspace(-4,5,100)
np.random.shuffle(x_train)
y_train = np.sin(x_train)+np.random.normal(0,0.2,x_train.shape)
Phi_X_train = features("Sigmoidal",10).fit(x_train,2)

Ridgemodel = RidgeRegression(alpha =0.001)
Ridgemodel.fit(Phi_X_train,y_train)

# Testing
x_test = np.linspace(-4,5,30)
Phi_X_test = features("Sigmoidal",10).fit(x_test,2)
y_predicted = Ridgemodel.predict(Phi_X_test)
y_test = np.sin(x_test)
plt.scatter(x_train, y_train, facecolor="none", edgecolor="b", s=50, label="training data")
plt.plot(x_test, y_test, label="$\sin(2\pi x)$")
plt.plot(x_test, y_predicted, label="prediction")
plt.legend()
plt.show()



# Bayesian Linear Regression

# %%
np.random.seed(314)
m = 4
x = np.linspace(-4,5,100)
y = np.sin(x)+np.random.normal(0,0.1,x.shape)
Phi = features("Polynomial",m).fit(x)
alpha = 1.2
beta  = 1
S0 = np.eye(m) * alpha
m0 = np.array([0]*Phi.shape[1])
SN = np.linalg.inv(np.linalg.inv(S0)+beta*Phi.T@Phi)
mN = SN@(np.linalg.inv(S0)@m0+beta*Phi.T@y)

x_test = np.linspace(-4,5,100)
Phi_X_test = features("Polynomial",m).fit(x_test)
n_samples = 20
# We create an instance of our random vector w
from scipy.stats import multivariate_normal
w = multivariate_normal(mean=mN.ravel(), cov=SN)
w_sample = w.rvs(n_samples)
y_test_sample = Phi_X_test @ w_sample.T
plt.plot(x_test, y_test_sample, c="tab:gray", alpha=0.5, zorder=1)
plt.scatter(x, y, c="tab:red", zorder=2)
plt.title("Posterior Samples", fontsize=15);
# %%
