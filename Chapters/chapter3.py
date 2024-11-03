import sys 
import os 
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)


from tools.basisfunctions import features
import matplotlib.pyplot as plt
import numpy as np


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

# %%
