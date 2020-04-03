import numpy as np

a= np.array([[1,2,3], [4,5,6]])
b= np.array([[10,20,30], [40, 50, 60]])
timesteps=np.array([[0.2, 0.3 ,0.4], [0.3, 0.4, 0.5]])



print(a[timesteps==0.4])