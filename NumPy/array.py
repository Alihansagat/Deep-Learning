import numpy as np
# Numpy - Numeric Python
# data type of numpy - ndarray = n-dimensional array

np1 = np.array([0, 1, 2, 3, 4, 5])

print(np1.shape)

np2 = np.arange(10) #! .arange() can create matrix with signed range
print(np2)

np3 = np.arange(0, 10, 2)
print(np3)

np4 = np.zeros(10) #! .zeros() create the array with signed number of zeros
print(np4)

#! 2D array
np5 = np.zeros((2, 5))
print(np5)

np6 = np.full((5), 2) #! fills the array with 2nd attribute 
print(np6)

#! 2D full
np7 = np.full((2, 5), 1)
print(np7)