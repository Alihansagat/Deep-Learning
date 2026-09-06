import numpy as np

# Numerical
np1 = np.array([6,3,2,4,7,0])
print(np1)
print(np.sort(np1))

#cAlphabeticaly
np2 = np.array(["John", "Alex", "Simon", "Norma"])
print(np2)
print(np.sort(np2))

# Boolean
np3 = np.array([True, False, True, True])
print(np3)
print(np.sort(np3))

#! When we sorting our array, we are just copying our array not chaging the original array

# 2D array
np4 = np.array([[1,4,3,0], [8,3,5,0]])
print(np4)
print(np.sort(np4))