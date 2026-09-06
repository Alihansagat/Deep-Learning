import numpy as np

np1 = np.array([1, 2, 3, 4, 5])

# Return 2 3 4 
print(np1[1:4])

# Return negative slices 3 4 5
print(np1[-3: ])

# Steps on the entire array
print(np1[::2])

#! Slice a 2D array
np2 = np.array([[1, 2, 3], [4, 5, 6]])

# Pull out a single item
print(np2[1,2])

#Slicing. 1 2
print(np2[:1,:2])

# Slicing from both array
print(np2[:2,:2])