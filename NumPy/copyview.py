import numpy as np

# Copy vs View
# Copy is a copy of our array, View is a like copy but its still connected to our array

np1 = np.array([0, 1, 2, 3, 4, 5])

#! Create a view
np2 = np1.view()

print(f'Original NP1 -> {np1}')
print(f'NP2 as a view -> {np2}')

np1[0] = 41
print(f'NP1 after changing -> {np1}')
print(f'NP2 as a view -> {np2}')

#! view as a same array and its will be stil connected to our array
#! if we will change the something from our original array, in the view it also will change, and same thing with view

#! Create a copy
np3 = np1.copy()
print(f'Original NP1 -> {np1}')
print(f'Original NP3 -> {np3}')

np1[0] = 0

print(f'NP1 after changing -> {np1}')
print(f'NP3 as a copy -> {np3}')

#! copy as a different array, it just copying the everything from our original array, and it goona be like independent array by itself
