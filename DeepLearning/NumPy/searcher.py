import numpy as np

np1 = np.array([1,2,3,4,5,6,7,8,3])

x = np.where(np1 == 3)
print(np1)

print(x)        # return the index with dtype informations
print(x[0])     # to return the exact indexes, we will write the [0]

# to take the tuple of this elements what we are searching
print(np1[x[0]])

# Return the even number's index
y = np.where(np1 % 2 == 0)

print(np1)
print(f"Even number's indexces -> {y[0]}")
print(f'The even numbers -> {np1[y[0]]}')

