import numpy as np

#Filtering the arrray with boolean indexing

np1 = np.array([1,2,3,4,5])

x = [True, True, False, False, False]

print(f'Original array -> {np1}')
print(f'Boolean array -> {x}')
print(f'Filtering by Boolean -> {np1[x]}')


# Otherwise 

filtering = []
for x in np1:
    if x % 2 == 0:
        filtering.append(True)
    else: 
        filtering.append(False)

print(f'Original array -> {np1}')
print(f'Boolean array -> {filtering}')
print(f'Filtering by Boolean -> {np1[filtering]}')


# Shortcut

filtering = np1 % 2 != 0
print(f'Original array -> {np1}')
print(f'Boolean array -> {filtering}')
print(f'Filtering by Boolean -> {np1[filtering]}')