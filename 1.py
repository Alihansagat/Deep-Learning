import numpy as np
import matplotlib.pyplot as plt
import random

x = np.random.rand(15)

#z = w*x + b

w = np.random.rand(2, 15)

b = np.random.rand(2, 15)

z = w * x + b

# c = np.vectorize(z.any)


def relu(z):
    if z < 0:
        a = 0
    else:
        a = z
        
c = np.vectorize(relu(z))

    
print(c)
    
# print(b)

# c = w*x

# print(c)

# print(c.shape)


# print(b.shape)