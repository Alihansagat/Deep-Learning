import numpy as np

# 1D array
np1 = np.array([1,2,3,4,5,6])
print(np1)
for x in np1:
    print(x)
    
# 2D array
np2 = np.array([[1,2,3], [4,5,6]])
print(np2)
for x in np2:
    for y in x:
        print(y)
        
# 3D array
np3 = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
print(np3)
print(np3.shape)
for x in np3:
    for y in x:
        for j in y:
            print(j)
            
            
#! Using np.nditer()
print("By using np.nditer()")

for x in np.nditer(np3):
    print(x)
    
print("for 2D array")
for x in np.nditer(np2):
    print(x)