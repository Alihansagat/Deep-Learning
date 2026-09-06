import numpy as np 

# 2D array
np1 = np.array([[1,2,3,4,5,12],[6,7,8,9,10,11]])
print(np1.shape)

#! Reshape 2D
np2 = np1.reshape(6,2)
print(np2)
print(np2.shape)

#! Reshape 3D
np3 = np1.reshape(2,3,2)    # 2 blocks, each one has 3 rows by 2 columns 
print(np3)
print(np3.shape)

#! Flatten to 1D
np4 = np3.reshape(-1)       #when we want to reshape our 3D or 2D array to 1D we use -1
print(np4)
print(np4.shape)