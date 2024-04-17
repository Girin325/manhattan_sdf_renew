import numpy as np

test = np.load('data/scannet/0050_00/depth_colmap/0.npy')
print(test)
print(test.shape)
np.savetxt('0_npy.txt', test, fmt='%f', delimiter=',')
