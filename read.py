import numpy as np
s = np.load('/Users/songda/Downloads/Terracotta_warriors_normal/terr_normal_2048_test_points.npy')
np.savetxt('/Users/songda/Downloads/Terracotta_warriors_normal/terr_normal_2048_train_points.txt',s[1000])
print(s[1000])