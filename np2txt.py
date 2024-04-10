import numpy as np

res = np.load('results/GEN_Ours_airplane_1680779495/out.npy')

j=0
for i in res:
    f = 'res/out/'+str(j)+'.txt'
    txt = np.savetxt(f,i)
    j +=1