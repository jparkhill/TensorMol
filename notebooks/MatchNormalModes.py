# qnm = q chem normal modes
# tnm = tensormol normal modes
import numpy as np 
qnm = np.load("adenine_nm.npy")
tnm = np.load("adenine_TM.dat")
found = []
nfound = []
for i in range(tnm.shape[0]):
    least_diff = float('inf')
    least_index = None
    for j in range(qnm.shape[0]):
        if j in found:
            continue
    	if j not in found:
        	diff = abs(qnm[j] - tnm[i])
        	if diff < least_diff:
            		least_diff = diff
            		least_index = j
    if least_index != None:
    	found.append(least_index)
for i in range(len(qnm)):
    if i not in found:
        nfound.append(i)
print qnm[found]
l = 0 
while l<2: 
    for i in range(len(found)):
        for j in range(i+1, len(found)):
            for k in range(len(nfound)):
                diff = abs(tnm[i] - qnm[found[i]]) + abs(tnm[j] - qnm[found[j]])
                diff1 = abs(tnm[i] - qnm[found[j]]) + abs(tnm[j] - qnm[found[i]])
                if diff1 < diff:
                    l = 0 
                    found[i], found[j] = found[j], found[i]
                    diff = diff1
                diff2 = abs(tnm[i] - qnm[found[i]]) + abs(tnm[j] - qnm[nfound[k]])
                if diff2 < diff:
                    l = 0 
                    found[i], found[j] = found[i], nfound[k]
                    diff2 = diff
                diff3 = abs(tnm[i] - qnm[nfound[k]]) + abs(tnm[j] - qnm[found[j]])
                if diff3 < diff:
                    l = 0 
                    found[i], found[j] = nfound[k], found[j]
                    diff3 = diff
                diff4 = abs(tnm[i] - qnm[found[j]]) + abs(tnm[j] - qnm[nfound[k]])
                if diff4 < diff:
                    l = 0 
                    found[i], found[j] = found[j], nfound[k]
                    diff4 = diff
                diff5 = abs(tnm[i] - qnm[nfound[k]]) + abs(tnm[j] - qnm[found[i]])
                if diff5 < diff:
                    l = 0 
                    found[i], found[j] = nfound[k], found[i]
                    diff5 = diff
    l += 1