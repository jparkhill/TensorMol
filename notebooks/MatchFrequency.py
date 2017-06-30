import numpy as np 
expv = np.loadtxt("adenine_exp.dat")
TMv = np.loadtxt("adenine_TM.dat")
found = []
nfound = []
for i in range(TMv.shape[0]):
    least_diff = float('inf')
    least_index = None
    for j in range(expv.shape[0]):
        if j in found:
            continue
    	if j not in found:
        	diff = abs(expv[j] - TMv[i])
        	if diff < least_diff:
            		least_diff = diff
            		least_index = j
    if least_index != None:
    	found.append(least_index)
for i in range(len(expv)):
    if i not in found:
        nfound.append(i)
print expv[found]
l = 0 
while l<2: 
    for i in range(len(found)):
        for j in range(i+1, len(found)):
            for k in range(len(nfound)):
                diff = abs(TMv[i] - expv[found[i]]) + abs(TMv[j] - expv[found[j]])
                diff1 = abs(TMv[i] - expv[found[j]]) + abs(TMv[j] - expv[found[i]])
                if diff1 < diff:
                    l = 0 
                    found[i], found[j] = found[j], found[i]
                    diff = diff1
                diff2 = abs(TMv[i] - expv[found[i]]) + abs(TMv[j] - expv[nfound[k]])
                if diff2 < diff:
                    l = 0 
                    found[i], found[j] = found[i], nfound[k]
                    diff2 = diff
                diff3 = abs(TMv[i] - expv[nfound[k]]) + abs(TMv[j] - expv[found[j]])
                if diff3 < diff:
                    l = 0 
                    found[i], found[j] = nfound[k], found[j]
                    diff3 = diff
                diff4 = abs(TMv[i] - expv[found[j]]) + abs(TMv[j] - expv[nfound[k]])
                if diff4 < diff:
                    l = 0 
                    found[i], found[j] = found[j], nfound[k]
                    diff4 = diff
                diff5 = abs(TMv[i] - expv[nfound[k]]) + abs(TMv[j] - expv[found[i]])
                if diff5 < diff:
                    l = 0 
                    found[i], found[j] = nfound[k], found[i]
                    diff5 = diff
    l += 1
print expv[found]
print TMv