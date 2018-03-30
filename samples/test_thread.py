import numpy as np
import multiprocessing as mp
import tensorflow as tf
import time

def MakeWork():
	tmp = np.array(range(np.power(2,22)),dtype=np.float64)
	tmp = np.sqrt(tmp)
	tmp=tmp*tmp
	tmp=tmp.reshape((np.power(2,11),np.power(2,11)))
	print (tmp.shape)
	tmp=np.dot(tmp,tmp)
	np.linalg.eig(tmp)
	print ("Work Complete")

t0 = time.time()
threads = []
for i in range(8):
	t = mp.Process(target=MakeWork)
	threads.append(t)
	t.start()
for t in threads:
	t.join()

t1 = time.time()
for i in range(8):
	MakeWork()
t2 = time.time()

print ("Times: ", t1-t0, t2-t1)
print ("Speedup: ", (t2-t1)/(t1-t0))
