import Make_CM,numpy


a=[]
b=numpy.zeros((3,10))
c=numpy.zeros((5,6))
a.append(b)
a.append(c)
A=40
B=30
print Make_CM.Make_CM(a,A,B)
