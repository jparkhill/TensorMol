#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <array>
#include <vector>

// inline(code, ['xyz', 'max_near', 'etype', 'dist_cut', 'elements', 'times', 'maxdisp'],headers=['<math.h>', '<algorithm>', '<cstdlib>','<iostream>', '<array>', '<vector>'] 


 static PyObject*  Make_CM (PyObject *self, PyObject  *args) { 
   
      PyObject *elements;
      PyArrayObject*  molxyz;
      double a,b;
      if (!PyArg_ParseTuple(args, "O!dd", 
         	&PyList_Type, &elements, &a, &b))  return NULL;
      int size = PyList_Size(elements);
      for (int i = 0;  i < size; i++)
	  molxyz  = (PyArrayObject*)((PyObject*)PyList_GetItem(elements,i));      
      npy_intp* Nxyz = molxyz->dimensions;
      int natom = Nxyz[0];
      std::cout<<natom<<" "<<a<<"  "<<b<<std::endl;
      return (PyObject*)molxyz;
}


static PyMethodDef CMMethods[] =
{
     {"Make_CM", Make_CM, METH_VARARGS,
         "Make_CM method"},
     {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
initMake_CM(void)
{
     (void) Py_InitModule("Make_CM", CMMethods);
     /* IMPORTANT: this must be called */
     import_array();
}
