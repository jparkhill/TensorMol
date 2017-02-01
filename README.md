-------------------------------------------
# TensorMol 0.0
##A Statistical Model of Molecular Structure

### Authors:
	Kun Yao (kyao@nd.edu), John Herr
	John Parkhill (john.parkhill@gmail.com)
-------------------------------------------
### License: GPLv3
- By using this software you agree to the terms in COPYING
-------------------------------------------
### Acknowledgements:
 - Google Inc. (for TensorFlow),
 - Vol Lillenfeld Group (for GBD9)
 - Chan Group (for PySCF)
-------------------------------------------
### INSTALLATION:
- " cd C_API/ "
- " sudo python setup.py install "
-------------------------------------------
### USAGE:
 - Refer to commented examples in test.py
 - "python test.py"
-------------------------------------------
### REQUIREMENTS:
- Minimum Pre-Requisites: Python2.7x, TensorFlow
- Useful Pre-Requisites: CUDA7.5, PySCF
- To Train Minimally: ~100GB Disk 20GB memory
- To Train Realistically: 1TB Disk, GTX1070++
- To Evaluate: Normal CPU and 10GB Mem
- For volume rendering: CUDA 7.5+ or Mathematica
- " cd volumeRender; make "
- Or open /densities/PlotDens.nb
-------------------------------------------
### COMMON ISSUES
- nan during training due to bad checkpoints in /networks (clean.sh)
- Also crashes when reviving networks from disk.
