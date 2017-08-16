# &#9658;TensorMol 0.1
![](newtitle.png)
-Title signature by Alex Graves' handwriting LSTM https://arxiv.org/abs/1308.0850

### Authors:
 Kun Yao (kyao@nd.edu), John Herr (jherr1@nd.edu),
 David Toth (dtoth1@nd.edu), John Parkhill (john.parkhill@gmail.com)

### Model Chemistries:
 - Behler-Parrinello
 - Many Body Expansion
 - Bonds in Molecules NN
 - Atomwise Forces
 - Inductive Charges

### Simulation Types:
 - Optimizations
 - Nudged Elastic Band
 - Molecular Dynamics (NVE,NVT Nose-Hoover)
 - Open/Periodic Boundary Conditions
 - Meta-Dynamics
 - Infrared spectra

### License: GPLv3
By using this software you agree to the terms in COPYING

### Installation:
 - Works on OSX, Ubuntu, and Windows subsystem for Linux.
```
git clone https://github.com/jparkhill/TensorMol.git
cd TensorMol
# If you are using python2x
sudo pip install -e .
# If you are using python3x
sudo pip3 install -e . 
python test.py
```

### Usage:
 - ```import TensorMol as tm```
 - TensorMol assumes a directory structure executing path which mirrors the git.
 - Please also refer to IPython notebooks in /notebooks.

### Sample Results
![](water.png)

- Water Trimer IR spectrum generated with david_testIR() in test.py
- The red lines are MP2(qchem) solid line is TensorMol's IR propagation.

### Requirements:
- Minimum Pre-Requisites: Python2.7x, TensorFlow
- Python3x support coming soon. 
- Useful Pre-Requisites: CUDA7.5, PySCF
- To Train Minimally: ~100GB Disk 20GB memory
- To Train Realistically: 1TB Disk, GTX1070++
- To Evaluate: Normal CPU and 10GB Mem

### Acknowledgements:
 - Google Inc. (for TensorFlow)
 - NVidia Corp. (hardware)
 - von Lilienfeld Group (for GBD9)
 - Chan Group (for PySCF)

### Common Issues:
- nan during training due to bad checkpoints in /networks (clean.sh)
- Also crashes when reviving networks from disk.
- if you have these issues try re-installing or:

```
sh clean.sh
```
