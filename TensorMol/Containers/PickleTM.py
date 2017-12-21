from __future__ import absolute_import
import pickle,sys

def PickleMapName(name):
	"""
	If you change the name of a function or module, then pickle, you can fix it with this.
	"""
	renametable = {
		'TensorMol.TensorMolData_EE': 'TensorMol.TensorMolDataEE',
		'TensorMol.TFMolInstance_EE': 'TensorMol.TFMolInstanceEE',
		'TensorMol.TensorMolData': 'TensorMol.Containers.TensorMolData',
		'TensorMol.Mol': 'TensorMol.Containers.Mol',
		'DigestMol': 'TensorMol.Containers.DigestMol',
		'TensorMol.DigestMol': 'TensorMol.Containers.DigestMol',
		'TensorMol.TFMolInstanceDirect': 'TensorMol.TFNetworks.TFMolInstanceDirect',
		'TensorMol.Transformer': 'TensorMol.ForceModifiers.Transformer',
		'TensorMolData_EE': 'TensorMolDataEE'
		}
	if name in renametable:
		#print("REMAPPING PICKLE LOAD:",name,"TO",renametable[name])
		return renametable[name]
	#else:
		#print("NOT REMAPPING PICKLE LOAD:",name)
	return name

def mapped_load_global(self):
	module = PickleMapName(self.readline()[:-1])
	name = PickleMapName(self.readline()[:-1])
	print("Finding ", module,name)
	klass = self.find_class(module, name)
	self.append(klass)

def UnPickleTM(file):
	"""
	Eventually we need to figure out how the mechanics of dispatch tables changed.
	Since we only use this as a hack anyways, I'll just comment out what changed
	between python2.7x and python3x.
	"""
	tmp = None
	if sys.version_info[0] < 3:
		unpickler = pickle.Unpickler(file)
		unpickler.dispatch[pickle.GLOBAL] = mapped_load_global
		tmp = unpickler.load()
	else:
		unpickler = pickle.Unpickler(file,encoding='latin1')
		tmp = unpickler.load()
	tmp.pop('evaluate',None)
	tmp.pop('MolInstance_fc_sqdiff_BP',None)
	tmp.pop('Eval_BPForceSingle',None)
	tmp.pop('TFMolManage',None)
	tmp.pop('Prepare',None)
	tmp.pop('Trainable',None)
	tmp.pop('TFMolManage.Trainable',None)
	tmp.pop('__init__',None)
	return tmp
