import pickle

def PickleMapName(name):
	"""
	If you change the name of a function or module, then pickle, you can fix it with this.
	"""
	renametable = {
		'TensorMol.TensorMolData_EE': 'TensorMol.TensorMolDataEE',
		'TensorMol.TFMolInstance_EE': 'TensorMol.TFMolInstanceEE',
		'TensorMolData_EE': 'TensorMolDataEE'
		}
	if name in renametable:
		return renametable[name]
	return name

def mapped_load_global(self):
	module = PickleMapName(self.readline()[:-1])
	name = PickleMapName(self.readline()[:-1])
	klass = self.find_class(module, name)
	self.append(klass)

def UnPickleTM(file):
	unpickler = pickle.Unpickler(file)
	unpickler.dispatch[pickle.GLOBAL] = mapped_load_global
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
