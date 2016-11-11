#Calculate an embeeding for a molecular, such as coulomb matrix


from Mol import *
from Util import *


class MolDigester:
	def __init__(self, name_="Coulomb", OType_="Energy"):
		self.name = name_
		self.OType = OType_
		self.lshape = None  # output is just the energy
		self.eshape = None	




	def EmbF(self, mol_):
		if (self.name =="Coulomb"):
			return self.make_cm
		else:
			raise Exception("Unknown Embedding Function")
		
		return 	

#	def Emb(self, mol_,  MakeOutputs=True):
#		Ins =




	def make_cm(self, mol_):
		natoms  = mol_.NAtoms()
		CM=np.zeros((natoms, natoms))
		deri_CM = np.zeros((natoms, natoms, 6))
		xyz = (mol_.coords).copy()
		ele = (mol_.atoms).copy()
		code = """
		double dist = 0.0;
		double deri[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
		for (int j=0; j < natoms; j++) {
			for (int k=0; k<natoms; k++) {
				if (k==j) {
					dist=1.0;
					deri[0] =0.0;
					deri[1] =0.0;
					deri[2] =0.0;
					deri[3] =0.0;
					deri[4] =0.0;
					deri[5] =0.0;
				} 
				else {
					dist=sqrt(pow(xyz[j*3+0]-xyz[k*3+0],2) + pow(xyz[j*3+1]-xyz[k*3+1],2) + pow(xyz[j*3+2]-xyz[k*3+2],2));
					deri[0] = -(xyz[j*3+0]-xyz[k*3+0])/(dist*dist*dist);
					deri[1] = -(xyz[j*3+1]-xyz[k*3+1])/(dist*dist*dist);
					deri[2] = -(xyz[j*3+2]-xyz[k*3+2])/(dist*dist*dist);
					deri[3] = -deri[0];
					deri[4] = -deri[1];
					deri[5] = -deri[2];
				}
				CM[natoms*j+k]= ele[j]*ele[k]/dist;
				deri_CM[natoms*j*6+k*6+0]=ele[j]*ele[k]*deri[0];
				deri_CM[natoms*j*6+k*6+1]=ele[j]*ele[k]*deri[1];
				deri_CM[natoms*j*6+k*6+2]=ele[j]*ele[k]*deri[2];
				deri_CM[natoms*j*6+k*6+3]=ele[j]*ele[k]*deri[3];
				deri_CM[natoms*j*6+k*6+4]=ele[j]*ele[k]*deri[4];
				deri_CM[natoms*j*6+k*6+5]=ele[j]*ele[k]*deri[5];
			}

		}

		"""
		res = inline(code, ['CM','natoms','xyz','ele', 'deri_CM'],headers=['<math.h>','<iostream>'], compiler='gcc')
		return CM, deri_CM



	def GetUpTri(self, CM):
		return  CM[np.triu_indices(CM.shape[0], 0)]
			


	def EvalDigest(self, mol_):
		CM, deri_CM = (self.EmbF(mol_))(mol_)
                UpTri = self.GetUpTri(CM)
                if self.lshape ==None or self.eshape==None:
                        self.lshape=1
                        self.eshape=UpTri.shape[0]
                return UpTri, deri_CM

	def TrainDigest(self, mol_):
		CM, deri_CM = (self.EmbF(mol_))(mol_)
		UpTri = self.GetUpTri(CM)
		out = mol_.frag_mbe_energy
		if self.lshape ==None or self.eshape==None:
			self.lshape=1
			self.eshape=UpTri.shape[0] 
		return UpTri, out

	def Print(self):
		print "Digest name: ", self.name
