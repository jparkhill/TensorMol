import socket
import numpy as np
from ..PhysicalData import *


HDRLEN = 12
class TMIPIManger():
	def __init__(self, EnergyForceField=None, TCP_IP="localhost", TCP_PORT= 31415):
		self.EnergyForceField = EnergyForceField
		self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.hasdata = False
		try:
			self.s.connect((TCP_IP, TCP_PORT))
			print ("Connect to server with address:", TCP_IP+" "+str(TCP_PORT))
		except:
			print ("Fail connect to server with address: ", TCP_IP+" "+str(TCP_PORT))


	def md_run(self):
		while (True):
			data = self.s.recv(HDRLEN)
			if data.strip() == "STATUS":
				if self.hasdata:
					print ("client has data.")
					self.s.sendall("HAVEDATA    ")
				else:
					print ("client is ready to get position from server")
					self.s.sendall("READY       ")
			elif data.strip() == "POSDATA":
					print ("server is sending positon.")
					buf_ = self.s.recv(9*8) # cellh np.float64
					cellh = np.fromstring(buf_, np.float64)/BOHRPERA
					buf_ = self.s.recv(9*8) # cellih np.float64
					cellih = np.fromstring(buf_, np.float64)*BOHRPERA
					buf_ = self.s.recv(4) # natom
					natom = np.fromstring(buf_, np.int32)[0]
					buf_ = self.s.recv(3*natom*8) # position
					position = (np.fromstring(buf_, np.float64)/BOHRPERA).reshape((-1, 3))
					print ("cellh:", cellh, "  cellih:", cellih, " natom:", natom)
					print ("position:", position)
					print ("now is running the client to calculate force...")

					energy, force=self.EnergyForceField(position)
					force = force/JOULEPERHARTREE/BOHRPERA
					# some dummyy function to calculte the energy, natom,
					vir = np.zeros((3,3))

					self.hasdata = True

			elif data.strip() == "GETFORCE":
					print ("server is ready to get force from client")
					self.s.sendall("FORCEREADY  ")
					self.s.sendall(np.float64(energy))
					self.s.sendall(np.int32(natom))
					self.s.sendall(force)
					self.s.sendall(vir)
					self.s.sendall(np.int32(7))
					self.s.sendall("nothing")
					self.hasdata = False
			else:
				raise Exception("wrong message from server")
