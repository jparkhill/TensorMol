#include <Python.h>
#include <numpy/arrayobject.h>
#include <dictobject.h>
#include <math.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "SH.hpp"

// So that parameters can be dealt with elsewhere.
static SHParams ParseParams(PyObject *Pdict)
{
	SHParams tore;
	{
		PyObject* RBFo = PyDict_GetItemString(Pdict, "RBFS");
		PyArrayObject* RBFa = (PyArrayObject*) RBFo;
		tore.RBFS = (double*)RBFa->data;
	}
	{
		PyObject* RBFo = PyDict_GetItemString(Pdict, "SRBF");
		PyArrayObject* RBFa = (PyArrayObject*) RBFo;
		tore.SRBF = (double*)RBFa->data;
	}
	{
		PyObject* RBFo = PyDict_GetItemString(Pdict, "ANES");
		PyArrayObject* RBFa = (PyArrayObject*) RBFo;
		tore.ANES = (double*)RBFa->data;
	}
	tore.SH_LMAX = PyInt_AS_LONG((PyDict_GetItemString(Pdict,"SH_LMAX")));
	tore.SH_NRAD = PyInt_AS_LONG((PyDict_GetItemString(Pdict,"SH_NRAD")));
	tore.SH_ORTH = PyInt_AS_LONG((PyDict_GetItemString(Pdict,"SH_ORTH")));
	tore.SH_MAXNR = PyInt_AS_LONG((PyDict_GetItemString(Pdict,"SH_MAXNR")));
	// HACK TO DEBUG>>>>>
	/*
	double t[9] = {1.0,0.,0.,0.,1.,0.,0.,0.,1.};
	double* out;
	TransInSHBasis(&tore,t, out);
	*/
	return tore;
}

static SymParams ParseSymParams(PyObject *Pdict)
{
	SymParams tore;
	{
		PyObject* RBFo = PyDict_GetItemString(Pdict, "AN1_r_Rs");
		PyArrayObject* RBFa = (PyArrayObject*) RBFo;
		tore.r_Rs = (double*)RBFa->data;
	}
	{
		PyObject* RBFo = PyDict_GetItemString(Pdict, "AN1_a_Rs");
		PyArrayObject* RBFa = (PyArrayObject*) RBFo;
		tore.a_Rs = (double*)RBFa->data;
	}
	{
		PyObject* RBFo = PyDict_GetItemString(Pdict, "AN1_a_As");
		PyArrayObject* RBFa = (PyArrayObject*) RBFo;
		tore.a_As = (double*)RBFa->data;
	}
	tore.num_r_Rs = PyInt_AS_LONG((PyDict_GetItemString(Pdict,"AN1_num_r_Rs")));
	tore.num_a_Rs = PyInt_AS_LONG((PyDict_GetItemString(Pdict,"AN1_num_a_Rs")));
	tore.num_a_As = PyInt_AS_LONG((PyDict_GetItemString(Pdict,"AN1_num_a_As")));
	tore.r_Rc = PyFloat_AsDouble((PyDict_GetItemString(Pdict,"AN1_r_Rc")));
	tore.a_Rc = PyFloat_AsDouble((PyDict_GetItemString(Pdict,"AN1_a_Rc")));
	tore.eta = PyFloat_AsDouble((PyDict_GetItemString(Pdict,"AN1_eta")));
	tore.zeta = PyFloat_AsDouble((PyDict_GetItemString(Pdict,"AN1_zeta")));
	return tore;
}

inline double fc(const double &dist, const double &dist_cut) {
	if (dist > dist_cut)
		return(0.0);
	else
		return (0.5*(cos(PI*dist/dist_cut)+1));
};

inline double gaussian(const double dist_cut, const int ngrids, double dist, const int j,  const double  width, const double height)  {
	double position;
	position = dist_cut / (double)ngrids * (double)j;
	return height*exp(-((dist - position)*(dist-position))/(2*width*width));
};

int Make_AM (const double*, const array<std::vector<int>, 100>, const int &, const int &,  const int &, const double*, npy_intp*, double*, const int &);
int G1(double *, const double *, const double *, int , int , const array<std::vector<int>, 100> , const int, const double *, const double *, const double );
int G2(double *, const double *, const double *, int , int,  int , const array<std::vector<int>, 100>  , const int , const int , const double , const double , const double);
void rdf(double *,  const int ,  const array<std::vector<int>, 100> ,  const int, const double, const double *,  const double, const double);

struct MyComparator
{
	const std::vector<double> & value_vector;

	MyComparator(const std::vector<double> & val_vec):
	value_vector(val_vec) {}

	bool operator()(int i1, int i2)
	{
		return value_vector[i1] < value_vector[i2];
	}
};

void rdf(double *data,  const int ngrids,  const array<std::vector<int>, 100> ele_index,  const int v_index, const double *center, const double *xyz,  const double dist_cut, double const width) {
	double dist;
	double height=1.0;
	for (std::size_t i = 0; i < ele_index[v_index].size(); i++) {
		dist = sqrt(pow(xyz[ele_index[v_index][i]*3+0] - center[0],2)+pow(xyz[ele_index[v_index][i]*3+1] - center[1],2)+pow(xyz[ele_index[v_index][i]*3+2] - center[2],2));
		//if (dist < dist_cut)
		for (int j = 0 ; j < ngrids; j++) {
			data[v_index*ngrids+j] += gaussian(dist_cut, ngrids, dist, j, width, height); // this can be easily parralled similar like ex-grids -JAP GOOD IDEA
		}
	}
}

int  PGaussian(double *data, const double *eta,  int dim_eta, const array<std::vector<int>, 100>  ele_index, const int v_index, const double *center, const double *xyz, const double dist_cut)
{
	double dist1, fc1,x,y,z;
	for (int i = 0; i < ele_index[v_index].size(); i++) {
		x = xyz[ele_index[v_index][i]*3+0] - center[0];
		y = xyz[ele_index[v_index][i]*3+1] - center[1];
		z = xyz[ele_index[v_index][i]*3+2] - center[2];
		dist1 = sqrt(pow(x,2)+pow(y,2)+pow(z,2));
		if (dist1 > dist_cut)
		continue;
		else {
			fc1=fc(dist1, dist_cut);
			for (int m = 0; m < dim_eta; m++) {
				data[dim_eta*0+m]=data[dim_eta*0+m]+x/dist1*exp(-(dist1/eta[m])*(dist1/eta[m]))*fc1; // x
				data[dim_eta*1+m]=data[dim_eta*1+m]+y/dist1*exp(-(dist1/eta[m])*(dist1/eta[m]))*fc1; // y
				data[dim_eta*2+m]=data[dim_eta*2+m]+z/dist1*exp(-(dist1/eta[m])*(dist1/eta[m]))*fc1; // z
			}
		}
	}
	return 0;
}

int  G1(double *data, const double *Rs, const double *eta, int dim_Rs, int dim_eta, const array<std::vector<int>, 100>  ele_index, const int v_index, const double *center, const double *xyz, const double dist_cut) {
	double dist1,fc1;
	for (int i = 0; i < ele_index[v_index].size(); i++) {
		dist1 = sqrt(pow(xyz[ele_index[v_index][i]*3+0] - center[0],2)+pow(xyz[ele_index[v_index][i]*3+1] - center[1],2)+pow(xyz[ele_index[v_index][i]*3+2] - center[2],2));
		if (dist1 > dist_cut)
		continue;
		else {
			fc1=fc(dist1, dist_cut);
			for (int n = 0; n < dim_Rs; n++)
			for (int m = 0; m < dim_eta; m++) {
				data[dim_eta*n+m]=data[dim_eta*n+m]+exp(-eta[m]*(dist1-Rs[n])*(dist1-Rs[n]))*fc1;
			}
		}
	}
	return 0;
}

double dist(double x0,double y0,double z0,double x1,double y1,double z1)
{
	return sqrt((x0-x1)*(x0-x1)+(y0-y1)*(y0-y1)+(z0-z1)*(z0-z1));
}

void  G2(double *data, const double *zeta, const double *eta, int dim_zeta, int dim_eta,  int lambda, const array<std::vector<int>, 100>  ele_index, const int v1_index, const int v2_index, const double *center, const double *xyz, const double dist_cut) {

	double dist1,dist2,dist3,fc1,fc2,fc3,theta,A,B,C,distsum;

	for (int i = 0; i < ele_index[v1_index].size(); i++) {
		dist1 = sqrt(pow(xyz[ele_index[v1_index][i]*3+0] - center[0],2)+pow(xyz[ele_index[v1_index][i]*3+1] - center[1],2)+pow(xyz[ele_index[v1_index][i]*3+2] - center[2],2));
		if (dist1 > dist_cut)
		continue;
		for (int j = 0; j < ele_index[v2_index].size(); j++) {
			dist2 = sqrt(pow(xyz[ele_index[v2_index][j]*3+0] - center[0],2)+pow(xyz[ele_index[v2_index][j]*3+1] - center[1],2)+pow(xyz[ele_index[v2_index][j]*3+2] - center[2],2));
			dist3 = sqrt(pow(xyz[ele_index[v2_index][j]*3+0] - xyz[ele_index[v1_index][i]*3+0] ,2)+pow(xyz[ele_index[v2_index][j]*3+1] - xyz[ele_index[v1_index][i]*3+1] ,2)+pow(xyz[ele_index[v2_index][j]*3+2] - xyz[ele_index[v1_index][i]*3+2], 2));
			if ((dist2 > dist_cut || dist3 > dist_cut)||(v1_index == v2_index && j <= i )) // v1 and v2 are same kind of elements, do not revisit.
			continue;
			else {
				fc1=fc(dist1, dist_cut),fc2=fc(dist2, dist_cut),fc3=fc(dist3, dist_cut);
				theta = (dist1*dist1+dist2*dist2-dist3*dist3)/(2.0*dist1*dist2);
				for (int n = 0; n <  dim_zeta; n++) {
					A=pow(2.0,1-zeta[n])*pow(1+lambda*theta,zeta[n]);
					C=fc1*fc2*fc3;
					distsum=dist1*dist1+dist2*dist2+dist3*dist3;
					for (int m = 0; m <  dim_eta; m++) {
						B=exp(-eta[m]*distsum);
						data[dim_eta*n+m]=data[dim_eta*n+m]+A*B*C;
					}
				}

			}
		}
	}
}


void SymFunction(double *Sym_data, const int data_pointer, const double *xyz, const uint8_t *atoms, const int natom,  const uint8_t *ele, const int nele, const int atom_num, const array<std::vector<int>, 10> ele_index, const double Rc, const double *g1_param_data, const int g1_dim, const double *g2_param_data, const int g2_dim) {
	double dist1, fc1, dist2, fc2, dist3, fc3, A1, A2,  B, C, theta;
	int bond_index = 0;
	for (int j = 0; j < nele; j++) {
		for (int k = 0; k < ele_index[j].size(); k++) {
			int ejk = ele_index[j][k];
			if (ejk != atom_num) {
				dist1 = sqrt(pow(xyz[ejk*3+0] - xyz[atom_num*3+0],2)+pow(xyz[ejk*3+1] - xyz[atom_num*3+1],2)+pow(xyz[ejk*3+2] - xyz[atom_num*3+2],2));
				if ( dist1 > Rc)
				continue;
				else {
					fc1 = fc(dist1, Rc);
					for (int m = 0; m <  g1_dim; m++)  {
						//std::cout<<"g1_param_data[m*2]"<<g1_param_data[m*2]<<"   "<<"<g1_param_data[m*2+1]"<<g1_param_data[m*2+1]<<std::endl;
						Sym_data[data_pointer + g1_dim*j + m] = Sym_data[data_pointer + g1_dim*j + m] + exp(-g1_param_data[m*2+1]*(dist1 - g1_param_data[m*2])*(dist1 - g1_param_data[m*2]))*fc1;
					}
				}
			}
		}
	}
	int half_total_g2 = g2_dim*nele*(nele+1)/2;
	for (int i = 0; i < nele; i++) {
		for (int j = i; j < nele; j++) {
			for ( int k =0; k < ele_index[i].size(); k++)  {
				dist1 = sqrt(pow(xyz[ele_index[i][k]*3+0] - xyz[atom_num*3+0],2)+pow(xyz[ele_index[i][k]*3+1] - xyz[atom_num*3+1],2)+pow(xyz[ele_index[i][k]*3+2] - xyz[atom_num*3+2],2));
				if (dist1 > Rc or ele_index[i][k] == atom_num)
				continue;
				else {
					for (int l = 0; l < ele_index[j].size(); l++) {
						dist2 = sqrt(pow(xyz[ele_index[j][l]*3+0] - xyz[atom_num*3+0],2)+pow(xyz[ele_index[j][l]*3+1] - xyz[atom_num*3+1],2)+pow(xyz[ele_index[j][l]*3+2] - xyz[atom_num*3+2],2));
						dist3 = sqrt(pow(xyz[ele_index[j][l]*3+0] - xyz[ele_index[i][k]*3+0],2)+pow(xyz[ele_index[j][l]*3+1] - xyz[ele_index[i][k]*3+1],2)+pow(xyz[ele_index[j][l]*3+2] - xyz[ele_index[i][k]*3+2],2));
						if ((dist2 > Rc) || (dist3 > Rc) || (i == j && l <= k) || ele_index[j][l] == atom_num)   // v1 and v2 are same kind of elements, do not revisit.
						continue;
						else {
							fc1=fc(dist1, Rc),fc2=fc(dist2, Rc),fc3=fc(dist3, Rc);
							theta = (dist1*dist1+dist2*dist2-dist3*dist3)/(2.0*dist1*dist2);
							for (int n = 0; n < g2_dim; n++) {
								A1 = pow(2.0, 1-g2_param_data[n*2])*pow((1+theta), g2_param_data[n*2]);
								A2 = pow(2.0, 1-g2_param_data[n*2])*pow((1-theta), g2_param_data[n*2]);
								B = exp(-g2_param_data[n*2+1]*(dist1*dist1+dist2*dist2+dist3*dist3));
								C = fc1*fc2*fc3;
								//std::cout<<" g2_dim*bond_index + n" << g2_dim*bond_index + n << "ele:"<<i<<" "<<j<<std::endl;
								Sym_data[data_pointer + g1_dim*nele + g2_dim*bond_index + n] = Sym_data[data_pointer + g1_dim*nele + g2_dim*bond_index + n] + A1*B*C;
								Sym_data[data_pointer + g1_dim*nele + half_total_g2 + g2_dim*bond_index + n] = Sym_data[data_pointer + g1_dim*nele + half_total_g2 + g2_dim*bond_index + n] + A2*B*C;
							}
						}
					}
				}
			}
			bond_index = bond_index + 1;
		}
	}
}

void ANI1_SymFunction_deri(double *ANI1_Sym_deri_data,  const int data_pointer, const double *xyz, const uint8_t *atoms, const int natom, const uint8_t *ele, const int nele, const int atom_num, const array<std::vector<int>, 10> ele_index, const double radius_Rc, const double angle_Rc, const double *radius_Rs, const int radius_Rs_dim, const double *angle_Rs, const int angle_Rs_dim, const double *angle_As, const int angle_As_dim, const double eta, const double zeta) {
	double dist1, fc1, dist2, fc2, dist3, theta, A, C ,B, tmp_v, theta_deri_ci, theta_deri_cj, theta_deri_ij, fc1_deri, fc2_deri, fc3_deri ;
	int g1_size = radius_Rs_dim;
	int bond_index = 0;
	int at3 = atom_num*3;
	int at31 = at3+1;
	int at32 = at3+2;
	for (int j = 0; j < nele; j++ ) {
		for ( int k = 0;  k < ele_index[j].size(); k++) {
			int ejk = ele_index[j][k];
			if (ejk != atom_num) {
				dist1 = sqrt(pow(xyz[ejk*3] - xyz[at3],2)+pow(xyz[ejk*3+1] - xyz[at31],2)+pow(xyz[ejk*3+2] - xyz[at32],2));
				if ( dist1 > radius_Rc)
					continue;
				else {
					fc1 = fc(dist1, radius_Rc);
					fc1_deri  = -0.5*sin(PI*dist1/radius_Rc)/dist1*PI/radius_Rc;
					for (int m = 0; m < radius_Rs_dim; m++) {
						int mshift = data_pointer + radius_Rs_dim*3*natom*j + m*3*natom;
						//std::cout<<"fc1: "<<fc1<<"exp: "<<( exp(-eta*(dist1-radius_Rs[m])*(dist1-radius_Rs[m]))*(-2*eta*(dist1 - radius_Rs[m]))/dist1*(-xyz[ejk*3])) << std::endl;
						//std::cout<<"  part 2:"<<fc1_deri*(-xyz[ejk*3])*exp(-eta*(dist1-radius_Rs[m])*(dist1-radius_Rs[m])) << std::endl;
						//std::cout<<"workiing on atom:"<<ejk<<" and atom:"<<atom_num<<"  index:"<<mshift + ejk*3 + 0<<" and index:"<<mshift + at3 + 1<<std::endl;
						//std::cout<<"term1: "<<fc1*( exp(-eta*(dist1-radius_Rs[m])*(dist1-radius_Rs[m]))*(-2*eta*(dist1 - radius_Rs[m]))/dist1*(-xyz[ejk*3+2]))  <<" term2:" << fc1_deri*(-xyz[ejk*3+2])*exp(-eta*(dist1-radius_Rs[m])*(dist1-radius_Rs[m]))<<std::endl;
						//std::cout<<xyz[ejk*3+2]<<std::endl;
						//std::cout<<"term3: "<<fc1*( exp(-eta*(dist1-radius_Rs[m])*(dist1-radius_Rs[m]))*(-2*eta*(dist1 - radius_Rs[m]))/dist1*(xyz[at32])) <<"term4 :" << fc1_deri*(xyz[at32])*exp(-eta*(dist1-radius_Rs[m])*(dist1-radius_Rs[m]))<<std::endl;

						double d1mrs = dist1-radius_Rs[m];
						double expgau = exp(-eta*(d1mrs)*(d1mrs));

						ANI1_Sym_deri_data[mshift + ejk*3 + 0] += fc1*( expgau)*(-2*eta*d1mrs)/dist1*(xyz[ejk*3]-xyz[at3]) + fc1_deri*(xyz[ejk*3] - xyz[at3])*expgau;
						ANI1_Sym_deri_data[mshift + ejk*3 + 1] += fc1*( expgau)*(-2*eta*d1mrs)/dist1*(xyz[ejk*3+1]-xyz[at31]) + fc1_deri*(xyz[ejk*3+1] - xyz[at31])*expgau;
						ANI1_Sym_deri_data[mshift + ejk*3 + 2] += fc1*( expgau)*(-2*eta*d1mrs)/dist1*(xyz[ejk*3+2]-xyz[at32]) + fc1_deri*(xyz[ejk*3+2] - xyz[at32])*expgau;

						ANI1_Sym_deri_data[mshift + at3 + 0] += fc1*( expgau)*(-2*eta*d1mrs)/dist1*(xyz[at3] - xyz[ejk*3]) + fc1_deri*(xyz[at3] - xyz[ejk*3])*expgau;
						ANI1_Sym_deri_data[mshift + at3 + 1] += fc1*( expgau)*(-2*eta*d1mrs)/dist1*(xyz[at31] - xyz[ejk*3+1]) + fc1_deri*(xyz[at31] - xyz[ejk*3+1])*expgau;
						ANI1_Sym_deri_data[mshift + at3 + 2] += fc1*( expgau)*(-2*eta*d1mrs)/dist1*(xyz[at32] - xyz[ejk*3+2]) + fc1_deri*(xyz[at32] - xyz[ejk*3+2])*expgau;
						//if (mshift + at3 + 2 == 2)
						//        std::cout<<ANI1_Sym_deri_data[mshift + at3 + 2]<<std::endl;
					}
				}
			}
		}
	}
	for (int i = 0; i < nele; i++) {
		for (int j = i; j < nele; j++) {
			for ( int k =0; k < ele_index[i].size(); k++)  {
				int eik = ele_index[i][k];
				int eik3 = 3*eik;
				dist1 = sqrt(pow(xyz[eik3] - xyz[at3],2)+pow(xyz[eik3+1] - xyz[at31],2)+pow(xyz[eik3+2] - xyz[at32],2));
				double dist1sq = dist1*dist1;
				if (dist1 > angle_Rc or eik == atom_num)
					continue;
				else {
					for (int l = 0; l < ele_index[j].size(); l++) {
						int ejl = ele_index[j][l];
						int ejl3 = 3*ejl;
						dist2 = sqrt(pow(xyz[ejl3] - xyz[at3],2)+pow(xyz[ejl3+1] - xyz[at31],2)+pow(xyz[ejl3+2] - xyz[at32],2));
						if ((dist2 > angle_Rc) || (i == j && l <= k) || ejl == atom_num) // change to <= since when v1 and v2 are same kind of element, do not revisit. diff by a factor of two
							continue;
						else {
							double d1pd2 = (dist1+dist2);
							dist3 = sqrt(pow(xyz[ejl3] - xyz[eik3],2)+pow(xyz[ejl3+1] - xyz[eik3+1],2)+pow(xyz[ejl3+2] - xyz[eik3+2],2));
							double dist2sq = dist2*dist2;
							double dist3sq = dist3*dist3;
							fc1 = fc(dist1, angle_Rc), fc2 = fc(dist2, angle_Rc);
							fc1_deri = -0.5*sin(PI*dist1/angle_Rc)/dist1*PI/angle_Rc, fc2_deri = -0.5*sin(PI*dist2/angle_Rc)/dist2*PI/angle_Rc;
							tmp_v = (dist1sq+dist2sq-dist3sq)/(2.0*dist1*dist2);
							if (tmp_v<-1.0)
								tmp_v =-1.0;
							else if (tmp_v > 1.0)
								tmp_v = 1.0;
							//theta = acos(tmp_v);
							theta = acos(tmp_v); // round to 7 decimal place
							theta_deri_ci = -(1.0/dist2-(dist1sq+dist2sq-dist3sq)/(2.0*dist1sq*dist2))/sqrt(1-pow(dist1sq+dist2sq-dist3sq,2.0)/(4.0*dist1sq*dist2sq));
							theta_deri_cj = -(1.0/dist1-(dist1sq+dist2sq-dist3sq)/(2.0*dist1*dist2sq))/sqrt(1-pow(dist1sq+dist2sq-dist3sq,2.0)/(4.0*dist1sq*dist2sq));
							theta_deri_ij = dist3/(dist1*dist2*sqrt(1.0-pow(dist1sq+dist2sq-dist3sq,2.0)/(4.0*dist1sq*dist2sq)));
							C = fc1*fc2;
							for (int m = 0; m < angle_As_dim; m++) {
								A = pow(1+cos(theta-angle_As[m]), zeta);
								double ttomz = pow(2.0, 1-zeta);
								double p1pc = pow(1+cos(theta-angle_As[m]), zeta-1);
								double stam = (-sin(theta-angle_As[m]));
								double zps = zeta*p1pc*stam;
								for (int n = 0; n < angle_Rs_dim; n++) {
									B = exp(-eta*(d1pd2/2.0-angle_Rs[n])*(d1pd2/2.0-angle_Rs[n]));
									int mnshift = data_pointer+radius_Rs_dim*nele*3*natom+bond_index*angle_Rs_dim*angle_As_dim*3*natom+ (m*angle_Rs_dim+n)*3*natom;
									double tedd2a = (-2*eta*(d1pd2/2.0-angle_Rs[n]));
									ANI1_Sym_deri_data[mnshift + eik3] +=  ttomz*(zps*(theta_deri_ci*(xyz[eik3] - xyz[at3])/dist1 + theta_deri_ij*(xyz[eik3]-xyz[ejl3])/dist3)*B*C + A*B*tedd2a/dist1/2.0*(xyz[eik3] - xyz[at3])*C + A*B*fc1_deri*(xyz[eik3]- xyz[at3])*fc2);
									ANI1_Sym_deri_data[mnshift + eik3+1] +=  ttomz*(zps*(theta_deri_ci*(xyz[eik3+1] - xyz[at31])/dist1 + theta_deri_ij*(xyz[eik3+1]-xyz[ejl3+1])/dist3)*B*C + A*B*tedd2a/dist1/2.0*(xyz[eik3+1] - xyz[at31])*C + A*B*fc1_deri*(xyz[eik3+1]- xyz[at31])*fc2);
									ANI1_Sym_deri_data[mnshift + eik3+2] +=  ttomz*(zps*(theta_deri_ci*(xyz[eik3+2] - xyz[at32])/dist1 + theta_deri_ij*(xyz[eik3+2]-xyz[ejl3+2])/dist3)*B*C + A*B*tedd2a/dist1/2.0*(xyz[eik3+2] - xyz[at32])*C + A*B*fc1_deri*(xyz[eik3+2]- xyz[at32])*fc2);

									ANI1_Sym_deri_data[mnshift + ejl3] +=  ttomz*(zps*(theta_deri_cj*(xyz[ejl3] - xyz[at3])/dist2 + theta_deri_ij*(xyz[ejl3]-xyz[eik3])/dist3)*B*C + A*B*tedd2a/dist2/2.0*(xyz[ejl3] - xyz[at3])*C + A*B*fc2_deri*(xyz[ejl3] - xyz[at3])*fc1);
									ANI1_Sym_deri_data[mnshift + ejl3+1] +=  ttomz*(zps*(theta_deri_cj*(xyz[ejl3+1] - xyz[at31])/dist2 + theta_deri_ij*(xyz[ejl3+1]-xyz[eik3+1])/dist3)*B*C + A*B*tedd2a/dist2/2.0*(xyz[ejl3+1] - xyz[at31])*C + A*B*fc2_deri*(xyz[ejl3+1] - xyz[at31])*fc1);
									ANI1_Sym_deri_data[mnshift + ejl3+2] +=  ttomz*(zps*(theta_deri_cj*(xyz[ejl3+2] - xyz[at32])/dist2 + theta_deri_ij*(xyz[ejl3+2]-xyz[eik3+2])/dist3)*B*C + A*B*tedd2a/dist2/2.0*(xyz[ejl3+2] - xyz[at32])*C + A*B*fc2_deri*(xyz[ejl3+2] - xyz[at32])*fc1);

									ANI1_Sym_deri_data[mnshift + at3] +=  ttomz*(zps*(theta_deri_ci*(xyz[at3] - xyz[eik3])/dist1 + theta_deri_cj*(xyz[at3] - xyz[ejl3])/dist2)*B*C + A*B*tedd2a/2.0*((xyz[at3] - xyz[eik3])/dist1 + (xyz[at3] - xyz[ejl3])/dist2)*C + A*B*fc1_deri*(xyz[at3] - xyz[eik3])*fc2 + A*B*fc2_deri*(xyz[at3] - xyz[ejl3])*fc1);
									ANI1_Sym_deri_data[mnshift + at31] +=  ttomz*(zps*(theta_deri_ci*(xyz[at31] - xyz[eik3+1])/dist1 + theta_deri_cj*(xyz[at31] - xyz[ejl3+1])/dist2)*B*C + A*B*tedd2a/2.0*((xyz[at31] - xyz[eik3+1])/dist1 + (xyz[at31] - xyz[ejl3+1])/dist2)*C + A*B*fc1_deri*(xyz[at31] - xyz[eik3+1])*fc2 + A*B*fc2_deri*(xyz[at31] - xyz[ejl3+1])*fc1);
									ANI1_Sym_deri_data[mnshift + at32] +=  ttomz*(zps*(theta_deri_ci*(xyz[at32] - xyz[eik3+2])/dist1 + theta_deri_cj*(xyz[at32] - xyz[ejl3+2])/dist2)*B*C + A*B*tedd2a/2.0*((xyz[at32] - xyz[eik3+2])/dist1 + (xyz[at32] - xyz[ejl3+2])/dist2)*C + A*B*fc1_deri*(xyz[at32] - xyz[eik3+2])*fc2 + A*B*fc2_deri*(xyz[at32] - xyz[ejl3+2])*fc1);
//									 if (   bond_index == 8 && m == 1 && n ==5 && atom_num == 7)  {
//										std::cout<<"zeta: "<<zeta<<"  angle_As[m]"<<angle_As[m]<<"   angle_Rs[n]"<<angle_Rs[n]<<"  angle_Rc"<<angle_Rc<<"  eta:"<<eta<<std::endl;
//										std::cout<<"ele_index[i][k]"<<ele_index[i][k]<<"  "<<" ejl"<<ejl<<std::endl;
//										std::cout<<"N:"<<xyz[ele_index[i][k]*3+2]<<" O "<<xyz[ejl*3+2]<<" H:"<<xyz[at32]<<std::endl;
//										std::cout<<ANI1_Sym_deri_data[mnshift + ele_index[i][k]*3+2]<<std::endl;
//										std::cout<<"pow(2.0, 1-zeta)"<<pow(2.0, 1-zeta)<<"  pow(1+cos(theta-angle_As[m]), zeta-1)"<<pow(1+cos(theta-angle_As[m]), zeta-1)<<"  -sin(theta-angle_As[m])"<<-sin(theta-angle_As[m])<<"  theta_deri_ci"<<theta_deri_ci<<"  theta_deri_ij"<<theta_deri_ij<<"  (-2*eta*((dist1+dist2)/2.0-angle_Rs[n]))"<<(-2*eta*((dist1+dist2)/2.0-angle_Rs[n]))<<"  fc1_deri"<<fc1_deri<<" A:"<<A<<"  B:"<<B<<" C:"<<C<<" (dist1+dist2)/2.0-angle_Rs[n]"<<(dist1+dist2)/2.0-angle_Rs[n]<<"    deri theta"<<(theta_deri_ci*(xyz[ele_index[i][k]*3+2] - xyz[at32])/dist1 + theta_deri_ij*(xyz[ele_index[i][k]*3+2]-xyz[ejl*3+2])/dist3)<<std::endl;
//
//}
//
								}
							}
						}
					}
				}
			}
			bond_index = bond_index + 1;
		}
	}
}

void ANI1_SymFunction(double *ANI1_Sym_data,  const int data_pointer, const double *xyz, const uint8_t *atoms, const int natom, const uint8_t *ele, const int nele, const int atom_num, const array<std::vector<int>, 10> ele_index, const double radius_Rc, const double angle_Rc, const double *radius_Rs, const int radius_Rs_dim, const double *angle_Rs, const int angle_Rs_dim, const double *angle_As, const int angle_As_dim, const double eta, const double zeta) {
	double dist1, fc1, dist2, fc2, dist3, fc3, theta, A, C ,B, tmp_v ;
	int g1_size = radius_Rs_dim;
	int g2_size = angle_Rs_dim * angle_As_dim;
	int bond_index = 0;
	int an3 = atom_num*3;
	int SYMdim = nele*radius_Rs_dim + nele*(nele+1)/2*angle_Rs_dim*angle_As_dim;
#pragma omp parallel for
	for (int j = 0; j < nele; j++ ) {
		for ( int k = 0;  k < ele_index[j].size(); k++) {
			int ejk = ele_index[j][k];
			if (ejk != atom_num)
			{
				dist1 = sqrt(pow(xyz[ejk*3] - xyz[an3],2)+pow(xyz[ejk*3+1] - xyz[an3+1],2)+pow(xyz[ejk*3+2] - xyz[an3+2],2));
				// Have to kill this hard cutoff...
				if ( dist1 > radius_Rc)
					continue;
				else {
					fc1 = fc(dist1, radius_Rc);
					for (int m = 0; m < radius_Rs_dim; m++) {
						ANI1_Sym_data[data_pointer + radius_Rs_dim*j + m] = ANI1_Sym_data[data_pointer + radius_Rs_dim*j + m] + exp(-eta*(dist1-radius_Rs[m])*(dist1-radius_Rs[m]))*fc1;
					}
				}
			}
		}
	}

//
//  Kun: this really needs to be OMP'd
//       Are only independent i's assigned in a loop?
	//#pragma omp parallel for     // do not getting any speed up
	for (int i = 0; i < nele; i++) {
		for (int j = i; j < nele; j++) {
			for ( int k =0; k < ele_index[i].size(); k++)  {
				dist1 = sqrt(pow(xyz[ele_index[i][k]*3] - xyz[atom_num*3],2)+pow(xyz[ele_index[i][k]*3+1] - xyz[atom_num*3+1],2)+pow(xyz[ele_index[i][k]*3+2] - xyz[atom_num*3+2],2));
				if (dist1 > angle_Rc or ele_index[i][k] == atom_num)
				continue;
				else {
					for (int l = 0; l < ele_index[j].size(); l++) {
					int ejl = ele_index[j][l];
					int ejl3 = ejl*3;
						dist2 = sqrt(pow(xyz[ejl3] - xyz[atom_num*3],2)+pow(xyz[ejl3+1] - xyz[atom_num*3+1],2)+pow(xyz[ejl3+2] - xyz[atom_num*3+2],2));
						dist3 = sqrt(pow(xyz[ejl3] - xyz[ele_index[i][k]*3],2)+pow(xyz[ejl3+1] - xyz[ele_index[i][k]*3+1],2)+pow(xyz[ejl3+2] - xyz[ele_index[i][k]*3+2],2));
						if ((dist2 > angle_Rc) || (i == j && l <= k) || ele_index[j][l] == atom_num) // change to <= since when v1 and v2 are same kind of element, do not revisit. diff by a factor of two
							continue;
						else {
							fc1 = fc(dist1, angle_Rc), fc2 = fc(dist2, angle_Rc), fc3 = fc(dist3, angle_Rc);
							tmp_v = (dist1*dist1+dist2*dist2-dist3*dist3)/(2.0*dist1*dist2);
							//theta = acos(tmp_v);
							if (tmp_v<-1.0)
								tmp_v =-1.0;
							else if (tmp_v > 1.0)
									tmp_v = 1.0;
							theta = acos(tmp_v); // round to 9 decimal place
							C = fc1*fc2;
							for (int m = 0; m < angle_As_dim; m++) {
								A = pow(2.0, 1-zeta)*pow(1+cos(theta-angle_As[m]), zeta);
								for (int n = 0; n < angle_Rs_dim; n++) {
									B = exp(-eta*((dist1+dist2)/2.0-angle_Rs[n])*((dist1+dist2)/2.0-angle_Rs[n]));
									ANI1_Sym_data[data_pointer+radius_Rs_dim*nele+bond_index*angle_Rs_dim*angle_As_dim+m*angle_Rs_dim+n] = ANI1_Sym_data[data_pointer+radius_Rs_dim*nele+bond_index*angle_Rs_dim*angle_As_dim+m*angle_Rs_dim+n] + A*B*C;
								}
							}
						}
					}
				}
			}
			bond_index = bond_index + 1;
		}
	}
}

//
// This isn't an embedding; it's fast code to make a go-model potential for a molecule.
//
static PyObject* Make_Go(PyObject *self, PyObject  *args) {

	PyArrayObject   *xyz, *DistMat, *Coords, *OutMat;
	int  theatom; // omit the training atom from the RDF
	if (!PyArg_ParseTuple(args, "O!O!O!O!i", &PyArray_Type, &xyz, &PyArray_Type, &DistMat,&PyArray_Type, &OutMat,&PyArray_Type, &Coords, &theatom)) return 0;

	double *xyz_data=(double*) xyz->data;
	double *distmat=(double*) DistMat->data;
	double *outmat=(double*) OutMat->data;
	double *coords=(double*) Coords->data;
	const int natom = (DistMat->dimensions)[0];
	const int nxyz = (xyz->dimensions)[0];
	const int noutmat = (OutMat->dimensions)[0];

	if (nxyz != noutmat)
	std::cout << "Bad input arrays :( " << std::endl;

	for (int s=0; s<nxyz; s++)
	{
		//std::cout << xyz_data[3*s] << " " << xyz_data[3*s+1] << " " << xyz_data[3*s+2] << std::endl;
		for (int i=0; i<natom; i++)
		{
			for (int j=i+1; j<natom; j++)
			{
				if (i==theatom)
				{
					outmat[s] += 0.0125*pow(distmat[i*natom+j]-dist(xyz_data[3*s],xyz_data[3*s+1],xyz_data[3*s+2],coords[3*j],coords[3*j+1],coords[3*j+2]),2.0);
				}
				else if (j==theatom)
				{
					outmat[s] += 0.0125*pow(distmat[i*natom+j]-dist(xyz_data[3*s],xyz_data[3*s+1],xyz_data[3*s+2],coords[3*i],coords[3*i+1],coords[3*i+2]),2.0);
				}
			}
		}
	}
	PyObject* nlist = PyList_New(0);
	return nlist;
}

static PyObject*  Make_RDF(PyObject *self, PyObject  *args) {

	PyArrayObject   *xyz, *grids, *atoms_, *elements;
	double   dist_cut,  width;
	int  ngrids, theatom; // omit the training atom from the RDF
	if (!PyArg_ParseTuple(args, "O!O!O!O!diid",
	&PyArray_Type, &xyz, &PyArray_Type, &grids, &PyArray_Type, &atoms_, &PyArray_Type, &elements, &dist_cut, &ngrids, &theatom, &width))  return NULL;
	PyObject* RDF_all = PyList_New(0);

	//npy_intp* nelep = elements->dimensions;
	const int nele = (elements->dimensions)[0];
	uint8_t* ele=(uint8_t*)elements->data;
	uint8_t* atoms=(uint8_t*)atoms_->data;

	npy_intp  RDFdim[2] = {nele, ngrids};
	PyObject* RDF;
	double *RDF_data, *xyz_data, *grids_data;
	double center[3]; // x y z of the center
	int natom, num_RDF;
	array<std::vector<int>, 100> ele_index;  // hold max 100 elements most
	array<std::vector<double>, 100> ele_dist;  //  hold max 100 elements most

	// for (int i = 0; i < nele; i++)
	//	ele[i] = PyFloat_AsDouble(PyList_GetItem(elements, i));

	npy_intp* Nxyz = xyz->dimensions;
	natom = Nxyz[0];
	npy_intp* Ngrids = grids->dimensions;
	num_RDF = Ngrids[0];
	xyz_data = (double*) xyz->data;
	grids_data = (double*) grids -> data;
	for (int j = 0; j < natom; j++) {
		if (j==theatom)
		continue;
		for (int k=0; k < nele; k++) {
			if (atoms[j] == ele[k])
			ele_index[k].push_back(j);
		}
	}

	for  (int i= 0; i < num_RDF; i++) {
		center[0] = grids_data[i*Ngrids[1]];
		center[1] = grids_data[i*Ngrids[1]+1];
		center[2] = grids_data[i*Ngrids[1]+2];
		RDF = PyArray_SimpleNew(2, RDFdim, NPY_DOUBLE);  //2 is the number of dimensions
		RDF_data = (double*) ((PyArrayObject*) RDF)->data;
		for (int n = 0 ; n < RDFdim[0]*RDFdim[1]; n++)
		RDF_data[n] = 0.0;
		for (int m =0; m < nele; m++) {
			if (ele_index[m].size() > 0 )
			rdf(RDF_data,  ngrids,  ele_index,  m, center, xyz_data,  dist_cut, width);
		}

		PyList_Append(RDF_all, RDF);

	}

	for (int j = 0; j < nele; j++)
	ele_index[j].clear(); // It should scope out anyways.

	PyObject* nlist = PyList_New(0);
	PyList_Append(nlist, RDF_all);
	//         PyList_Append(nlist, AM_all);

	return  nlist;
}

int  Make_AM (const double *center_m, const array<std::vector<int>, 100>  ele_index, const int &nele, const int &center_etype_index, const int &center_index, const double *molxyz_data, npy_intp *Nxyz, double *AM_data, const int &max_near){
	double dist1, dist2, dist3, angel, dist;
	int angel_index = 0;
	array<std::vector<double>, 100> ele_angel;  //  hold max 100 elements most
	array<std::vector<double>, 100> ele_angel_dist;  //  hold max 100 elements most

	for (int i = 0; i < nele; i++)
	for (int j = i; j < nele; j++) {
		for (int m = 0; m < ele_index[i].size(); m++)  {
			if (i == center_etype_index && m == center_index)
			continue;
			for (int n = 0; n < ele_index[j].size(); n++) {
				if ( (i == j && n <= m )|| (j == center_etype_index && n == center_index))
				continue;
				dist1 = sqrt(pow(molxyz_data[ele_index[i][m]*Nxyz[1]+1]-center_m[0], 2.0) + pow(molxyz_data[ele_index[i][m]*Nxyz[1]+2]-center_m[1], 2.0) + pow(molxyz_data[ele_index[i][m]*Nxyz[1]+3]-center_m[2], 2.0));
				dist2 = sqrt(pow(molxyz_data[ele_index[j][n]*Nxyz[1]+1]-center_m[0], 2.0) + pow(molxyz_data[ele_index[j][n]*Nxyz[1]+2]-center_m[1], 2.0) + pow(molxyz_data[ele_index[j][n]*Nxyz[1]+3]-center_m[2], 2.0));
				dist3 = sqrt(pow(molxyz_data[ele_index[j][n]*Nxyz[1]+1]-molxyz_data[ele_index[i][m]*Nxyz[1]+1], 2.0) + pow(molxyz_data[ele_index[j][n]*Nxyz[1]+2]-molxyz_data[ele_index[i][m]*Nxyz[1]+2], 2.0) + pow(molxyz_data[ele_index[j][n]*Nxyz[1]+3]-molxyz_data[ele_index[i][m]*Nxyz[1]+3], 2.0));
				angel = ((dist1*dist1+dist2*dist2-dist3*dist3)/(2.0*dist1*dist2));
				dist = dist1*dist1+dist2*dist2;
				ele_angel_dist[angel_index].push_back(dist);
				ele_angel[angel_index].push_back(angel+2.0) ;  // is an angel do not exist set to 0, this make the range of angel [3, 1]

			}

		}


		std::sort(ele_angel[angel_index].begin(), ele_angel[angel_index].end(), MyComparator(ele_angel_dist[angel_index]));
		std::reverse(ele_angel[angel_index].begin(), ele_angel[angel_index].end());
		for (int k = 0; k < ele_angel[angel_index].size() && k < max_near; k++ ) {
			AM_data[angel_index*max_near+k] = ele_angel[angel_index][k];

		}
		angel_index++;
	}
	return 1;


}

static PyObject*  Make_CM (PyObject *self, PyObject  *args)
{
	//     printtest();
	PyArrayObject *xyz, *grids,  *elements, *atoms_;
	double   dist_cut,  mask, mask_prob,dist;
	int  ngrids,  theatom;
	if (!PyArg_ParseTuple(args, "O!O!O!O!diid",
	&PyArray_Type, &xyz, &PyArray_Type, &grids, &PyArray_Type, &atoms_, &PyArray_Type, &elements, &dist_cut, &ngrids, &theatom, &mask))
	return NULL;
	PyObject* CM_all = PyList_New(0);

	const int nele = (elements->dimensions)[0];
	uint8_t* ele=(uint8_t*)elements->data;
	uint8_t* atoms=(uint8_t*)atoms_->data;

	npy_intp  CMdim[2] = {nele, ngrids};
	PyObject* CM;
	double *CM_data, *xyz_data, *grids_data;
	double center[3]; // x y z of the center
	int natom, num_CM;

	array<std::vector<int>, 100> ele_index;  // hold max 100 elements most
	array<std::vector<int>, 100> ele_index_mask;
	array<std::vector<double>, 100> ele_dist;

	npy_intp* Nxyz = xyz->dimensions;
	natom = Nxyz[0];
	npy_intp* Ngrids = grids->dimensions;
	num_CM = Ngrids[0];
	xyz_data = (double*) ((PyArrayObject*) xyz)->data;
	grids_data = (double*) grids -> data;

	for (int j = 0; j < natom; j++) {
		if (j==theatom)
		continue;
		for (int k=0; k < nele; k++) {
			if (atoms[j] == ele[k])
			ele_index[k].push_back(j);
		}
	}
	//    for (int j = 0; j < nele; j++)
	//       for (auto k = ele_index[j].begin(); k != ele_index[j].end(); ++k)
	//         std::cout << i << "  "<< j <<  "  " <<  *k << std::endl;

	for (int i = 0; i < num_CM;  i++) {  //loop over different atoms of the same type
		center[0] = grids_data[i*Ngrids[1]];
		center[1] = grids_data[i*Ngrids[1]+1];
		center[2] = grids_data[i*Ngrids[1]+2];

		CM = PyArray_SimpleNew(2, CMdim, NPY_DOUBLE);
		CM_data = (double*) ((PyArrayObject*) CM)->data;
		for (int n = 0 ; n < CMdim[0]*CMdim[1]; n++)
		CM_data[n] = 0.0;

		for (int j = 0; j < nele; j++) {
			ele_index_mask[j].clear();
		}
		for (int m = 0 ; m < nele; m++) {
			for(int k = 0; k < ele_index[m].size(); k++) {
				mask_prob = static_cast <float> (rand()) / static_cast <float> (RAND_MAX/(double(1.0)));
				if (mask_prob > mask) {
					ele_index_mask[m].push_back(ele_index[m][k]);
				}
			}
		}

		for (int m = 0; m < nele; m++) {
			for (int k = 0; k < ele_index_mask[m].size(); k++) {
				dist=sqrt(pow(xyz_data[ele_index_mask[m][k]*Nxyz[1]]-center[0], 2.0) + pow(xyz_data[ele_index_mask[m][k]*Nxyz[1]+1]-center[1], 2.0) + pow(xyz_data[ele_index_mask[m][k]*Nxyz[1]+2]-center[2], 2.0));
				//           std::cout<<disp<<"  "<<dist<<std::endl;
				//     std::cout<<" "<<m<<"  "<<k<<"  "<<1/dist*(1 - erf(4*(dist-dist_cut)))/2<<"  "<<1/dist<<std::endl;
				//if (dist > 0.5)
				//ele_dist[m].push_back(1/dist*(1 - erf(4*(dist-dist_cut)))/2);    // add a smooth cut erf function with 1/x
				//else
				//ele_dist[m].push_back(-4*dist*dist+3);  // when dist< 0.5, replace 1/x with -4x^2+3 to ensure converge
				ele_dist[m].push_back(1/dist);
			}
		}

		for (int m = 0; m < nele; m++) {
			std::sort(ele_dist[m].begin(), ele_dist[m].end());
			std::reverse(ele_dist[m].begin(), ele_dist[m].end());
		}

		for (int m = 0;  m < nele; m++)
		for (int n= 0; n < ele_dist[m].size() && n < ngrids; n++)
		CM_data[m*CMdim[1]+n] = ele_dist[m][n];

		for (int m = 0; m < nele; m++)
		ele_dist[m].clear();

		PyList_Append(CM_all, CM);

	}

	for (int j = 0; j < nele; j++) {
		ele_index[j].clear();
		ele_index_mask[j].clear();
	}
	PyObject* nlist = PyList_New(0);
	PyList_Append(nlist, CM_all);
	return  nlist;
}

//
// Embed molecule using Gaussian X Spherical Harmonic basis
// Does all atoms if theatom argument < 0
// These vectors are flattened [gau,l,m] arrays
//
static PyObject* Make_SH(PyObject *self, PyObject  *args)
{
	PyArrayObject *xyz, *grids,  *elements, *atoms_;
	double   dist_cut;
	int  ngrids,  theatom;
	PyObject *Prm_;
	if (!PyArg_ParseTuple(args, "O!O!O!i", &PyDict_Type, &Prm_, &PyArray_Type, &xyz, &PyArray_Type, &atoms_ , &theatom))
	return NULL;

	SHParams Prmo = ParseParams(Prm_);SHParams* Prm=&Prmo;
	uint8_t* atoms=(uint8_t*)atoms_->data;
	npy_intp* Nxyz = xyz->dimensions; // Now assumed to be natomX3.
	int natom;
	natom = Nxyz[0];

	//npy_intp outdim[2] = {natom,SH_NRAD*(1+SH_LMAX)*(1+SH_LMAX)};
	npy_intp outdim[2] = {1,Prm->SH_NRAD*(1+Prm->SH_LMAX)*(1+Prm->SH_LMAX)};
	if (theatom<0)
	outdim[0] = natom;
	PyObject* SH = PyArray_ZEROS(2, outdim, NPY_DOUBLE, 0);

	double *SH_data, *xyz_data, *grids_data;
	double center[3]; // x y z of the center
	xyz_data = (double*) ((PyArrayObject*) xyz)->data;
	// Note that Grids are currently unused...
	//grids_data = (double*) grids -> data;
	SH_data = (double*) ((PyArrayObject*)SH)->data;
	int SHdim = SizeOfGauSH(Prm);
	if (theatom<0)
	{
		#pragma omp parallel for
		for (int i=0; i<natom; ++i)
		{
			double xc = xyz_data[i*3];
			double yc = xyz_data[i*3+1];
			double zc = xyz_data[i*3+2];
			for (int j = 0; j < natom; j++)
			{
				double x = xyz_data[j*3];
				double y = xyz_data[j*3+1];
				double z = xyz_data[j*3+2];
				//RadSHProjection(x-xc,y-yc,z-zc,SH_data + i*SH_NRAD*(1+SH_LMAX)*(1+SH_LMAX), natom);
				//RadSHProjection(Prm,x-xc,y-yc,z-zc,SH_data + i*SHdim, (double)atoms[j]);
				RadSHProjection(Prm,x-xc,y-yc,z-zc,SH_data + i*SHdim, Prm->ANES[atoms[j]-1]);
			}
		}
	}
	else{
		int i = theatom;
		int ai=0;

		double xc = xyz_data[i*3];
		double yc = xyz_data[i*3+1];
		double zc = xyz_data[i*3+2];

		for (int j = 0; j < natom; j++)
		{
			double x = xyz_data[j*3];
			double y = xyz_data[j*3+1];
			double z = xyz_data[j*3+2];
			RadSHProjection(Prm,x-xc,y-yc,z-zc,SH_data + ai*SHdim, Prm->ANES[atoms[j]-1]);
		}
	}
	return SH;
}

//
// Embed molecule using Gaussian X Spherical Harmonic basis
// Does all atoms if theatom argument < 0
// These vectors are flattened [gau,l,m] arrays
// This version embeds a single atom after several transformations.
//
static PyObject* Make_SH_Transf(PyObject *self, PyObject  *args)
{
	PyArrayObject *xyz, *grids,  *elements, *atoms_, *transfs;
	double   dist_cut;
	int  theatom;
	PyObject *Prm_;
	if (!PyArg_ParseTuple(args, "O!O!O!iO!", &PyDict_Type, &Prm_, &PyArray_Type, &xyz, &PyArray_Type, &atoms_ , &theatom, &PyArray_Type, &transfs ))
	return NULL;

	SHParams Prmo = ParseParams(Prm_);SHParams* Prm=&Prmo;
	uint8_t* atoms=(uint8_t*)atoms_->data;

	npy_intp* Nxyz = xyz->dimensions; // Now assumed to be natomX3.
	npy_intp* Ntrans = transfs->dimensions; // Now assumed to be natomX3.
	int natom, ntr;

	natom = Nxyz[0];
	ntr = Ntrans[0];

	npy_intp outdim[2] = {ntr,Prm->SH_NRAD*(1+Prm->SH_LMAX)*(1+Prm->SH_LMAX)};
	PyObject* SH = PyArray_ZEROS(2, outdim, NPY_DOUBLE, 0);

	double *SH_data, *xyz_data, *grids_data, *t_data;
	double center[3]; // x y z of the center
	xyz_data = (double*) ((PyArrayObject*) xyz)->data;
	t_data = (double*) ((PyArrayObject*) transfs)->data;
	// Note that Grids are currently unused...
	//grids_data = (double*) grids -> data;
	SH_data = (double*) ((PyArrayObject*)SH)->data;
	double t_coords0[natom][3];
	int SHdim = SizeOfGauSH(Prm);

	{
		// Center the atom and then perform the transformations.

		double xc = xyz_data[theatom*3];
		double yc = xyz_data[theatom*3+1];
		double zc = xyz_data[theatom*3+2];

		for (int j = 0; j < natom; j++)
		{
			t_coords0[j][0] = xyz_data[j*3] - xc;
			t_coords0[j][1] = xyz_data[j*3+1] - yc;
			t_coords0[j][2] = xyz_data[j*3+2] - zc;
		}

		#pragma omp parallel for
		for (int i=0; i<ntr; ++i)
		{
			double* tr = t_data+i*9;
			/*
			cout << tr[0] << tr[1] << tr[2] << endl;
			cout << tr[3] << tr[4] << tr[5] << endl;
			cout << tr[6] << tr[7] << tr[8] << endl;
			*/
			for (int j = 0; j < natom; j++)
			{
				// Perform the transformation, embed and out...
				double x = (tr[0*3]*t_coords0[j][0]+tr[0*3+1]*t_coords0[j][1]+tr[0*3+2]*t_coords0[j][2]);
				double y = (tr[1*3]*t_coords0[j][0]+tr[1*3+1]*t_coords0[j][1]+tr[1*3+2]*t_coords0[j][2]);
				double z = (tr[2*3]*t_coords0[j][0]+tr[2*3+1]*t_coords0[j][1]+tr[2*3+2]*t_coords0[j][2]);
				RadSHProjection(Prm, x, y, z, SH_data + i*SHdim, Prm->ANES[atoms[j]-1]);
			}
		}
	}
	return SH;
}

static PyObject* Make_SH_EleUniq(PyObject *self, PyObject  *args)
{
	PyArrayObject *xyz, *grids,  *elements, *atoms_;
	double   dist_cut;
	int  theatom;
	PyObject *Prm_;
	if (!PyArg_ParseTuple(args, "O!O!O!i", &PyDict_Type, &Prm_, &PyArray_Type, &xyz, &PyArray_Type, &atoms_ , &theatom))
	return NULL;

	SHParams Prmo = ParseParams(Prm_);SHParams* Prm=&Prmo;
	const int nele = (elements->dimensions)[0];
	uint8_t* ele=(uint8_t*)elements->data;
	uint8_t* atoms=(uint8_t*)atoms_->data;
	npy_intp* Nxyz = xyz->dimensions;
	int natom;
	natom = Nxyz[0];

	//npy_intp outdim[2] = {natom,SH_NRAD*(1+SH_LMAX)*(1+SH_LMAX)};
	npy_intp outdim[2] = {1,Prm->SH_NRAD*(1+Prm->SH_LMAX)*(1+Prm->SH_LMAX)};
	if (theatom<0)
	outdim[0] = natom;
	PyObject* SH = PyArray_ZEROS(2, outdim, NPY_DOUBLE, 0);

	double *SH_data, *xyz_data, *grids_data;
	double center[3]; // x y z of the center
	xyz_data = (double*) ((PyArrayObject*) xyz)->data;
	// Note that Grids are currently unused...
	//grids_data = (double*) grids -> data;
	SH_data = (double*) ((PyArrayObject*)SH)->data;
	int SHdim = SizeOfGauSH(Prm);
	if (theatom<0)
	{
		#pragma omp parallel for
		for (int i=0; i<natom; ++i)
		{
			double xc = xyz_data[i*Nxyz[1]];
			double yc = xyz_data[i*Nxyz[1]+1];
			double zc = xyz_data[i*Nxyz[1]+2];
			for (int j = 0; j < natom; j++)
			{
				double x = xyz_data[j*Nxyz[1]];
				double y = xyz_data[j*Nxyz[1]+1];
				double z = xyz_data[j*Nxyz[1]+2];
				//RadSHProjection(x-xc,y-yc,z-zc,SH_data + i*SH_NRAD*(1+SH_LMAX)*(1+SH_LMAX), natom);
				RadSHProjection_Orth_EleUniq(Prm,x-xc,y-yc,z-zc,SH_data + i*SHdim, (double)atoms[j]);
			}
		}
	}
	else{
		int i = theatom;
		int ai=0;

		double xc = xyz_data[i*Nxyz[1]];
		double yc = xyz_data[i*Nxyz[1]+1];
		double zc = xyz_data[i*Nxyz[1]+2];

		for (int j = 0; j < natom; j++)
		{
			double x = xyz_data[j*Nxyz[1]];
			double y = xyz_data[j*Nxyz[1]+1];
			double z = xyz_data[j*Nxyz[1]+2];
			RadSHProjection(Prm,x-xc,y-yc,z-zc,SH_data + ai*SHdim, (double)atoms[j]);
		}
	}
	return SH;
}

//
// Embed molecule using Gaussian X Spherical Harmonic basis
// Does all atoms... just for debug really.
// Make invariant embedding of all the atoms.
// This is not a reversible embedding.
static PyObject* Make_Inv(PyObject *self, PyObject  *args)
{
	PyArrayObject *xyz, *grids,  *elements, *atoms_;
	double   dist_cut,  mask, mask_prob,dist;
	int theatom;
	PyObject* Prm_;
	if (!PyArg_ParseTuple(args, "O!O!O!i",
	&PyDict_Type, &Prm_, &PyArray_Type, &xyz, &PyArray_Type, &atoms_ , &theatom))
	return NULL;

	SHParams Prmo = ParseParams(Prm_);SHParams* Prm=&Prmo;
	uint8_t* ele=(uint8_t*)elements->data;
	uint8_t* atoms=(uint8_t*)atoms_->data;
	npy_intp* Nxyz = xyz->dimensions;
	int natom, num_CM;
	natom = Nxyz[0];

	npy_intp outdim[2];
	if (theatom>=0)
	outdim[0] = 1;
	else
	outdim[0] = natom;
	outdim[1]=Prm->SH_NRAD*(1+Prm->SH_LMAX);
	PyObject* SH = PyArray_ZEROS(2, outdim, NPY_DOUBLE, 0);

	double *SH_data, *xyz_data, *grids_data;
	xyz_data = (double*) ((PyArrayObject*) xyz)->data;
	//grids_data = (double*) grids -> data;
	SH_data = (double*) ((PyArrayObject*)SH)->data;

	/*double center[3]={0.,0.,0.};
	for (int j = 0; j < natom; j++)
	{
	center[0]+=xyz_data[j*Nxyz[1]+0];
	center[1]+=xyz_data[j*Nxyz[1]+1];
	center[2]+=xyz_data[j*Nxyz[1]+2];
}
center[0]/= natom;
center[1]/= natom;
center[2]/= natom;*/

if (theatom >= 0)
{
	int i = theatom;
	double xc = xyz_data[i*Nxyz[1]+0];
	double yc = xyz_data[i*Nxyz[1]+1];
	double zc = xyz_data[i*Nxyz[1]+2];
	for (int j = 0; j < natom; j++)
	{
		double x = xyz_data[j*Nxyz[1]+0];
		double y = xyz_data[j*Nxyz[1]+1];
		double z = xyz_data[j*Nxyz[1]+2];
		RadInvProjection(Prm, x-xc,y-yc,z-zc,SH_data,(double)atoms[j]);
	}
}
else
{
	#pragma omp parallel for
	for (int i=0; i<natom; ++i )
	{
		double xc = xyz_data[i*Nxyz[1]+0];
		double yc = xyz_data[i*Nxyz[1]+1];
		double zc = xyz_data[i*Nxyz[1]+2];
		for (int j = 0; j < natom; j++)
		{
			double x = xyz_data[j*Nxyz[1]+0];
			double y = xyz_data[j*Nxyz[1]+1];
			double z = xyz_data[j*Nxyz[1]+2];
			RadInvProjection(Prm, x-xc,y-yc,z-zc,SH_data+i*(outdim[1]),(double)atoms[j]);
		}
	}
}
return SH;
}

//
// returns a [Nrad X Nang] x [npts] array which contains rasterized versions of the
// non-orthogonal basis functions.
//
static PyObject* Raster_SH(PyObject *self, PyObject  *args)
{
	PyArrayObject *xyz, *grids,  *elements, *atoms_;
	double   dist_cut,  mask, mask_prob,dist;
	int theatom;
	PyObject* Prm_;
	if (!PyArg_ParseTuple(args, "O!O!", &PyDict_Type, &Prm_, &PyArray_Type, &xyz))
	return NULL;

	SHParams Prmo = ParseParams(Prm_);SHParams* Prm=&Prmo;
	const int npts = (xyz->dimensions)[0];
	int nbas = Prm->SH_NRAD*(1+Prm->SH_LMAX)*(1+Prm->SH_LMAX);
	npy_intp outdim[2] = {nbas,npts};
	PyObject* SH = PyArray_ZEROS(2, outdim, NPY_DOUBLE, 0);

	double *SH_data, *xyz_data;
	double center[3]; // x y z of the center
	xyz_data = (double*) ((PyArrayObject*) xyz)->data;
	SH_data = (double*) ((PyArrayObject*)SH)->data;

	for (int pt=0; pt<npts; ++pt)
	{
		double r = sqrt(xyz_data[pt*3+0]*xyz_data[pt*3+0]+xyz_data[pt*3+1]*xyz_data[pt*3+1]+xyz_data[pt*3+2]*xyz_data[pt*3+2]);
		double theta = acos(xyz_data[pt*3+2]/r);
		double phi = atan2(xyz_data[pt*3+1],xyz_data[pt*3+0]);

		//		cout << "r" << r << endl;
		//		cout << "r" << theta << endl;
		//		cout << "r" << phi << endl;

		int bi = 0;
		for (int i=0; i<Prm->SH_NRAD ; ++i)
		{
			double Gv = Gau(r, Prm->RBFS[i*2],Prm->RBFS[i*2+1]);
			for (int l=0; l<Prm->SH_LMAX+1 ; ++l)
			{
				//				cout << "l=" << l << " Gv "<< Gv <<endl;
				for (int m=-l; m<l+1 ; ++m)
				{
					SH_data[bi*npts+pt] += Gv*RealSphericalHarmonic(l,m,theta,phi);
					//					cout << SH_data[bi*npts+pt] << endl;
					++bi;
				}
			}
		}
	}
	return SH;
}

// Gives the projection of a delta function at xyz
static PyObject* Project_SH(PyObject *self, PyObject  *args)
{
	double x,y,z;
	PyArrayObject *xyz;
	PyObject* Prm_;
	if (!PyArg_ParseTuple(args, "O!O!", &PyDict_Type, &Prm_, &PyArray_Type, &xyz))
	return NULL;

	SHParams Prmo = ParseParams(Prm_);SHParams* Prm=&Prmo;
	int nbas = Prm->SH_NRAD*(1+Prm->SH_LMAX)*(1+Prm->SH_LMAX);
	npy_intp outdim[2] = {1,nbas};
	PyObject* SH = PyArray_ZEROS(2, outdim, NPY_DOUBLE, 0);
	double *SH_data,*xyz_data;

	xyz_data = (double*) ((PyArrayObject*) xyz)->data;
	x=xyz_data[0];
	y=xyz_data[1];
	z=xyz_data[2];
	SH_data = (double*) ((PyArrayObject*)SH)->data;
	int Nbas = Prm->SH_NRAD*(1+Prm->SH_LMAX)*(1+Prm->SH_LMAX);
	//	int Nang = (1+SH_LMAX)*(1+SH_LMAX);
	double r = sqrt(x*x+y*y+z*z);

	if (r<pow(10.0,-11.0))
	return SH;
	double theta = acos(z/r);
	double phi = atan2(y,x);

	//	cout << "R,theta,phi" << r << " " << theta << " " <<phi << " " <<endl;
	int bi = 0;
	for (int i=0; i<Prm->SH_NRAD ; ++i)
	{
		double Gv = Gau(r, Prm->RBFS[i*2],Prm->RBFS[i*2+1]);
		cout << Prm->RBFS[i*2] << " " << Gv << endl;

		for (int l=0; l<Prm->SH_LMAX+1 ; ++l)
		{
			for (int m=-l; m<l+1 ; ++m)
			{
				SH_data[bi] += Gv*RealSphericalHarmonic(l,m,theta,phi);
				++bi;
			}
		}
	}
	return SH;
}

static PyObject* Make_DistMat(PyObject *self, PyObject  *args)
{
	PyArrayObject *xyz;
	if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &xyz))
	return NULL;
	const int nat = (xyz->dimensions)[0];
	npy_intp outdim[2] = {nat,nat};
	PyObject* SH = PyArray_ZEROS(2, outdim, NPY_DOUBLE, 0);
	double *SH_data,*xyz_data;
	xyz_data = (double*) ((PyArrayObject*) xyz)->data;
	SH_data = (double*) ((PyArrayObject*)SH)->data;
	for (int i=0; i < nat; ++i)
	for (int j=i+1; j < nat; ++j)
	{
		SH_data[i*nat+j] = sqrt((xyz_data[i*3+0]-xyz_data[j*3+0])*(xyz_data[i*3+0]-xyz_data[j*3+0])+(xyz_data[i*3+1]-xyz_data[j*3+1])*(xyz_data[i*3+1]-xyz_data[j*3+1])+(xyz_data[i*3+2]-xyz_data[j*3+2])*(xyz_data[i*3+2]-xyz_data[j*3+2])) + 0.00000000001;
		SH_data[j*nat+i] = SH_data[i*nat+j];
	}
	return SH;
}

//
// Make a neighborlist using a naive, quadratic algorithm.
// returns a python list.
// Only does up to the nreal-th atom. (for periodic tesselation)
//
static PyObject* Make_NListNaive(PyObject *self, PyObject  *args)
{
	PyArrayObject *xyz;
	double rng;
	int nreal;
	if (!PyArg_ParseTuple(args, "O!di", &PyArray_Type, &xyz, &rng, &nreal))
		return NULL;
	double *xyz_data;
	xyz_data = (double*) ((PyArrayObject*) xyz)->data;
	const int nat = (xyz->dimensions)[0];
// Avoid stupid python reference counting issues by just using std::vector...
std::vector< std::vector<int> > tmp(nat);
#pragma omp parallel for
	for (int i=0; i < nreal; ++i)
	{
		for (int j=i+1; j < nat; ++j)
		{
			double dij = sqrt((xyz_data[i*3+0]-xyz_data[j*3+0])*(xyz_data[i*3+0]-xyz_data[j*3+0])+(xyz_data[i*3+1]-xyz_data[j*3+1])*(xyz_data[i*3+1]-xyz_data[j*3+1])+(xyz_data[i*3+2]-xyz_data[j*3+2])*(xyz_data[i*3+2]-xyz_data[j*3+2])) + 0.00000000001;
			if (dij < rng)
			{
				tmp[i].push_back(j);
				// For now we're not doing the permutations...
			}
		}
	}
	PyObject* Tore = PyList_New(nreal);
	for (int i=0; i < nreal; ++i)
	{
		PyObject* tl = PyList_New(tmp[i].size());
		for (int j=0; j<tmp[i].size();++j)
		{
			PyObject* ti = PyInt_FromLong(tmp[i][j]);
			PyList_SetItem(tl,j,ti);
		}
		PyList_SetItem(Tore,i,tl);
	}
	return Tore;
}

static PyObject* DipoleAutoCorr(PyObject *self, PyObject  *args)
{
	PyArrayObject *xyz;
	if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &xyz))
	return NULL;
	const int nat = (xyz->dimensions)[0];
	npy_intp outdim[2] = {nat,1};
	PyObject* SH = PyArray_ZEROS(2, outdim, NPY_DOUBLE, 0);
	double *SH_data,*xyz_data;
	xyz_data = (double*) ((PyArrayObject*) xyz)->data;
	SH_data = (double*) ((PyArrayObject*)SH)->data;
#pragma omp parallel for
	for (int i=0; i < nat; ++i) // Distance between points.
	{
		for (int j=0; j < nat-i; ++j) // points to sum over.
		{
			SH_data[i] += (xyz_data[j*3+0]*xyz_data[(j+i)*3+0]+xyz_data[j*3+1]*xyz_data[(j+i)*3+1]+xyz_data[j*3+2]*xyz_data[(j+i)*3+2]);
		}
		SH_data[i] /= double(nat-i);
	}
	return SH;
}

static PyObject* Norm_Matrices(PyObject *self, PyObject *args)
{
	PyArrayObject *dmat1, *dmat2;
	if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &dmat1, &PyArray_Type, &dmat2))
	return NULL;
	double norm = 0;
	const int dim1 = (dmat1->dimensions)[0];
	const int dim2 = (dmat1->dimensions)[1];
	double *dmat1_data, *dmat2_data;
	double normmat[dim1*dim2];
	dmat1_data = (double*) ((PyArrayObject*)dmat1)->data;
	dmat2_data = (double*) ((PyArrayObject*)dmat2)->data;
	#ifdef OPENMP
	#pragma omp parallel for reduction(+:norm)
	#endif
	for (int i=0; i < dim1; ++i)
	for (int j=0; j < dim2; ++j)
	norm += (dmat1_data[i*dim2+j] - dmat2_data[i*dim2+j])*(dmat1_data[i*dim2+j] - dmat2_data[i*dim2+j]);
	return PyFloat_FromDouble(sqrt(norm));
}

static PyObject* Make_GoForce(PyObject *self, PyObject  *args)
{
	PyArrayObject *xyz, *EqDistMat;
	int at, spherical;
	if (!PyArg_ParseTuple(args, "O!O!ii", &PyArray_Type, &xyz, &PyArray_Type, &EqDistMat, &at, &spherical))
	return NULL;
	const int nat = (xyz->dimensions)[0];
	npy_intp outdim[2] = {nat,3};
	if (at>=0)
	outdim[0] = 1;
	PyObject* hess = PyArray_ZEROS(2, outdim, NPY_DOUBLE, 0);
	double *frc_data,*xyz_data,*d_data;
	xyz_data = (double*) ((PyArrayObject*)xyz)->data;
	frc_data = (double*) ((PyArrayObject*)hess)->data;
	d_data = (double*) ((PyArrayObject*)EqDistMat)->data;
	double u[3]={0.0,0.0,0.0};
	if (at<0)
	{
		#pragma omp parallel for
		for (int i=0; i < nat; ++i)
		{
			for (int j=0; j < nat; ++j)
			{
				if (i==j)
				continue;
				double dij = sqrt((xyz_data[i*3+0]-xyz_data[j*3+0])*(xyz_data[i*3+0]-xyz_data[j*3+0])+(xyz_data[i*3+1]-xyz_data[j*3+1])*(xyz_data[i*3+1]-xyz_data[j*3+1])+(xyz_data[i*3+2]-xyz_data[j*3+2])*(xyz_data[i*3+2]-xyz_data[j*3+2]));
				u[0] = (xyz_data[j*3]-xyz_data[i*3])/dij;
				u[1] = (xyz_data[j*3+1]-xyz_data[i*3+1])/dij;
				u[2] = (xyz_data[j*3+2]-xyz_data[i*3+2])/dij;
				frc_data[i*3+0] += -2*(dij-d_data[i*nat+j])*u[0];
				frc_data[i*3+1] += -2*(dij-d_data[i*nat+j])*u[1];
				frc_data[i*3+2] += -2*(dij-d_data[i*nat+j])*u[2];
			}
		}
		if (spherical)
		for (int i=0; i < nat; ++i)
		CartToSphere(frc_data+i*3);
	}
	else
	{
		int i=at;
		for (int j=0; j < nat; ++j)
		{
			if (i==j)
			continue;
			double dij = sqrt((xyz_data[i*3+0]-xyz_data[j*3+0])*(xyz_data[i*3+0]-xyz_data[j*3+0])+(xyz_data[i*3+1]-xyz_data[j*3+1])*(xyz_data[i*3+1]-xyz_data[j*3+1])+(xyz_data[i*3+2]-xyz_data[j*3+2])*(xyz_data[i*3+2]-xyz_data[j*3+2]));
			u[0] = (xyz_data[j*3]-xyz_data[i*3])/dij;
			u[1] = (xyz_data[j*3+1]-xyz_data[i*3+1])/dij;
			u[2] = (xyz_data[j*3+2]-xyz_data[i*3+2])/dij;
			//std::cout<<i<<"  "<<j<<"  "<<'dij'<<"  "<<dij<<"  "<<(-2*(dij-d_data[i*nat+j])*u[0])<<"  "<<(-2*(dij-d_data[i*nat+j])*u[1])<<"  "<<(-2*(dij-d_data[i*nat+j])*u[2])<<std::endl;
			frc_data[0] += -2*(dij-d_data[i*nat+j])*u[0];
			frc_data[1] += -2*(dij-d_data[i*nat+j])*u[1];
			frc_data[2] += -2*(dij-d_data[i*nat+j])*u[2];
		}
		if (spherical)
		CartToSphere(frc_data);
	}
	return hess;
}

static PyObject* Make_GoForceLocal(PyObject *self, PyObject  *args)
{
	PyArrayObject *xyz, *EqDistMat;
	int at;
	if (!PyArg_ParseTuple(args, "O!O!i", &PyArray_Type, &xyz, &PyArray_Type, &EqDistMat, &at))
	return NULL;
	const int nat = (xyz->dimensions)[0];
	npy_intp outdim[2] = {nat,3};
	if (at>=0)
	outdim[0] = 1;
	PyObject* hess = PyArray_ZEROS(2, outdim, NPY_DOUBLE, 0);
	double *frc_data,*xyz_data,*d_data;
	xyz_data = (double*) ((PyArrayObject*)xyz)->data;
	frc_data = (double*) ((PyArrayObject*)hess)->data;
	d_data = (double*) ((PyArrayObject*)EqDistMat)->data;
	double u[3]={0.0,0.0,0.0};
	if (at<0)
	{
		for (int i=0; i < nat; ++i)
		{
			for (int j=0; j < nat; ++j)
			{
				if (i==j)
				continue;
				double dij = sqrt((xyz_data[i*3+0]-xyz_data[j*3+0])*(xyz_data[i*3+0]-xyz_data[j*3+0])+(xyz_data[i*3+1]-xyz_data[j*3+1])*(xyz_data[i*3+1]-xyz_data[j*3+1])+(xyz_data[i*3+2]-xyz_data[j*3+2])*(xyz_data[i*3+2]-xyz_data[j*3+2]));
				if (dij > 15.0)
				continue;
				u[0] = (xyz_data[j*3]-xyz_data[i*3])/dij;
				u[1] = (xyz_data[j*3+1]-xyz_data[i*3+1])/dij;
				u[2] = (xyz_data[j*3+2]-xyz_data[i*3+2])/dij;
				frc_data[i*3+0] += -2*(dij-d_data[i*nat+j])*u[0];
				frc_data[i*3+1] += -2*(dij-d_data[i*nat+j])*u[1];
				frc_data[i*3+2] += -2*(dij-d_data[i*nat+j])*u[2];
			}
		}
	}
	else
	{
		int i=at;
		for (int j=0; j < nat; ++j)
		{
			if (i==j)
			continue;
			double dij = sqrt((xyz_data[i*3+0]-xyz_data[j*3+0])*(xyz_data[i*3+0]-xyz_data[j*3+0])+(xyz_data[i*3+1]-xyz_data[j*3+1])*(xyz_data[i*3+1]-xyz_data[j*3+1])+(xyz_data[i*3+2]-xyz_data[j*3+2])*(xyz_data[i*3+2]-xyz_data[j*3+2]));
			if (dij > 15.0)
			continue;
			u[0] = (xyz_data[j*3]-xyz_data[i*3])/dij;
			u[1] = (xyz_data[j*3+1]-xyz_data[i*3+1])/dij;
			u[2] = (xyz_data[j*3+2]-xyz_data[i*3+2])/dij;
			frc_data[0] += -2*(dij-d_data[i*nat+j])*u[0];
			frc_data[1] += -2*(dij-d_data[i*nat+j])*u[1];
			frc_data[2] += -2*(dij-d_data[i*nat+j])*u[2];
		}
	}
	return hess;
}

static PyObject* Make_GoHess(PyObject *self, PyObject  *args)
{
	PyArrayObject *xyz, *EqDistMat;
	if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &xyz, &PyArray_Type, &EqDistMat))
	return NULL;
	const int nat = (xyz->dimensions)[0];
	npy_intp outdim[2] = {3*nat,3*nat};
	PyObject* hess = PyArray_ZEROS(2, outdim, NPY_DOUBLE, 0);
	double *hess_data,*xyz_data,*d_data;
	xyz_data = (double*) ((PyArrayObject*)xyz)->data;
	hess_data = (double*) ((PyArrayObject*)hess)->data;
	d_data = (double*) ((PyArrayObject*)EqDistMat)->data;
	double u[3]={0.0,0.0,0.0};
	for (int i=0; i < nat; ++i)
	{
		for (int j=0; j < nat; ++j)
		{
			if (i==j)
			continue;
			double dij = sqrt((xyz_data[i*3+0]-xyz_data[j*3+0])*(xyz_data[i*3+0]-xyz_data[j*3+0])+(xyz_data[i*3+1]-xyz_data[j*3+1])*(xyz_data[i*3+1]-xyz_data[j*3+1])+(xyz_data[i*3+2]-xyz_data[j*3+2])*(xyz_data[i*3+2]-xyz_data[j*3+2]));
			u[0] = (xyz_data[j*3]-xyz_data[i*3])/dij;
			u[1] = (xyz_data[j*3+1]-xyz_data[i*3+1])/dij;
			u[2] = (xyz_data[j*3+2]-xyz_data[i*3+2])/dij;
			for (int ip=0; ip < 3; ++ip)
			{
				for (int jp=0; jp < 3; ++jp)
				{
					if (ip==jp)
					{
						hess_data[(i*3+ip)*3*nat+i*3+ip] += 2.0*(u[ip]*u[ip]- (dij-d_data[i*nat+j])*(u[ip]*u[ip])/dij + (dij-d_data[i*nat+j])/dij);
						hess_data[(i*3+ip)*3*nat+j*3+jp] = 2.0*(-u[ip]*u[ip] + (dij-d_data[i*nat+j])*(u[ip]*u[ip])/dij - (dij-d_data[i*nat+j])/dij);
					}
					else
					{
						hess_data[(i*3+ip)*3*nat+i*3+jp] += 2.0*(u[ip]*u[jp] - (dij-d_data[i*nat+j])*u[ip]*u[jp]/dij);
						hess_data[(i*3+ip)*3*nat+j*3+jp] = 2.0*(-u[ip]*u[jp] + (dij-d_data[i*nat+j])*u[ip]*u[jp]/dij);
					}
				}
			}
		}
	}
	return hess;
}

static PyObject* Make_LJForce(PyObject *self, PyObject  *args)
{
	PyArrayObject *xyz, *EqDistMat, *eps;
	int at;
	if (!PyArg_ParseTuple(args, "O!O!O!i", &PyArray_Type, &xyz, &PyArray_Type, &EqDistMat, &PyArray_Type, &eps, &at ))
	return NULL;
	const int nat = (xyz->dimensions)[0];
	npy_intp outdim[2] = {nat,3};
	if (at>=0)
	outdim[0] = 1;
	PyObject* hess = PyArray_ZEROS(2, outdim, NPY_DOUBLE, 0);
	double *frc_data,*xyz_data,*eps_data,*d_data;
	xyz_data = (double*) ((PyArrayObject*)xyz)->data;
	eps_data = (double*) ((PyArrayObject*)eps)->data;
	frc_data = (double*) ((PyArrayObject*)hess)->data;
	d_data = (double*) ((PyArrayObject*)EqDistMat)->data;
	double u[3]={0.0,0.0,0.0};

	if (at<0)
	{
		for (int i=0; i < nat; ++i)
		{
			for (int j=0; j < nat; ++j)
			{
				if (i==j)
				continue;
				double dij = (pow((xyz_data[i*3+0]-xyz_data[j*3+0]), 2.0) + pow((xyz_data[i*3+1]-xyz_data[j*3+1]), 2.0) + pow((xyz_data[i*3+2]-xyz_data[j*3+2]), 2.0));
				u[0] = (xyz_data[i*3] - xyz_data[j*3]);
				u[1] = (xyz_data[i*3+1] - xyz_data[j*3+1]);
				u[2] = (xyz_data[i*3+2] - xyz_data[j*3+2]);
				frc_data[i*3+0] += -eps_data[i*nat+j]*((12.*pow(d_data[i*nat+j],12.0)/pow(dij,7.0))*u[0]-(12.*pow(d_data[i*nat+j],6.0)/pow(dij,4.0))*u[0]);
				frc_data[i*3+1] += -eps_data[i*nat+j]*((12.*pow(d_data[i*nat+j],12.0)/pow(dij,7.0))*u[1]-(12.*pow(d_data[i*nat+j],6.0)/pow(dij,4.0))*u[1]);
				frc_data[i*3+2] += -eps_data[i*nat+j]*((12.*pow(d_data[i*nat+j],12.0)/pow(dij,7.0))*u[2]-(12.*pow(d_data[i*nat+j],6.0)/pow(dij,4.0))*u[2]);

			}
		}
	}
	else
	{
		int i=at;

		for (int j=0; j < nat; ++j)
		{
			if (i==j)
			continue;
			double dij = (pow((xyz_data[i*3+0]-xyz_data[j*3+0]), 2.0) + pow((xyz_data[i*3+1]-xyz_data[j*3+1]), 2.0) + pow((xyz_data[i*3+2]-xyz_data[j*3+2]), 2.0));
			u[0] = (xyz_data[i*3] - xyz_data[j*3])/dij;
			u[1] = (xyz_data[i*3+1] - xyz_data[j*3+1])/dij;
			u[2] = (xyz_data[i*3+2] - xyz_data[j*3+2])/dij;
			frc_data[0] += -eps_data[i*nat+j]*((12.*pow(d_data[i*nat+j],12.0)/pow(dij,7.0))*u[0]-(12.*pow(d_data[i*nat+j],6.0)/pow(dij,4.0))*u[0]);
			frc_data[1] += -eps_data[i*nat+j]*((12.*pow(d_data[i*nat+j],12.0)/pow(dij,7.0))*u[1]-(12.*pow(d_data[i*nat+j],6.0)/pow(dij,4.0))*u[1]);
			frc_data[2] += -eps_data[i*nat+j]*((12.*pow(d_data[i*nat+j],12.0)/pow(dij,7.0))*u[2]-(12.*pow(d_data[i*nat+j],6.0)/pow(dij,4.0))*u[2]);
		}
	}
	return hess;
}

//
// returns a [Nrad X Nang] x [npts] array which contains rasterized versions of the
// non-orthogonal basis functions.
//
static PyObject* Overlap_SH(PyObject *self, PyObject  *args)
{
	PyObject* Prm_;
	if (!PyArg_ParseTuple(args, "O!", &PyDict_Type, &Prm_))
	return NULL;

	SHParams Prmo = ParseParams(Prm_);SHParams* Prm=&Prmo;
	int Nbas = Prm->SH_NRAD*(1+Prm->SH_LMAX)*(1+Prm->SH_LMAX);
	npy_intp outdim[2] = {Nbas,Nbas};
	PyObject* SH = PyArray_ZEROS(2, outdim, NPY_DOUBLE, 0);
	double *SH_data;
	SH_data = (double*) ((PyArrayObject*)SH)->data;
	int Nang = (1+Prm->SH_LMAX)*(1+Prm->SH_LMAX);
	for (int i=0; i<Prm->SH_NRAD ; ++i)
	{
		for (int j=i; j<Prm->SH_NRAD ; ++j)
		{
			double S = GOverlap(Prm->RBFS[i*2],Prm->RBFS[j*2],Prm->RBFS[i*2+1],Prm->RBFS[j*2+1]);
			for (int l=0; l<Nang ; ++l)
			{
				int r = i*Nang + l;
				int c = j*Nang + l;
				SH_data[r*Nbas+c] = S;
				SH_data[c*Nbas+r] = S;
			}
		}
	}
	return SH;
}

static PyObject* Overlap_RBF(PyObject *self, PyObject  *args)
{
	PyObject* Prm_;
	if (!PyArg_ParseTuple(args, "O!", &PyDict_Type, &Prm_))
	return NULL;

	SHParams Prmo = ParseParams(Prm_);SHParams* Prm=&Prmo;
	int nbas = Prm->SH_NRAD;
	npy_intp outdim[2] = {nbas,nbas};
	PyObject* SRBF = PyArray_ZEROS(2, outdim, NPY_DOUBLE, 0);
	double *SRBF_data;
	SRBF_data = (double*) ((PyArrayObject*)SRBF)->data;
	for (int i=0; i<Prm->SH_NRAD ; ++i)
	{
		for (int j=0; j<Prm->SH_NRAD ; ++j)
		{
			SRBF_data[i*Prm->SH_NRAD+j] = GOverlap(Prm->RBFS[i*2],Prm->RBFS[j*2],Prm->RBFS[i*2+1],Prm->RBFS[j*2+1]);
		}
	}
	return SRBF;
}

static PyObject* Overlap_RBFS(PyObject *self, PyObject  *args)
{
	PyObject *Prm_, *RBF_obj;
	if (!PyArg_ParseTuple(args, "O!O!", &PyDict_Type, &Prm_, &PyArray_Type, &RBF_obj))
	return NULL;

	PyObject *RBF_array = PyArray_FROM_OTF(RBF_obj, NPY_DOUBLE, NPY_IN_ARRAY);

	if (RBF_array == NULL) {
		Py_XDECREF(RBF_array);
		return NULL;
	}

	int Nele = (int)PyArray_DIM(RBF_array, 0);
	int BasMax = (int)PyArray_DIM(RBF_array, 1);
	int Bassz = (int)PyArray_DIM(RBF_array, 2);

	// std::cout << "Nele: " << Nele << " BasMax: " << BasMax << " L: " << L << std::endl;

	double *RBF = (double*)PyArray_DATA(RBF_array);
	SHParams Prmo = ParseParams(Prm_);SHParams* Prm=&Prmo;
	int nbas = Prm->SH_NRAD;
	npy_intp outdim[3] = {Nele,nbas,nbas};
	PyObject* SRBF = PyArray_ZEROS(3, outdim, NPY_DOUBLE, 0);
	double *SRBF_data;
	SRBF_data = (double*) ((PyArrayObject*)SRBF)->data;
	// for (int i=0; i<100; ++i)
	// 	std::cout << RBF[i] << std::endl;
	for (int k=0; k<Nele ; ++k)
	{
		for (int i=0; i<Prm->SH_NRAD ; ++i)
		{
			for (int j=0; j<Prm->SH_NRAD ; ++j)
			{
				// std::cout << "k: " << k << " i: " << i << " RBFS_i: " << RBFS_data[k*12*2+i*2] << " " << RBFS_data[k*12*2+i*2+1] << " RBFS_j: " << RBFS_data[k*12*2+j*2] << " " << RBFS_data[k*12*2+j*2+1] << std::endl;
				SRBF_data[k*nbas*nbas+i*nbas+j] = GOverlap(RBF[k*BasMax*Bassz+i*2],RBF[k*BasMax*Bassz+j*2],RBF[k*BasMax*Bassz+i*2+1],RBF[k*BasMax*Bassz+j*2+1]);
			}
		}
	}
	Py_DECREF(RBF_array);
	return SRBF;
}

static PyObject*  Make_CM_vary_coords (PyObject *self, PyObject  *args)
{
	//     printtest();
	PyArrayObject *xyz, *grids,  *elements, *atoms_; // the grids now are all the possible points where varyatom can be. and make the CM of theatom for each situation
	double   dist_cut, dist;
	int theatom, varyatom,ngrids;
	if (!PyArg_ParseTuple(args, "O!O!O!O!diii", &PyArray_Type, &xyz, &PyArray_Type, &grids, &PyArray_Type, &atoms_, &PyArray_Type, &elements, &dist_cut, &ngrids, &varyatom, &theatom))
	return NULL;
	PyObject* CM_all = PyList_New(0);

	const int nele = (elements->dimensions)[0];
	uint8_t* ele=(uint8_t*)elements->data;
	uint8_t* atoms=(uint8_t*)atoms_->data;

	npy_intp  CMdim[2] = {nele, ngrids};
	PyObject* CM;
	double *CM_data, *xyz_data, *grids_data;
	double center[3]; // x y z of the center
	int natom, num_CM;

	array<std::vector<int>, 100> ele_index;  // hold max 100 elements most
	array<std::vector<double>, 100> ele_dist;

	npy_intp* Nxyz = xyz->dimensions;
	natom = Nxyz[0];
	npy_intp* Ngrids = grids->dimensions;
	num_CM = Ngrids[0];
	xyz_data = (double*) ((PyArrayObject*) xyz)->data;
	double *xyz_data_tmp = new double[Nxyz[0]*Nxyz[1]];

	grids_data = (double*) grids -> data;

	for (int j = 0; j < natom; j++) {
		if (j==theatom)
		continue;
		for (int k=0; k < nele; k++) {
			if (atoms[j] == ele[k])
			ele_index[k].push_back(j);
		}
	}

	for (int i = 0; i < num_CM;  i++) {
		for (int j = 0 ; j < Nxyz[0]*Nxyz[1]; j++)
		xyz_data_tmp[j] = xyz_data[j];
		xyz_data_tmp[varyatom*Nxyz[1]+0] = grids_data[i*Ngrids[1]+0];
		xyz_data_tmp[varyatom*Nxyz[1]+1] = grids_data[i*Ngrids[1]+1];
		xyz_data_tmp[varyatom*Nxyz[1]+2] = grids_data[i*Ngrids[1]+2];

		center[0] = xyz_data_tmp[theatom*Nxyz[1]+0];
		center[1] = xyz_data_tmp[theatom*Nxyz[1]+1];
		center[2] = xyz_data_tmp[theatom*Nxyz[1]+2];

		CM = PyArray_SimpleNew(2, CMdim, NPY_DOUBLE);
		CM_data = (double*) ((PyArrayObject*) CM)->data;
		for (int n = 0 ; n < CMdim[0]*CMdim[1]; n++)
		CM_data[n] = 0.0;

		for (int m = 0; m < nele; m++) {
			for (int k = 0; k < ele_index[m].size(); k++) {
				dist=sqrt(pow(xyz_data_tmp[ele_index[m][k]*Nxyz[1]+0]-center[0], 2.0) + pow(xyz_data_tmp[ele_index[m][k]*Nxyz[1]+1]-center[1], 2.0) + pow(xyz_data_tmp[ele_index[m][k]*Nxyz[1]+2]-center[2], 2.0));
				//           std::cout<<disp<<"  "<<dist<<std::endl;
				//     std::cout<<" "<<m<<"  "<<k<<"  "<<1/dist*(1 - erf(4*(dist-dist_cut)))/2<<"  "<<1/dist<<std::endl;
				if (dist > 0.5)
				ele_dist[m].push_back(1/dist*(1 - erf(4*(dist-dist_cut)))/2);    // add a smooth cut erf function with 1/x
				else
				ele_dist[m].push_back(-4*dist*dist+3);  // when dist< 0.5, replace 1/x with -4x^2+3 to ensure converge
			}
		}

		for (int m = 0; m < nele; m++) {
			std::sort(ele_dist[m].begin(), ele_dist[m].end());
			std::reverse(ele_dist[m].begin(), ele_dist[m].end());
		}

		for (int m = 0;  m < nele; m++)
		for (int n= 0; n < ele_dist[m].size() && n < ngrids; n++)
		CM_data[m*CMdim[1]+n] = ele_dist[m][n];

		for (int m = 0; m < nele; m++)
		ele_dist[m].clear();

		PyList_Append(CM_all, CM);

	}


	for (int j = 0; j < nele; j++) {
		ele_index[j].clear();
	}
	delete[] xyz_data_tmp;
	PyObject* nlist = PyList_New(0);
	PyList_Append(nlist, CM_all);
	return  nlist;
}


static PyObject*  Make_PGaussian (PyObject *self, PyObject  *args) {

	PyArrayObject   *xyz, *grids, *atoms_, *elements;
	PyObject    *eta_py;
	double   dist_cut;
	int theatom;
	if (!PyArg_ParseTuple(args, "O!O!O!O!idO!",
	&PyArray_Type, &xyz, &PyArray_Type, &grids, &PyArray_Type, &atoms_, &PyArray_Type, &elements, &theatom, &dist_cut,   &PyList_Type, &eta_py))  return NULL;
	PyObject* PGaussian_all = PyList_New(0);
	int dim_eta = PyList_Size(eta_py);
	double  eta[dim_eta];
	const int nele = (elements->dimensions)[0];
	npy_intp  gdim[2] = { 3, dim_eta };
	PyObject* g;
	double *g_data, *xyz_data, *grids_data;
	uint8_t* ele=(uint8_t*)elements->data;
	uint8_t* atoms=(uint8_t*)atoms_->data;
	double center[3]; // x y z of the center
	int natom, num_PGaussian;
	std::array<std::vector<int>, 100> ele_index;  // hold max 100 elements most
	std::array<std::vector<double>, 100> ele_dist;  //  hold max 100 elements most


	npy_intp* Nxyz = xyz->dimensions;
	natom = Nxyz[0];
	npy_intp* Ngrids = grids->dimensions;
	num_PGaussian = Ngrids[0];
	xyz_data = (double*) xyz->data;
	grids_data = (double*) grids -> data;


	for (int i = 0; i < dim_eta; i++)
	eta[i] = PyFloat_AsDouble(PyList_GetItem(eta_py, i));

	for (int j = 0; j < natom; j++) {
		if (j==theatom)
		continue;
		for (int k=0; k < nele; k++) {
			if (atoms[j] == ele[k])
			ele_index[k].push_back(j);
		}
	}

	for  (int i= 0; i < num_PGaussian; i++) {
		center[0] = grids_data[i*Ngrids[1]+0];
		center[1] = grids_data[i*Ngrids[1]+1];
		center[2] = grids_data[i*Ngrids[1]+2];
		for (int m =0; m < nele; m++) {
			g = PyArray_SimpleNew(2, gdim, NPY_DOUBLE);
			g_data = (double*) ((PyArrayObject*) g)->data;
			for (int k = 0; k < gdim[0]*gdim[1]; k++)
			g_data[k] = 0.0;
			if (ele_index[m].size() > 0 ) {
				PGaussian(g_data,  eta,   dim_eta,  ele_index,  m, center, xyz_data,  dist_cut);
			}
			PyList_Append(PGaussian_all, g);
		}
	}

	for (int j = 0; j < nele; j++)
	ele_index[j].clear();

	PyObject* nlist = PyList_New(0);
	PyList_Append(nlist, PGaussian_all);
	//         PyList_Append(nlist, AM_all);

	return  nlist;
}

static PyObject*  Make_ANI1_Sym_deri (PyObject *self, PyObject  *args)
{
	PyArrayObject   *xyz, *atoms_, *elements;
	PyObject    *radius_Rs_py, *angle_Rs_py, *angle_As_py, *Prm_;
	double   radius_Rc, angle_Rc, eta, zeta;
	int theatom;
	if (!PyArg_ParseTuple(args, "O!O!O!O!i", &PyDict_Type, &Prm_, &PyArray_Type, &xyz,  &PyArray_Type, &atoms_, &PyArray_Type, &elements, &theatom))  return NULL;
	SymParams Prmo = ParseSymParams(Prm_);
	SymParams* Prm=&Prmo;

	radius_Rc = Prm->r_Rc;
	angle_Rc = Prm->a_Rc;
	eta = Prm->eta;
	zeta = Prm->zeta;
	double* radius_Rs = Prm->r_Rs;
	double* angle_Rs = Prm->a_Rs;
	double* angle_As = Prm->a_As;
	int dim_radius_Rs = Prm->num_r_Rs;
	int dim_angle_Rs = Prm->num_a_Rs;
	int dim_angle_As = Prm->num_a_As;

	const int nele = (elements->dimensions)[0];
	double  *xyz_data;
	double  *ANI1_Sym_deri_data;
	xyz_data = (double*) xyz->data;
	npy_intp* Nxyz = xyz->dimensions;
	const int natom = Nxyz[0];
	uint8_t* ele=(uint8_t*)elements->data;
	uint8_t* atoms=(uint8_t*)atoms_->data;

	int SYMdim = nele*dim_radius_Rs + nele*(nele+1)/2*dim_angle_Rs*dim_angle_As;
	npy_intp outdim[3] = {1, SYMdim, 3*natom};
	if (theatom<0)
		outdim[0] = natom;
	PyObject* ANI1_Sym_deri = PyArray_ZEROS(3, outdim, NPY_DOUBLE, 0);
	ANI1_Sym_deri_data = (double*) ((PyArrayObject*)ANI1_Sym_deri)->data;

	int data_pointer = 0;
	if (theatom < 0) {
		//#pragma omp parallel for
		for (int i=0; i < natom;  i++) {
			array<std::vector<int>, 10> ele_index;  // hold max 10 elements most
			for (int j = 0; j < natom; j++) {
				if (j==i)
				continue;
				for (int k=0; k < nele; k++) {
					if (atoms[j] == ele[k])
					ele_index[k].push_back(j);
				}
			}
			ANI1_SymFunction_deri(ANI1_Sym_deri_data, i*SYMdim*3*natom, xyz_data, atoms, natom,  ele, nele, i, ele_index, radius_Rc, angle_Rc, radius_Rs, dim_radius_Rs, angle_Rs, dim_angle_Rs, angle_As, dim_angle_As, eta, zeta);
			//data_pointer += SYMdim;
		}
	}
	else {
		array<std::vector<int>, 10> ele_index;  // hold max 10 elements most
		for (int j = 0; j < natom; j++) {
			if (j==theatom)
			continue;
			for (int k=0; k < nele; k++) {
				if (atoms[j] == ele[k])
				ele_index[k].push_back(j);
			}
		}
		ANI1_SymFunction_deri(ANI1_Sym_deri_data, 0, xyz_data, atoms, natom,  ele, nele, theatom, ele_index, radius_Rc, angle_Rc, radius_Rs, dim_radius_Rs, angle_Rs, dim_angle_Rs, angle_As, dim_angle_As, eta, zeta);
	}
	return ANI1_Sym_deri;
}

static PyObject*  Make_ANI1_Sym (PyObject *self, PyObject  *args)
{
	PyArrayObject   *xyz, *atoms_, *elements;
	PyObject    *radius_Rs_py, *angle_Rs_py, *angle_As_py;
	PyObject *Prm_;
	double   radius_Rc, angle_Rc, eta, zeta;
	int theatom;

	if (!PyArg_ParseTuple(args, "O!O!O!O!i", &PyDict_Type, &Prm_, &PyArray_Type, &xyz,  &PyArray_Type, &atoms_, &PyArray_Type, &elements, &theatom))  return NULL;
	SymParams Prmo = ParseSymParams(Prm_);
	SymParams* Prm=&Prmo;
	// Kun: this is why it's good to keep the same names.
	// You could have find-replaced to get all this stuff concise, instead of this.
	//Prm->Print();
	radius_Rc = Prm->r_Rc;
	angle_Rc = Prm->a_Rc;
	eta = Prm->eta;
	zeta = Prm->zeta;
	double* radius_Rs = Prm->r_Rs;
	double* angle_Rs = Prm->a_Rs;
	double* angle_As = Prm->a_As;
	int dim_radius_Rs = Prm->num_r_Rs;
	int dim_angle_Rs = Prm->num_a_Rs;
	int dim_angle_As = Prm->num_a_As;

	const int nele = (elements->dimensions)[0];
	if (nele < 1)
	{
		cout << "AN1 called without elements.... " << endl;
		throw;
	}
	double  *xyz_data, *ANI1_Sym_data;
	xyz_data = (double*) xyz->data;
	npy_intp* Nxyz = xyz->dimensions;
	const int natom = Nxyz[0];
	uint8_t* ele=(uint8_t*)elements->data;
	uint8_t* atoms=(uint8_t*)atoms_->data;

	int SYMdim = nele*dim_radius_Rs + nele*(nele+1)/2*dim_angle_Rs*dim_angle_As;
	npy_intp outdim[2] = {1, SYMdim};
	if (theatom<0)
	outdim[0] = natom;
	PyObject* ANI1_Sym = PyArray_ZEROS(2, outdim, NPY_DOUBLE, 0);
	ANI1_Sym_data = (double*) ((PyArrayObject*)ANI1_Sym)->data;

	int data_pointer = 0;
	if (theatom < 0) {
		#pragma omp parallel for
		for (int i=0; i < natom;  i++) {
			array<std::vector<int>, 10> ele_index;  // hold max 10 elements most
			for (int j = 0; j < natom; j++) {
				if (j==i)
				continue;
				for (int k=0; k < nele; k++) {
					if (atoms[j] == ele[k])
					ele_index[k].push_back(j);
				}
			}
			ANI1_SymFunction(ANI1_Sym_data, i*SYMdim, xyz_data, atoms, natom,  ele, nele, i, ele_index, radius_Rc, angle_Rc, radius_Rs, dim_radius_Rs, angle_Rs, dim_angle_Rs, angle_As, dim_angle_As, eta, zeta);
			//data_pointer += SYMdim;
		}
	}
	else {
		array<std::vector<int>, 10> ele_index;  // hold max 10 elements most
		for (int j = 0; j < natom; j++) {
			if (j==theatom)
			continue;
			for (int k=0; k < nele; k++) {
				if (atoms[j] == ele[k])
				ele_index[k].push_back(j);
			}
		}
		ANI1_SymFunction(ANI1_Sym_data, 0, xyz_data, atoms, natom,  ele, nele, theatom, ele_index, radius_Rc, angle_Rc, radius_Rs, dim_radius_Rs, angle_Rs, dim_angle_Rs, angle_As, dim_angle_As, eta, zeta);
	}
	return ANI1_Sym;
}

static PyObject*  Make_Sym_Update (PyObject *self, PyObject  *args) {
	PyArrayObject   *xyz, *atoms_, *elements, *g1_param, *g2_param;
	double   Rc;
	int theatom;
	if (!PyArg_ParseTuple(args, "O!O!O!dO!O!i",
	&PyArray_Type, &xyz,  &PyArray_Type, &atoms_, &PyArray_Type, &elements, &Rc, &PyArray_Type, &g1_param, &PyArray_Type, &g2_param, &theatom))  return NULL;
	const int g1_dim = (g1_param -> dimensions)[0];
	const int g2_dim = (g2_param -> dimensions)[0];
	const int nele = (elements->dimensions)[0];
	double  *xyz_data, *Sym_data, *g1_param_data, *g2_param_data;
	xyz_data = (double*) xyz->data;
	npy_intp* Nxyz = xyz->dimensions;
	g1_param_data = (double*) g1_param -> data;
	g2_param_data = (double*) g2_param -> data;
	const int natom = Nxyz[0];
	uint8_t* ele=(uint8_t*)elements->data;
	uint8_t* atoms=(uint8_t*)atoms_->data;

	int SYMdim = nele*g1_dim + nele*(nele+1)*g2_dim;
	npy_intp outdim[2] = {1, SYMdim};
	if (theatom<0)
	outdim[0] = natom;
	PyObject* Sym = PyArray_ZEROS(2, outdim, NPY_DOUBLE, 0);
	Sym_data = (double*) ((PyArrayObject*)Sym)->data;

	if (theatom < 0) {
		#pragma omp parallel for
		for (int i=0; i < natom;  i++) {
			array<std::vector<int>, 10> ele_index;  // hold max 10 elements most
			for (int j = 0; j < natom; j++) {
				if (j==i)
				continue;
				for (int k=0; k < nele; k++) {
					if (atoms[j] == ele[k])
					ele_index[k].push_back(j);
				}
			}
			SymFunction(Sym_data, i*SYMdim, xyz_data, atoms, natom,  ele, nele, i, ele_index, Rc, g1_param_data, g1_dim, g2_param_data, g2_dim);
			//data_pointer += SYMdim;
		}


	}
	else {
		array<std::vector<int>, 10> ele_index;  // hold max 10 elements most
		for (int j = 0; j < natom; j++) {
			if (j==theatom)
			continue;
			for (int k=0; k < nele; k++) {
				if (atoms[j] == ele[k])
				ele_index[k].push_back(j);
			}
		}
		SymFunction(Sym_data, 0 , xyz_data, atoms, natom,  ele, nele, theatom, ele_index, Rc, g1_param_data, g1_dim, g2_param_data, g2_dim);
	}

	return Sym;

}

static PyObject*  Make_Sym(PyObject *self, PyObject  *args) {

	PyArrayObject   *xyz, *grids, *atoms_, *elements;
	PyObject    *zeta_py, *eta1_py, *eta2_py, *Rs_py;
	double   dist_cut;
	int theatom;
	if (!PyArg_ParseTuple(args, "O!O!O!O!idO!O!O!O!",
	&PyArray_Type, &xyz, &PyArray_Type, &grids, &PyArray_Type, &atoms_, &PyArray_Type, &elements, &theatom, &dist_cut,   &PyList_Type, &zeta_py,  &PyList_Type, &eta1_py,  &PyList_Type, &eta2_py,  &PyList_Type, &Rs_py))  return NULL;
	PyObject* SYM_all = PyList_New(0);
	int dim_zeta = PyList_Size(zeta_py);
	int dim_eta1 = PyList_Size(eta1_py);
	int dim_eta2 = PyList_Size(eta2_py);
	int dim_Rs = PyList_Size(Rs_py);
	double  zeta[dim_zeta], eta1[dim_eta1], Rs[dim_Rs], eta2[dim_eta2];
	const int nele = (elements->dimensions)[0];
	int SYMdim = nele+nele*(nele+1);
	npy_intp  g1dim[2] = {dim_Rs, dim_eta1};
	npy_intp  g2dim[2] = {dim_zeta, dim_eta2};
	PyObject* g1;
	PyObject* g2;
	double *g1_data, *g2_data, *xyz_data, *grids_data;
	uint8_t* ele=(uint8_t*)elements->data;
	uint8_t* atoms=(uint8_t*)atoms_->data;
	double center[3]; // x y z of the center
	int natom, lambda, num_SYM;
	array<std::vector<int>, 100> ele_index;  // hold max 100 elements most
	array<std::vector<double>, 100> ele_dist;  //  hold max 100 elements most

	npy_intp* Nxyz = xyz->dimensions;
	natom = Nxyz[0];
	npy_intp* Ngrids = grids->dimensions;
	num_SYM = Ngrids[0];
	xyz_data = (double*) xyz->data;
	grids_data = (double*) grids -> data;

	//for (int i=0; i < natom; i++)
	//    std::cout<<"atoms[i]:"<<static_cast<int16_t>(atoms[i])<<std::endl;   // tricky way to print uint8, uint8 can not printed by cout directly.

	for (int i = 0; i < dim_zeta; i++) {
		zeta[i] = PyFloat_AsDouble(PyList_GetItem(zeta_py, i));
	}
	for (int i = 0; i < dim_eta1; i++)
	eta1[i] = PyFloat_AsDouble(PyList_GetItem(eta1_py, i));
	for (int i = 0; i < dim_eta2; i++)
	eta2[i] = PyFloat_AsDouble(PyList_GetItem(eta2_py, i));
	for (int i = 0; i < dim_Rs; i++)
	Rs[i] = PyFloat_AsDouble(PyList_GetItem(Rs_py, i));

	//for (int j = 0; j < 10; j++)
	//	std::cout<<xyz_data[j]<<std::endl;

	for (int j = 0; j < natom; j++) {
		if (j==theatom)
		continue;
		for (int k=0; k < nele; k++) {
			if (atoms[j] == ele[k])
			ele_index[k].push_back(j);
		}
	}

	for  (int i= 0; i < num_SYM; i++) {
		center[0] = grids_data[i*Ngrids[1]+0];
		center[1] = grids_data[i*Ngrids[1]+1];
		center[2] = grids_data[i*Ngrids[1]+2];
		for (int m =0; m < nele; m++) {
			g1 = PyArray_SimpleNew(2, g1dim, NPY_DOUBLE);
			g1_data = (double*) ((PyArrayObject*) g1)->data;
			for (int k = 0; k < g1dim[0]*g1dim[1]; k++)
			g1_data[k] = 0.0;
			if (ele_index[m].size() > 0 ) {
				G1(g1_data,  Rs, eta1,  dim_Rs,  dim_eta1,  ele_index,  m, center, xyz_data,  dist_cut);
			}
			PyList_Append(SYM_all, g1);
		}
		for (int m = 0; m < nele; m++)
		for (int n =m; n < nele; n++) {
			g2 = PyArray_SimpleNew(2, g2dim, NPY_DOUBLE);
			g2_data = (double*) ((PyArrayObject*) g2)->data;
			for (int k = 0; k < g2dim[0]*g2dim[1]; k++)
			g2_data[k] = 0.0;
			lambda = 1;
			if (ele_index[m].size() > 0 && ele_index[n].size() > 0)
			G2(g2_data, zeta, eta2,  dim_zeta,  dim_eta2,  lambda,  ele_index, m, n, center, xyz_data,  dist_cut);
			PyList_Append(SYM_all, g2);
			g2 = PyArray_SimpleNew(2, g2dim, NPY_DOUBLE);
			g2_data = (double*) ((PyArrayObject*) g2)->data;
			for (int k = 0; k < g2dim[0]*g2dim[1]; k++)
			g2_data[k] = 0.0;
			lambda = -1;
			if (ele_index[m].size() > 0 && ele_index[n].size() > 0)
			G2(g2_data, zeta, eta2,  dim_zeta,  dim_eta2,  lambda,  ele_index, m, n, center, xyz_data,  dist_cut);
			PyList_Append(SYM_all, g2);
		}
	}
	for (int j = 0; j < nele; j++)
	ele_index[j].clear();
	PyObject* nlist = PyList_New(0);
	PyList_Append(nlist, SYM_all);
	//         PyList_Append(nlist, AM_all);
	return  nlist;
}

static PyMethodDef EmbMethods[] =
{
	{"Make_NListNaive", Make_NListNaive, METH_VARARGS,
	"Make_NListNaive method"},
	{"DipoleAutoCorr", DipoleAutoCorr, METH_VARARGS,
	"DipoleAutoCorr method"},
	{"Make_DistMat", Make_DistMat, METH_VARARGS,
	"Make_DistMat method"},
	{"Norm_Matrices", Norm_Matrices, METH_VARARGS,
	"Norm_Matrices method"},
	{"Make_CM", Make_CM, METH_VARARGS,
	"Make_CM method"},
	{"Make_RDF", Make_RDF, METH_VARARGS,
	"Make_RDF method"},
	{"Make_Go", Make_Go, METH_VARARGS,
	"Make_Go method"},
	{"Make_GoForce", Make_GoForce, METH_VARARGS,
	"Make_GoForce method"},
	{"Make_GoForceLocal", Make_GoForceLocal, METH_VARARGS,
	"Make_GoForceLocal method"},
	{"Make_GoHess", Make_GoHess, METH_VARARGS,
	"Make_GoHess method"},
	{"Make_LJForce", Make_LJForce, METH_VARARGS,
	"Make_LJForce method"},
	{"Make_SH", Make_SH, METH_VARARGS,
	"Make_SH method"},
	{"Make_SH_Transf", Make_SH_Transf, METH_VARARGS,
	"Make_SH_Transf method"},
	{"Make_Inv", Make_Inv, METH_VARARGS,
	"Make_Inv method"},
	{"Raster_SH", Raster_SH, METH_VARARGS,
	"Raster_SH method"},
	{"Overlap_SH", Overlap_SH, METH_VARARGS,
	"Overlap_SH method"},
	{"Overlap_RBF", Overlap_RBF, METH_VARARGS,
	"Overlap_RBF method"},
	{"Overlap_RBFS", Overlap_RBFS, METH_VARARGS,
	"Overlap_RBFS method"},
	{"Project_SH", Project_SH, METH_VARARGS,
	"Project_SH method"},
	{"Make_PGaussian", Make_PGaussian, METH_VARARGS,
	"Make_PGaussian method"},
	{"Make_Sym", Make_Sym, METH_VARARGS,
	"Make_Sym method"},
	{"Make_Sym_Update", Make_Sym_Update, METH_VARARGS,
	"Make_Sym_Update method"},
	{"Make_ANI1_Sym", Make_ANI1_Sym, METH_VARARGS,
	"Make_ANI1_Sym method"},
	{"Make_ANI1_Sym_deri", Make_ANI1_Sym_deri, METH_VARARGS,
	"Make_ANI1_Sym_deri method"},
	{"Make_CM_vary_coords", Make_CM_vary_coords, METH_VARARGS,
	"Make_CM_vary_coords method"},
	{NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
initMolEmb(void)
{
	(void) Py_InitModule("MolEmb", EmbMethods);
	/* IMPORTANT: this must be called */
	import_array();
}
