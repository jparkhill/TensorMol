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
	PyObject* RBFo = PyDict_GetItemString(Pdict, "RBFS");
	PyArrayObject* RBFa = (PyArrayObject*) RBFo;
	double* RBFd = (double*)RBFa->data;
	tore.SH_NRAD = (RBFa->dimensions)[0];
	tore.SH_LMAX = PyInt_AS_LONG((PyDict_GetItemString(Pdict,"SH_LMAX")));
	tore.SH_NRAD = PyInt_AS_LONG((PyDict_GetItemString(Pdict,"SH_NRAD")));
	//cout << "tore.SH_LMAX: " << tore.SH_LMAX << endl;
	/for (int i=0; i<tore.SH_NRAD; ++i)
	{
		  //      cout << RBFd[i*2] << RBFd[i*2+1] << endl;
			tore.RBFS[i][0] = RBFd[i*2];
			tore.RBFS[i][1] = RBFd[i*2+1];
			/*
			for (int j=i; j<tore.SH_NRAD; ++j)
			{
				tore.SRBF[i][j] = GOverlap(RBFd[i*2],RBFd[j*2],RBFd[i*2+1],RBFd[j*2+1]);
				tore.SRBF[j][i] = tore.SRBF[i][j];
			}
			*/
	}
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
		if (dist < dist_cut)
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
					outmat[s] += 0.125*pow(distmat[i*natom+j]-dist(xyz_data[3*s],xyz_data[3*s+1],xyz_data[3*s+2],coords[3*j],coords[3*j+1],coords[3*j+2]),2.0);
				}
				else if (j==theatom)
				{
					outmat[s] += 0.125*pow(distmat[i*natom+j]-dist(xyz_data[3*s],xyz_data[3*s+1],xyz_data[3*s+2],coords[3*i],coords[3*i+1],coords[3*i+2]),2.0);
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
		center[0] = grids_data[i*Ngrids[1]+0];
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
		center[0] = grids_data[i*Ngrids[1]+0];
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
				dist=sqrt(pow(xyz_data[ele_index_mask[m][k]*Nxyz[1]+0]-center[0], 2.0) + pow(xyz_data[ele_index_mask[m][k]*Nxyz[1]+1]-center[1], 2.0) + pow(xyz_data[ele_index_mask[m][k]*Nxyz[1]+2]-center[2], 2.0));
				//           std::cout<<disp<<"  "<<dist<<std::endl;
				//     std::cout<<" "<<m<<"  "<<k<<"  "<<1/dist*(1 - erf(4*(dist-dist_cut)))/2<<"  "<<1/dist<<std::endl;
				//if (dist > 0.5)
				//ele_dist[m].push_back(1/dist*(1 - erf(4*(dist-dist_cut)))/2);    // add a smooth cut erf function with 1/x
				//else
				//ele_dist[m].push_back(-4*dist*dist+3);  // when dist< 0.5, replace 1/x with -4x^2+3 to ensure converge
				ele_dist[m].push_back(1/dist);
			}
		}

		//for (int m = 0; m < nele; m++) {
		//	std::sort(ele_dist[m].begin(), ele_dist[m].end());
		//	std::reverse(ele_dist[m].begin(), ele_dist[m].end());
		//}

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
	double   dist_cut,  mask, mask_prob,dist;
	int  ngrids,  theatom;
	PyObject *Prm_;
	if (!PyArg_ParseTuple(args, "O!O!O!O!diid",
	&PyDict_Type, &Prm_, &PyArray_Type, &xyz, &PyArray_Type, &grids, &PyArray_Type, &atoms_ , &dist_cut, &ngrids, &theatom, &mask))
	return NULL;

	SHParams Prmo = ParseParams(Prm_);SHParams* Prm=&Prmo;
	const int nele = (elements->dimensions)[0];
	uint8_t* ele=(uint8_t*)elements->data;
	uint8_t* atoms=(uint8_t*)atoms_->data;
	npy_intp* Nxyz = xyz->dimensions;
	int natom, num_CM;
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

	if (theatom<0)
	{
		#pragma omp parallel for
		for (int i=0; i<natom; ++i)
		{
			double xc = xyz_data[i*Nxyz[1]+0];
			double yc = xyz_data[i*Nxyz[1]+1];
			double zc = xyz_data[i*Nxyz[1]+2];
			for (int j = 0; j < natom; j++)
			{
				double x = xyz_data[j*Nxyz[1]+0];
				double y = xyz_data[j*Nxyz[1]+1];
				double z = xyz_data[j*Nxyz[1]+2];
				//RadSHProjection(x-xc,y-yc,z-zc,SH_data + i*SH_NRAD*(1+SH_LMAX)*(1+SH_LMAX), natom);
				RadSHProjection_Orth(Prm,x-xc,y-yc,z-zc,SH_data + i*Prm->SH_NRAD*(1+Prm->SH_LMAX)*(1+Prm->SH_LMAX), natom);
			}
		}
	}
	else{
		int i = theatom;
		int ai=0;

		double xc = xyz_data[i*Nxyz[1]+0];
		double yc = xyz_data[i*Nxyz[1]+1];
		double zc = xyz_data[i*Nxyz[1]+2];

		for (int j = 0; j < natom; j++)
		{
			double x = xyz_data[j*Nxyz[1]+0];
			double y = xyz_data[j*Nxyz[1]+1];
			double z = xyz_data[j*Nxyz[1]+2];
			RadSHProjection(Prm,x-xc,y-yc,z-zc,SH_data + ai*Prm->SH_NRAD*(1+Prm->SH_LMAX)*(1+Prm->SH_LMAX), natom);
		}
	}
	//	}
	return SH;
}

//
// Embed molecule using Gaussian X Spherical Harmonic basis
// Does all atoms... just for debug really.
// Make invariant embedding of all the atoms.
// This is not a reversible embedding.
//
static PyObject* Make_Inv(PyObject *self, PyObject  *args)
{
	PyArrayObject *xyz, *grids,  *elements, *atoms_;
	double   dist_cut,  mask, mask_prob,dist;
	int  ngrids,  theatom;
	PyObject* Prm_;
	if (!PyArg_ParseTuple(args, "O!O!O!O!di",
	&PyDict_Type, &Prm_, &PyArray_Type, &xyz, &PyArray_Type, &grids, &PyArray_Type, &atoms_ , &dist_cut, &theatom))
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
	int  ngrids,  theatom;
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
			double Gv = Gau(r, Prm->RBFS[i][0],Prm->RBFS[i][1]);
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

//
// Gives the projection of a delta function at xyz
//
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
		double Gv = Gau(r, Prm->RBFS[i][0],Prm->RBFS[i][1]);
		cout << Prm->RBFS[i][0] << " " << Gv << endl;

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
	int nbas = Prm->SH_NRAD*(1+Prm->SH_LMAX)*(1+Prm->SH_LMAX);
	npy_intp outdim[2] = {nbas,nbas};
	PyObject* SH = PyArray_ZEROS(2, outdim, NPY_DOUBLE, 0);
	double *SH_data;
	SH_data = (double*) ((PyArrayObject*)SH)->data;
	int Nbas = Prm->SH_NRAD*(1+Prm->SH_LMAX)*(1+Prm->SH_LMAX);
	int Nang = (1+Prm->SH_LMAX)*(1+Prm->SH_LMAX);
	for (int i=0; i<Prm->SH_NRAD ; ++i)
	{
		for (int j=i; j<Prm->SH_NRAD ; ++j)
		{
			double S = GOverlap(Prm->RBFS[i][0],Prm->RBFS[j][0],Prm->RBFS[i][1],Prm->RBFS[j][1]);
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

static PyObject*  Make_CM_vary_coords (PyObject *self, PyObject  *args)
{
	//     printtest();
	PyArrayObject *xyz, *grids,  *elements, *atoms_; // the grids now are all the possible points where varyatom can be. and make the CM of theatom for each situation
	double   dist_cut, dist;
	int  ngrids,  theatom, varyatom;
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

static PyObject*  Make_Sym (PyObject *self, PyObject  *args) {

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
	{"Make_Inv", Make_Inv, METH_VARARGS,
	"Make_Inv method"},
	{"Raster_SH", Raster_SH, METH_VARARGS,
	"Raster_SH method"},
	{"Overlap_SH", Overlap_SH, METH_VARARGS,
	"Overlap_SH method"},
	{"Project_SH", Project_SH, METH_VARARGS,
	"Project_SH method"},
	{"Make_PGaussian", Make_PGaussian, METH_VARARGS,
	"Make_PGaussian method"},
	{"Make_Sym", Make_Sym, METH_VARARGS,
	"Make_Sym method"},
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
