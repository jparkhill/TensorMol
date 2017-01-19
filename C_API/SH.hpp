#include <math.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

#ifdef __clang__
#if __clang_major__ >= 7
#include <array>
using namespace std;
#else
#include <omp.h>
#include <tr1/array>
using namespace std::tr1;
#endif
#else
#include <array>
using namespace std;
#include <omp.h>
#endif

using namespace std;
#define PI 3.14159265358979

#define SH_NRAD 10
#define SH_LMAX 6
// r0, sigma
// if you want to sense beyond 15A, you need to fix this grid.
const double RBFS[12][2]={{0.1, 0.156787}, {0.3, 0.3}, {0.5, 0.5}, {0.7, 0.7}, {1.3, 1.3}, {2.2,
	2.4}, {4.4, 2.4}, {6.6, 2.4}, {8.8, 2.4}, {11., 2.4}, {13.2,
		2.4}, {15.4, 2.4}};

		//
		// Real Spherical Harmonics...
		//
		double RealSphericalHarmonic(int l, int m, double theta, double phi)
		{
			if (l > 11)
			{
				cout << "lmax = 11" << endl;
				throw;
			}
			switch (l)
			{
				case 0:
				switch (m){
					case 0:
					return 1/(2.*sqrt(PI));
					default:
					throw; return 0.0;
				}
				case 1:
				switch (m){
					case -1:
					return (sqrt(3/PI)*sin(phi)*sin(theta))/2.;
					case 0:
					return (sqrt(3/PI)*cos(theta))/2.;
					case 1:
					return (sqrt(3/PI)*cos(phi)*sin(theta))/2.;
					default:
					throw; return 0.0;
				}
				case 2:
				switch (m){
					case -2:
					return -(sqrt(15/PI)*sin(2*phi)*pow(sin(theta),2))/4.;
					case -1:
					return (sqrt(15/PI)*sin(phi)*sin(2*theta))/4.;
					case 0:
					return (sqrt(5/PI)*(1 + 3*cos(2*theta)))/8.;
					case 1:
					return (sqrt(15/PI)*cos(phi)*sin(2*theta))/4.;
					case 2:
					return (sqrt(15/PI)*cos(2*phi)*pow(sin(theta),2))/4.;
					default:
					throw; return 0.0;
				}
				case 3:
				switch (m){
					case -3:
					return (sqrt(35/(2.*PI))*sin(3*phi)*pow(sin(theta),3))/4.;
					case -2:
					return -(sqrt(105/PI)*cos(theta)*sin(2*phi)*pow(sin(theta),2))/4.;
					case -1:
					return (sqrt(21/(2.*PI))*(3 + 5*cos(2*theta))*sin(phi)*sin(theta))/8.;
					case 0:
					return (sqrt(7/PI)*cos(theta)*(-1 + 5*cos(2*theta)))/8.;
					case 1:
					return (sqrt(21/(2.*PI))*cos(phi)*(3 + 5*cos(2*theta))*sin(theta))/8.;
					case 2:
					return (sqrt(105/PI)*cos(2*phi)*cos(theta)*pow(sin(theta),2))/4.;
					case 3:
					return (sqrt(35/(2.*PI))*cos(3*phi)*pow(sin(theta),3))/4.;
					default:
					throw; return 0.0;
				}
				case 4:
				switch (m){
					case -4:
					return (-3*sqrt(35/PI)*sin(4*phi)*pow(sin(theta),4))/16.;
					case -3:
					return (3*sqrt(35/(2.*PI))*cos(theta)*sin(3*phi)*pow(sin(theta),3))/4.;
					case -2:
					return (-3*sqrt(5/PI)*(5 + 7*cos(2*theta))*sin(2*phi)*pow(sin(theta),2))/16.;
					case -1:
					return (3*sqrt(5/(2.*PI))*(1 + 7*cos(2*theta))*sin(phi)*sin(2*theta))/16.;
					case 0:
					return (3*(9 + 20*cos(2*theta) + 35*cos(4*theta)))/(128.*sqrt(PI));
					case 1:
					return (3*sqrt(5/(2.*PI))*cos(phi)*(1 + 7*cos(2*theta))*sin(2*theta))/16.;
					case 2:
					return (3*sqrt(5/PI)*cos(2*phi)*(5 + 7*cos(2*theta))*pow(sin(theta),2))/16.;
					case 3:
					return (3*sqrt(35/(2.*PI))*cos(3*phi)*cos(theta)*pow(sin(theta),3))/4.;
					case 4:
					return (3*sqrt(35/PI)*cos(4*phi)*pow(sin(theta),4))/16.;
					default:
					throw; return 0.0;
				}
				case 5:
				switch (m){
					case -5:
					return (3*sqrt(77/(2.*PI))*sin(5*phi)*pow(sin(theta),5))/16.;
					case -4:
					return (-3*sqrt(385/PI)*cos(theta)*sin(4*phi)*pow(sin(theta),4))/16.;
					case -3:
					return (sqrt(385/(2.*PI))*(7 + 9*cos(2*theta))*sin(3*phi)*pow(sin(theta),3))/32.;
					case -2:
					return -(sqrt(1155/PI)*cos(theta)*(1 + 3*cos(2*theta))*sin(2*phi)*pow(sin(theta),2))/16.;
					case -1:
					return (sqrt(165/PI)*(1 - 14*pow(cos(theta),2) + 21*pow(cos(theta),4))*sin(phi)*sin(theta))/16.;
					case 0:
					return (sqrt(11/PI)*(30*cos(theta) + 35*cos(3*theta) + 63*cos(5*theta)))/256.;
					case 1:
					return (sqrt(165/PI)*cos(phi)*(1 - 14*pow(cos(theta),2) + 21*pow(cos(theta),4))*sin(theta))/16.;
					case 2:
					return (sqrt(1155/PI)*cos(2*phi)*cos(theta)*(1 + 3*cos(2*theta))*pow(sin(theta),2))/16.;
					case 3:
					return (sqrt(385/(2.*PI))*cos(3*phi)*(7 + 9*cos(2*theta))*pow(sin(theta),3))/32.;
					case 4:
					return (3*sqrt(385/PI)*cos(4*phi)*cos(theta)*pow(sin(theta),4))/16.;
					case 5:
					return (3*sqrt(77/(2.*PI))*cos(5*phi)*pow(sin(theta),5))/16.;
					default:
					throw; return 0.0;
				}
				case 6:
				switch (m){
					case -6:
					return -(sqrt(3003/(2.*PI))*sin(6*phi)*pow(sin(theta),6))/32.;
					case -5:
					return (3*sqrt(1001/(2.*PI))*cos(theta)*sin(5*phi)*pow(sin(theta),5))/16.;
					case -4:
					return (-3*sqrt(91/PI)*(9 + 11*cos(2*theta))*sin(4*phi)*pow(sin(theta),4))/64.;
					case -3:
					return (sqrt(1365/(2.*PI))*cos(theta)*(5 + 11*cos(2*theta))*sin(3*phi)*pow(sin(theta),3))/32.;
					case -2:
					return -(sqrt(1365/(2.*PI))*cos(phi)*(1 - 18*pow(cos(theta),2) + 33*pow(cos(theta),4))*sin(phi)*pow(sin(theta),2))/16.;
					case -1:
					return (sqrt(273/PI)*cos(theta)*(5 - 30*pow(cos(theta),2) + 33*pow(cos(theta),4))*sin(phi)*sin(theta))/16.;
					case 0:
					return (sqrt(13/PI)*(-5 + 105*pow(cos(theta),2) - 315*pow(cos(theta),4) + 231*pow(cos(theta),6)))/32.;
					case 1:
					return (sqrt(273/PI)*cos(phi)*cos(theta)*(5 - 30*pow(cos(theta),2) + 33*pow(cos(theta),4))*sin(theta))/16.;
					case 2:
					return (sqrt(1365/(2.*PI))*cos(2*phi)*(1 - 18*pow(cos(theta),2) + 33*pow(cos(theta),4))*pow(sin(theta),2))/32.;
					case 3:
					return (sqrt(1365/(2.*PI))*cos(3*phi)*cos(theta)*(5 + 11*cos(2*theta))*pow(sin(theta),3))/32.;
					case 4:
					return (3*sqrt(91/PI)*cos(4*phi)*(9 + 11*cos(2*theta))*pow(sin(theta),4))/64.;
					case 5:
					return (3*sqrt(1001/(2.*PI))*cos(5*phi)*cos(theta)*pow(sin(theta),5))/16.;
					case 6:
					return (sqrt(3003/(2.*PI))*cos(6*phi)*pow(sin(theta),6))/32.;
					default:
					throw; return 0.0;
				}
				case 7:
				switch (m){
					case -7:
					return (3*sqrt(715/PI)*sin(7*phi)*pow(sin(theta),7))/64.;
					case -6:
					return (-3*sqrt(5005/(2.*PI))*cos(theta)*sin(6*phi)*pow(sin(theta),6))/32.;
					case -5:
					return (3*sqrt(385/PI)*(-1 + 13*pow(cos(theta),2))*sin(5*phi)*pow(sin(theta),5))/64.;
					case -4:
					return (-3*sqrt(385/PI)*cos(theta)*(7 + 13*cos(2*theta))*sin(4*phi)*pow(sin(theta),4))/64.;
					case -3:
					return (3*sqrt(35/PI)*(3 - 66*pow(cos(theta),2) + 143*pow(cos(theta),4))*sin(3*phi)*pow(sin(theta),3))/64.;
					case -2:
					return (-3*sqrt(35/(2.*PI))*cos(phi)*cos(theta)*(15 - 110*pow(cos(theta),2) + 143*pow(cos(theta),4))*sin(phi)*pow(sin(theta),2))/16.;
					case -1:
					return (sqrt(105/PI)*(-5 + 135*pow(cos(theta),2) - 495*pow(cos(theta),4) + 429*pow(cos(theta),6))*sin(phi)*sin(theta))/64.;
					case 0:
					return (sqrt(15/PI)*cos(theta)*(-35 + 315*pow(cos(theta),2) - 693*pow(cos(theta),4) + 429*pow(cos(theta),6)))/32.;
					case 1:
					return (sqrt(105/PI)*cos(phi)*(-5 + 135*pow(cos(theta),2) - 495*pow(cos(theta),4) + 429*pow(cos(theta),6))*sin(theta))/64.;
					case 2:
					return (3*sqrt(35/(2.*PI))*cos(2*phi)*cos(theta)*(15 - 110*pow(cos(theta),2) + 143*pow(cos(theta),4))*pow(sin(theta),2))/32.;
					case 3:
					return (3*sqrt(35/PI)*cos(3*phi)*(3 - 66*pow(cos(theta),2) + 143*pow(cos(theta),4))*pow(sin(theta),3))/64.;
					case 4:
					return (3*sqrt(385/PI)*cos(4*phi)*cos(theta)*(7 + 13*cos(2*theta))*pow(sin(theta),4))/64.;
					case 5:
					return (3*sqrt(385/PI)*cos(5*phi)*(-1 + 13*pow(cos(theta),2))*pow(sin(theta),5))/64.;
					case 6:
					return (3*sqrt(5005/(2.*PI))*cos(6*phi)*cos(theta)*pow(sin(theta),6))/32.;
					case 7:
					return (3*sqrt(715/PI)*cos(7*phi)*pow(sin(theta),7))/64.;
					default:
					throw; return 0.0;
				}
				case 8:
				switch (m){
					case -8:
					return (-3*sqrt(12155/PI)*sin(8*phi)*pow(sin(theta),8))/256.;
					case -7:
					return (3*sqrt(12155/PI)*cos(theta)*sin(7*phi)*pow(sin(theta),7))/64.;
					case -6:
					return -(sqrt(7293/(2.*PI))*(-1 + 15*pow(cos(theta),2))*sin(6*phi)*pow(sin(theta),6))/64.;
					case -5:
					return (3*sqrt(17017/PI)*cos(theta)*(3 + 5*cos(2*theta))*sin(5*phi)*pow(sin(theta),5))/128.;
					case -4:
					return (-3*sqrt(1309/PI)*(1 - 26*pow(cos(theta),2) + 65*pow(cos(theta),4))*sin(4*phi)*pow(sin(theta),4))/128.;
					case -3:
					return (sqrt(19635/PI)*cos(theta)*(3 - 26*pow(cos(theta),2) + 39*pow(cos(theta),4))*sin(3*phi)*pow(sin(theta),3))/64.;
					case -2:
					return (-3*sqrt(595/(2.*PI))*cos(phi)*(-1 + 33*pow(cos(theta),2) - 143*pow(cos(theta),4) + 143*pow(cos(theta),6))*sin(phi)*pow(sin(theta),2))/32.;
					case -1:
					return (3*sqrt(17/PI)*(178 + 869*cos(2*theta) + 286*cos(4*theta) + 715*cos(6*theta))*sin(phi)*sin(2*theta))/4096.;
					case 0:
					return (sqrt(17/PI)*(35 - 1260*pow(cos(theta),2) + 6930*pow(cos(theta),4) - 12012*pow(cos(theta),6) + 6435*pow(cos(theta),8)))/256.;
					case 1:
					return (3*sqrt(17/PI)*cos(phi)*(178 + 869*cos(2*theta) + 286*cos(4*theta) + 715*cos(6*theta))*sin(2*theta))/4096.;
					case 2:
					return (3*sqrt(595/(2.*PI))*cos(2*phi)*(-1 + 33*pow(cos(theta),2) - 143*pow(cos(theta),4) + 143*pow(cos(theta),6))*pow(sin(theta),2))/64.;
					case 3:
					return (sqrt(19635/PI)*cos(3*phi)*cos(theta)*(3 - 26*pow(cos(theta),2) + 39*pow(cos(theta),4))*pow(sin(theta),3))/64.;
					case 4:
					return (3*sqrt(1309/PI)*cos(4*phi)*(1 - 26*pow(cos(theta),2) + 65*pow(cos(theta),4))*pow(sin(theta),4))/128.;
					case 5:
					return (3*sqrt(17017/PI)*cos(5*phi)*cos(theta)*(3 + 5*cos(2*theta))*pow(sin(theta),5))/128.;
					case 6:
					return (sqrt(7293/(2.*PI))*cos(6*phi)*(-1 + 15*pow(cos(theta),2))*pow(sin(theta),6))/64.;
					case 7:
					return (3*sqrt(12155/PI)*cos(7*phi)*cos(theta)*pow(sin(theta),7))/64.;
					case 8:
					return (3*sqrt(12155/PI)*cos(8*phi)*pow(sin(theta),8))/256.;
					default:
					throw; return 0.0;
				}
				case 9:
				switch (m){
					case -9:
					return (sqrt(230945/(2.*PI))*sin(9*phi)*pow(sin(theta),9))/256.;
					case -8:
					return (-3*sqrt(230945/PI)*cos(theta)*sin(8*phi)*pow(sin(theta),8))/256.;
					case -7:
					return (3*sqrt(13585/(2.*PI))*(15 + 17*cos(2*theta))*sin(7*phi)*pow(sin(theta),7))/512.;
					case -6:
					return -(sqrt(40755/(2.*PI))*cos(theta)*(-3 + 17*pow(cos(theta),2))*sin(6*phi)*pow(sin(theta),6))/64.;
					case -5:
					return (3*sqrt(2717/(2.*PI))*(1 - 30*pow(cos(theta),2) + 85*pow(cos(theta),4))*sin(5*phi)*pow(sin(theta),5))/128.;
					case -4:
					return (-3*sqrt(95095/PI)*cos(theta)*(1 - 10*pow(cos(theta),2) + 17*pow(cos(theta),4))*sin(4*phi)*pow(sin(theta),4))/128.;
					case -3:
					return (sqrt(21945/(2.*PI))*(-1 + 39*pow(cos(theta),2) - 195*pow(cos(theta),4) + 221*pow(cos(theta),6))*sin(3*phi)*pow(sin(theta),3))/128.;
					case -2:
					return (-3*sqrt(1045/(2.*PI))*cos(phi)*cos(theta)*(-7 + 91*pow(cos(theta),2) - 273*pow(cos(theta),4) + 221*pow(cos(theta),6))*sin(phi)*pow(sin(theta),2))/32.;
					case -1:
					return (3*sqrt(95/PI)*(7 - 308*pow(cos(theta),2) + 2002*pow(cos(theta),4) - 4004*pow(cos(theta),6) + 2431*pow(cos(theta),8))*sin(phi)*sin(theta))/256.;
					case 0:
					return (sqrt(19/PI)*(4410*cos(theta) + 11*(420*cos(3*theta) + 13*(36*cos(5*theta) + 45*cos(7*theta) + 85*cos(9*theta)))))/65536.;
					case 1:
					return (3*sqrt(95/PI)*cos(phi)*(7 - 308*pow(cos(theta),2) + 2002*pow(cos(theta),4) - 4004*pow(cos(theta),6) + 2431*pow(cos(theta),8))*sin(theta))/256.;
					case 2:
					return (3*sqrt(1045/(2.*PI))*cos(2*phi)*cos(theta)*(-7 + 91*pow(cos(theta),2) - 273*pow(cos(theta),4) + 221*pow(cos(theta),6))*pow(sin(theta),2))/64.;
					case 3:
					return (sqrt(21945/(2.*PI))*cos(3*phi)*(-1 + 39*pow(cos(theta),2) - 195*pow(cos(theta),4) + 221*pow(cos(theta),6))*pow(sin(theta),3))/128.;
					case 4:
					return (3*sqrt(95095/PI)*cos(4*phi)*cos(theta)*(1 - 10*pow(cos(theta),2) + 17*pow(cos(theta),4))*pow(sin(theta),4))/128.;
					case 5:
					return (3*sqrt(2717/(2.*PI))*cos(5*phi)*(1 - 30*pow(cos(theta),2) + 85*pow(cos(theta),4))*pow(sin(theta),5))/128.;
					case 6:
					return (sqrt(40755/(2.*PI))*cos(6*phi)*cos(theta)*(-3 + 17*pow(cos(theta),2))*pow(sin(theta),6))/64.;
					case 7:
					return (3*sqrt(13585/(2.*PI))*cos(7*phi)*(15 + 17*cos(2*theta))*pow(sin(theta),7))/512.;
					case 8:
					return (3*sqrt(230945/PI)*cos(8*phi)*cos(theta)*pow(sin(theta),8))/256.;
					case 9:
					return (sqrt(230945/(2.*PI))*cos(9*phi)*pow(sin(theta),9))/256.;
					default:
					throw; return 0.0;
				}
				case 10:
				switch (m){
					case -10:
					return -(sqrt(969969/(2.*PI))*sin(10*phi)*pow(sin(theta),10))/512.;
					case -9:
					return (sqrt(4849845/(2.*PI))*cos(theta)*sin(9*phi)*pow(sin(theta),9))/256.;
					case -8:
					return -(sqrt(255255/PI)*(-1 + 19*pow(cos(theta),2))*sin(8*phi)*pow(sin(theta),8))/512.;
					case -7:
					return (3*sqrt(85085/(2.*PI))*cos(theta)*(13 + 19*cos(2*theta))*sin(7*phi)*pow(sin(theta),7))/512.;
					case -6:
					return (-3*sqrt(5005/(2.*PI))*(3 - 102*pow(cos(theta),2) + 323*pow(cos(theta),4))*sin(6*phi)*pow(sin(theta),6))/512.;
					case -5:
					return (3*sqrt(1001/(2.*PI))*cos(theta)*(15 - 170*pow(cos(theta),2) + 323*pow(cos(theta),4))*sin(5*phi)*pow(sin(theta),5))/128.;
					case -4:
					return (-3*sqrt(5005/PI)*(-1 + 45*pow(cos(theta),2) - 255*pow(cos(theta),4) + 323*pow(cos(theta),6))*sin(4*phi)*pow(sin(theta),4))/256.;
					case -3:
					return (3*sqrt(5005/(2.*PI))*cos(theta)*(-7 + 105*pow(cos(theta),2) - 357*pow(cos(theta),4) + 323*pow(cos(theta),6))*sin(3*phi)*pow(sin(theta),3))/128.;
					case -2:
					return (-3*sqrt(385/PI)*cos(phi)*(7 - 364*pow(cos(theta),2) + 2730*pow(cos(theta),4) - 6188*pow(cos(theta),6) + 4199*pow(cos(theta),8))*sin(phi)*pow(sin(theta),2))/256.;
					case -1:
					return (sqrt(1155/PI)*cos(theta)*(63 - 1092*pow(cos(theta),2) + 4914*pow(cos(theta),4) - 7956*pow(cos(theta),6) + 4199*pow(cos(theta),8))*sin(phi)*sin(theta))/256.;
					case 0:
					return (sqrt(21/PI)*(-63 + 3465*pow(cos(theta),2) - 30030*pow(cos(theta),4) + 90090*pow(cos(theta),6) - 109395*pow(cos(theta),8) + 46189*pow(cos(theta),10)))/512.;
					case 1:
					return (sqrt(1155/PI)*cos(phi)*cos(theta)*(63 - 1092*pow(cos(theta),2) + 4914*pow(cos(theta),4) - 7956*pow(cos(theta),6) + 4199*pow(cos(theta),8))*sin(theta))/256.;
					case 2:
					return (3*sqrt(385/PI)*cos(2*phi)*(7 - 364*pow(cos(theta),2) + 2730*pow(cos(theta),4) - 6188*pow(cos(theta),6) + 4199*pow(cos(theta),8))*pow(sin(theta),2))/512.;
					case 3:
					return (3*sqrt(5005/(2.*PI))*cos(3*phi)*cos(theta)*(-7 + 105*pow(cos(theta),2) - 357*pow(cos(theta),4) + 323*pow(cos(theta),6))*pow(sin(theta),3))/128.;
					case 4:
					return (3*sqrt(5005/PI)*cos(4*phi)*(-1 + 45*pow(cos(theta),2) - 255*pow(cos(theta),4) + 323*pow(cos(theta),6))*pow(sin(theta),4))/256.;
					case 5:
					return (3*sqrt(1001/(2.*PI))*cos(5*phi)*cos(theta)*(15 - 170*pow(cos(theta),2) + 323*pow(cos(theta),4))*pow(sin(theta),5))/128.;
					case 6:
					return (3*sqrt(5005/(2.*PI))*cos(6*phi)*(3 - 102*pow(cos(theta),2) + 323*pow(cos(theta),4))*pow(sin(theta),6))/512.;
					case 7:
					return (3*sqrt(85085/(2.*PI))*cos(7*phi)*cos(theta)*(13 + 19*cos(2*theta))*pow(sin(theta),7))/512.;
					case 8:
					return (sqrt(255255/PI)*cos(8*phi)*(-1 + 19*pow(cos(theta),2))*pow(sin(theta),8))/512.;
					case 9:
					return (sqrt(4849845/(2.*PI))*cos(9*phi)*cos(theta)*pow(sin(theta),9))/256.;
					case 10:
					return (sqrt(969969/(2.*PI))*cos(10*phi)*pow(sin(theta),10))/512.;
					default:
					throw; return 0.0;
				}
				case 11:
				switch (m){
					case -11:
					return (sqrt(2028117/PI)*sin(11*phi)*pow(sin(theta),11))/1024.;
					case -10:
					return -(sqrt(22309287/(2.*PI))*cos(theta)*sin(10*phi)*pow(sin(theta),10))/512.;
					case -9:
					return (sqrt(1062347/PI)*(19 + 21*cos(2*theta))*sin(9*phi)*pow(sin(theta),9))/2048.;
					case -8:
					return -(sqrt(15935205/PI)*cos(theta)*(5 + 7*cos(2*theta))*sin(8*phi)*pow(sin(theta),8))/1024.;
					case -7:
					return (sqrt(838695/PI)*(1 - 38*pow(cos(theta),2) + 133*pow(cos(theta),4))*sin(7*phi)*pow(sin(theta),7))/1024.;
					case -6:
					return -(sqrt(167739/(2.*PI))*cos(theta)*(15 - 190*pow(cos(theta),2) + 399*pow(cos(theta),4))*sin(6*phi)*pow(sin(theta),6))/512.;
					case -5:
					return (3*sqrt(3289/PI)*(-5 + 255*pow(cos(theta),2) - 1615*pow(cos(theta),4) + 2261*pow(cos(theta),6))*sin(5*phi)*pow(sin(theta),5))/1024.;
					case -4:
					return (-3*sqrt(23023/PI)*cos(theta)*(-5 + 85*pow(cos(theta),2) - 323*pow(cos(theta),4) + 323*pow(cos(theta),6))*sin(4*phi)*pow(sin(theta),4))/256.;
					case -3:
					return (sqrt(345345/(2.*PI))*(1 - 60*pow(cos(theta),2) + 510*pow(cos(theta),4) - 1292*pow(cos(theta),6) + 969*pow(cos(theta),8))*sin(3*phi)*pow(sin(theta),3))/512.;
					case -2:
					return -(sqrt(49335/PI)*cos(phi)*cos(theta)*(21 - 420*pow(cos(theta),2) + 2142*pow(cos(theta),4) - 3876*pow(cos(theta),6) + 2261*pow(cos(theta),8))*sin(phi)*pow(sin(theta),2))/256.;
					case -1:
					return (sqrt(759/(2.*PI))*(-21 + 1365*pow(cos(theta),2) - 13650*pow(cos(theta),4) + 46410*pow(cos(theta),6) - 62985*pow(cos(theta),8) + 29393*pow(cos(theta),10))*sin(phi)*sin(theta))/512.;
					case 0:
					return (sqrt(23/PI)*(29106*cos(theta) + 13*(2310*cos(3*theta) + 2475*cos(5*theta) + 2805*cos(7*theta) + 3553*cos(9*theta) + 6783*cos(11*theta))))/524288.;
					case 1:
					return (sqrt(759/(2.*PI))*cos(phi)*(-21 + 1365*pow(cos(theta),2) - 13650*pow(cos(theta),4) + 46410*pow(cos(theta),6) - 62985*pow(cos(theta),8) + 29393*pow(cos(theta),10))*sin(theta))/512.;
					case 2:
					return (sqrt(49335/PI)*cos(2*phi)*cos(theta)*(21 - 420*pow(cos(theta),2) + 2142*pow(cos(theta),4) - 3876*pow(cos(theta),6) + 2261*pow(cos(theta),8))*pow(sin(theta),2))/512.;
					case 3:
					return (sqrt(345345/(2.*PI))*cos(3*phi)*(1 - 60*pow(cos(theta),2) + 510*pow(cos(theta),4) - 1292*pow(cos(theta),6) + 969*pow(cos(theta),8))*pow(sin(theta),3))/512.;
					case 4:
					return (3*sqrt(23023/PI)*cos(4*phi)*cos(theta)*(-5 + 85*pow(cos(theta),2) - 323*pow(cos(theta),4) + 323*pow(cos(theta),6))*pow(sin(theta),4))/256.;
					case 5:
					return (3*sqrt(3289/PI)*cos(5*phi)*(-5 + 255*pow(cos(theta),2) - 1615*pow(cos(theta),4) + 2261*pow(cos(theta),6))*pow(sin(theta),5))/1024.;
					case 6:
					return (sqrt(167739/(2.*PI))*cos(6*phi)*cos(theta)*(15 - 190*pow(cos(theta),2) + 399*pow(cos(theta),4))*pow(sin(theta),6))/512.;
					case 7:
					return (sqrt(838695/PI)*cos(7*phi)*(1 - 38*pow(cos(theta),2) + 133*pow(cos(theta),4))*pow(sin(theta),7))/1024.;
					case 8:
					return (sqrt(15935205/PI)*cos(8*phi)*cos(theta)*(5 + 7*cos(2*theta))*pow(sin(theta),8))/1024.;
					case 9:
					return (sqrt(1062347/PI)*cos(9*phi)*(19 + 21*cos(2*theta))*pow(sin(theta),9))/2048.;
					case 10:
					return (sqrt(22309287/(2.*PI))*cos(10*phi)*cos(theta)*pow(sin(theta),10))/512.;
					case 11:
					return (sqrt(2028117/PI)*cos(11*phi)*pow(sin(theta),11))/1024.;
					default:
					throw; return 0.0;
				}
				default:
				return 0.0;
			}
		}

		// 1-18-2017 Sped up a bit with precomputation.
		// x_ etc. are arrays up to [9] where x[0] = x*x x[9] = x**11
		double CartSphericalHarmonic(int& l, int& m, double& x, double& y, double& z ,double& r ,double* x_, double* y_, double* z_, double* r_)
		{
			// These should be precomputed in MolEmb.cpp and passed here I think since they will be the same for all lm.
			if (l > 11)
			{
				cout << "lmax = 11" << endl;
				throw;
			}
			switch (l)
			{
				case 0:
				switch (m){
					case 0:
					return 0.28209479177387814;
					default:
					throw; return 0.0;
				}
				case 1:
				switch (m){
					case -1:
					return (0.4886025119029199*y)*r;
					case 0:
					return (0.4886025119029199*z)*r;
					case 1:
					return (0.4886025119029199*x)*r;
					default:
					throw; return 0.0;
				}
				case 2:
				switch (m){
					case -2:
					return (-1.0925484305920792*x*y)*r_[0];
					case -1:
					return (1.0925484305920792*y*z)*r_[0];
					case 0:
					return (-0.31539156525252005*(x_[0] + y_[0] - 2.*z_[0]))*r_[0];
					case 1:
					return (1.0925484305920792*x*z)*r_[0];
					case 2:
					return (0.5462742152960396*(x_[0] - 1.*y_[0]))*r_[0];
					default:
					throw; return 0.0;
				}
				case 3:
				switch (m){
					case -3:
					return (-0.5900435899266435*y*(-3.*x_[0] + y_[0]))*r_[1];
					case -2:
					return (-2.890611442640554*x*y*z)*r_[1];
					case -1:
					return (-0.4570457994644658*y*(x_[0] + y_[0] - 4.*z_[0]))*r_[1];
					case 0:
					return (0.3731763325901154*z*(-3.*x_[0] - 3.*y_[0] + 2.*z_[0]))*r_[1];
					case 1:
					return (-0.4570457994644658*x*(x_[0] + y_[0] - 4.*z_[0]))*r_[1];
					case 2:
					return (1.445305721320277*(x_[0] - 1.*y_[0])*z)*r_[1];
					case 3:
					return (0.5900435899266435*x*(x_[0] - 3.*y_[0]))*r_[1];
					default:
					throw; return 0.0;
				}
				case 4:
				switch (m){
					case -4:
					return (2.5033429417967046*x*y*(-1.*x_[0] + y_[0]))*r_[2];
					case -3:
					return (-1.7701307697799304*y*(-3.*x_[0] + y_[0])*z)*r_[2];
					case -2:
					return (0.9461746957575601*x*y*(x_[0] + y_[0] - 6.*z_[0]))*r_[2];
					case -1:
					return (-0.6690465435572892*y*z*(3.*x_[0] + 3.*y_[0] - 4.*z_[0]))*r_[2];
					case 0:
					return (0.10578554691520431*(3.*x_[2] + 3.*y_[2] - 24.*y_[0]*z_[0] + 8.*z_[2] + 6.*x_[0]*(y_[0] - 4.*z_[0])))*r_[2];
					case 1:
					return (-0.6690465435572892*x*z*(3.*x_[0] + 3.*y_[0] - 4.*z_[0]))*r_[2];
					case 2:
					return (-0.47308734787878004*(x_[0] - 1.*y_[0])*(x_[0] + y_[0] - 6.*z_[0]))*r_[2];
					case 3:
					return (1.7701307697799304*x*(x_[0] - 3.*y_[0])*z)*r_[2];
					case 4:
					return (0.6258357354491761*(x_[2] - 6.*x_[0]*y_[0] + y_[2]))*r_[2];
					default:
					throw; return 0.0;
				}
				case 5:
				switch (m){
					case -5:
					return (0.6563820568401701*y*(5.*x_[2] - 10.*x_[0]*y_[0] + y_[2]))*r_[3];
					case -4:
					return (8.302649259524166*x*y*(-1.*x_[0] + y_[0])*z)*r_[3];
					case -3:
					return (0.4892382994352504*y*(-3.*x_[0] + y_[0])*(x_[0] + y_[0] - 8.*z_[0]))*r_[3];
					case -2:
					return (4.793536784973324*x*y*z*(x_[0] + y_[0] - 2.*z_[0]))*r_[3];
					case -1:
					return (0.45294665119569694*y*(x_[2] + y_[2] - 12.*y_[0]*z_[0] + 8.*z_[2] + 2.*x_[0]*(y_[0] - 6.*z_[0])))*r_[3];
					case 0:
					return (0.1169503224534236*z*(15.*x_[2] + 15.*y_[2] - 40.*y_[0]*z_[0] + 8.*z_[2] + 10.*x_[0]*(3.*y_[0] - 4.*z_[0])))*r_[3];
					case 1:
					return (0.45294665119569694*x*(x_[2] + y_[2] - 12.*y_[0]*z_[0] + 8.*z_[2] + 2.*x_[0]*(y_[0] - 6.*z_[0])))*r_[3];
					case 2:
					return (-2.396768392486662*(x_[0] - 1.*y_[0])*z*(x_[0] + y_[0] - 2.*z_[0]))*r_[3];
					case 3:
					return (-0.4892382994352504*x*(x_[0] - 3.*y_[0])*(x_[0] + y_[0] - 8.*z_[0]))*r_[3];
					case 4:
					return (2.0756623148810416*(x_[2] - 6.*x_[0]*y_[0] + y_[2])*z)*r_[3];
					case 5:
					return (0.6563820568401701*x*(x_[2] - 10.*x_[0]*y_[0] + 5.*y_[2]))*r_[3];
					default:
					throw; return 0.0;
				}
				case 6:
				switch (m){
					case -6:
					return (-1.3663682103838286*x*y*(3.*x_[2] - 10.*x_[0]*y_[0] + 3.*y_[2]))*r_[4];
					case -5:
					return (2.366619162231752*y*(5.*x_[2] - 10.*x_[0]*y_[0] + y_[2])*z)*r_[4];
					case -4:
					return (2.0182596029148967*x*y*(x_[0] - 1.*y_[0])*(x_[0] + y_[0] - 10.*z_[0]))*r_[4];
					case -3:
					return (0.9212052595149236*y*(-3.*x_[0] + y_[0])*z*(3.*x_[0] + 3.*y_[0] - 8.*z_[0]))*r_[4];
					case -2:
					return (-0.9212052595149236*x*y*(x_[2] + y_[2] - 16.*y_[0]*z_[0] + 16.*z_[2] + 2.*x_[0]*(y_[0] - 8.*z_[0])))*r_[4];
					case -1:
					return (0.5826213625187314*y*z*(5.*x_[2] + 5.*y_[2] - 20.*y_[0]*z_[0] + 8.*z_[2] + 10.*x_[0]*(y_[0] - 2.*z_[0])))*r_[4];
					case 0:
					return (-0.06356920226762842*(5.*x_[4] + 5.*y_[4] - 90.*y_[2]*z_[0] + 120.*y_[0]*z_[2] - 16.*z_[4] + 15.*x_[2]*(y_[0] - 6.*z_[0]) + 15.*x_[0]*(y_[2] - 12.*y_[0]*z_[0] + 8.*z_[2])))*r_[4];
					case 1:
					return (0.5826213625187314*x*z*(5.*x_[2] + 5.*y_[2] - 20.*y_[0]*z_[0] + 8.*z_[2] + 10.*x_[0]*(y_[0] - 2.*z_[0])))*r_[4];
					case 2:
					return (0.4606026297574618*(x_[0] - 1.*y_[0])*(x_[2] + y_[2] - 16.*y_[0]*z_[0] + 16.*z_[2] + 2.*x_[0]*(y_[0] - 8.*z_[0])))*r_[4];
					case 3:
					return (-0.9212052595149236*x*(x_[0] - 3.*y_[0])*z*(3.*x_[0] + 3.*y_[0] - 8.*z_[0]))*r_[4];
					case 4:
					return (-0.5045649007287242*(x_[2] - 6.*x_[0]*y_[0] + y_[2])*(x_[0] + y_[0] - 10.*z_[0]))*r_[4];
					case 5:
					return (2.366619162231752*x*(x_[2] - 10.*x_[0]*y_[0] + 5.*y_[2])*z)*r_[4];
					case 6:
					return (0.6831841051919143*(x_[4] - 15.*x_[2]*y_[0] + 15.*x_[0]*y_[2] - 1.*y_[4]))*r_[4];
					default:
					throw; return 0.0;
				}
				case 7:
				switch (m){
					case -7:
					return (-0.7071627325245962*y*(-7.*x_[4] + 35.*x_[2]*y_[0] - 21.*x_[0]*y_[2] + y_[4]))*r_[5];
					case -6:
					return (-5.291921323603801*x*y*(3.*x_[2] - 10.*x_[0]*y_[0] + 3.*y_[2])*z)*r_[5];
					case -5:
					return (-0.5189155787202604*y*(5.*x_[2] - 10.*x_[0]*y_[0] + y_[2])*(x_[0] + y_[0] - 12.*z_[0]))*r_[5];
					case -4:
					return (4.151324629762083*x*y*(x_[0] - 1.*y_[0])*z*(3.*x_[0] + 3.*y_[0] - 10.*z_[0]))*r_[5];
					case -3:
					return (-0.15645893386229404*y*(-3.*x_[0] + y_[0])*(3.*x_[2] + 3.*y_[2] - 60.*y_[0]*z_[0] + 80.*z_[2] + 6.*x_[0]*(y_[0] - 10.*z_[0])))*r_[5];
					case -2:
					return (-0.4425326924449826*x*y*z*(15.*x_[2] + 15.*y_[2] - 80.*y_[0]*z_[0] + 48.*z_[2] + 10.*x_[0]*(3.*y_[0] - 8.*z_[0])))*r_[5];
					case -1:
					return (-0.0903316075825173*y*(5.*x_[4] + 5.*y_[4] - 120.*y_[2]*z_[0] + 240.*y_[0]*z_[2] - 64.*z_[4] + 15.*x_[2]*(y_[0] - 8.*z_[0]) + 15.*x_[0]*(y_[2] - 16.*y_[0]*z_[0] + 16.*z_[2])))*r_[5];
					case 0:
					return (0.06828427691200495*z*(-35.*x_[4] - 35.*y_[4] + 210.*y_[2]*z_[0] - 168.*y_[0]*z_[2] + 16.*z_[4] - 105.*x_[2]*(y_[0] - 2.*z_[0]) - 21.*x_[0]*(5.*y_[2] - 20.*y_[0]*z_[0] + 8.*z_[2])))*r_[5];
					case 1:
					return (-0.0903316075825173*x*(5.*x_[4] + 5.*y_[4] - 120.*y_[2]*z_[0] + 240.*y_[0]*z_[2] - 64.*z_[4] + 15.*x_[2]*(y_[0] - 8.*z_[0]) + 15.*x_[0]*(y_[2] - 16.*y_[0]*z_[0] + 16.*z_[2])))*r_[5];
					case 2:
					return (0.2212663462224913*(x_[0] - 1.*y_[0])*z*(15.*x_[2] + 15.*y_[2] - 80.*y_[0]*z_[0] + 48.*z_[2] + 10.*x_[0]*(3.*y_[0] - 8.*z_[0])))*r_[5];
					case 3:
					return (0.15645893386229404*x*(x_[0] - 3.*y_[0])*(3.*x_[2] + 3.*y_[2] - 60.*y_[0]*z_[0] + 80.*z_[2] + 6.*x_[0]*(y_[0] - 10.*z_[0])))*r_[5];
					case 4:
					return (-1.0378311574405208*(x_[2] - 6.*x_[0]*y_[0] + y_[2])*z*(3.*x_[0] + 3.*y_[0] - 10.*z_[0]))*r_[5];
					case 5:
					return (-0.5189155787202604*x*(x_[2] - 10.*x_[0]*y_[0] + 5.*y_[2])*(x_[0] + y_[0] - 12.*z_[0]))*r_[5];
					case 6:
					return (2.6459606618019005*(x_[4] - 15.*x_[2]*y_[0] + 15.*x_[0]*y_[2] - 1.*y_[4])*z)*r_[5];
					case 7:
					return (0.7071627325245962*x*(x_[4] - 21.*x_[2]*y_[0] + 35.*x_[0]*y_[2] - 7.*y_[4]))*r_[5];
					default:
					throw; return 0.0;
				}
				case 8:
				switch (m){
					case -8:
					return (-5.831413281398639*x*y*(x_[4] - 7.*x_[2]*y_[0] + 7.*x_[0]*y_[2] - 1.*y_[4]))*r_[6];
					case -7:
					return (-2.9157066406993195*y*(-7.*x_[4] + 35.*x_[2]*y_[0] - 21.*x_[0]*y_[2] + y_[4])*z)*r_[6];
					case -6:
					return (1.0646655321190852*x*y*(3.*x_[2] - 10.*x_[0]*y_[0] + 3.*y_[2])*(x_[0] + y_[0] - 14.*z_[0]))*r_[6];
					case -5:
					return (-3.449910622098108*y*(5.*x_[2] - 10.*x_[0]*y_[0] + y_[2])*z*(x_[0] + y_[0] - 4.*z_[0]))*r_[6];
					case -4:
					return (-1.9136660990373227*x*y*(x_[0] - 1.*y_[0])*(x_[2] + y_[2] - 24.*y_[0]*z_[0] + 40.*z_[2] + 2.*x_[0]*(y_[0] - 12.*z_[0])))*r_[6];
					case -3:
					return (-1.2352661552955442*y*(-3.*x_[0] + y_[0])*z*(3.*x_[2] + 3.*y_[2] - 20.*y_[0]*z_[0] + 16.*z_[2] + x_[0]*(6.*y_[0] - 20.*z_[0])))*r_[6];
					case -2:
					return (0.912304516869819*x*y*(x_[4] + y_[4] - 30.*y_[2]*z_[0] + 80.*y_[0]*z_[2] - 32.*z_[4] + 3.*x_[2]*(y_[0] - 10.*z_[0]) + x_[0]*(3.*y_[2] - 60.*y_[0]*z_[0] + 80.*z_[2])))*r_[6];
					case -1:
					return (-0.10904124589877995*y*z*(35.*x_[4] + 35.*y_[4] - 280.*y_[2]*z_[0] + 336.*y_[0]*z_[2] - 64.*z_[4] + 35.*x_[2]*(3.*y_[0] - 8.*z_[0]) + 7.*x_[0]*(15.*y_[2] - 80.*y_[0]*z_[0] + 48.*z_[2])))*r_[6];
					case 0:
					return (0.009086770491564996*(35.*x_[6] + 35.*y_[6] - 1120.*y_[4]*z_[0] + 3360.*y_[2]*z_[2] - 1792.*y_[0]*z_[4] + 128.*z_[6] + 140.*x_[4]*(y_[0] - 8.*z_[0]) + 210.*x_[2]*(y_[2] - 16.*y_[0]*z_[0] + 16.*z_[2]) + 28.*x_[0]*(5.*y_[4] - 120.*y_[2]*z_[0] + 240.*y_[0]*z_[2] - 64.*z_[4])))*r_[6];
					case 1:
					return (-0.10904124589877995*x*z*(35.*x_[4] + 35.*y_[4] - 280.*y_[2]*z_[0] + 336.*y_[0]*z_[2] - 64.*z_[4] + 35.*x_[2]*(3.*y_[0] - 8.*z_[0]) + 7.*x_[0]*(15.*y_[2] - 80.*y_[0]*z_[0] + 48.*z_[2])))*r_[6];
					case 2:
					return (-0.4561522584349095*(x_[0] - 1.*y_[0])*(x_[4] + y_[4] - 30.*y_[2]*z_[0] + 80.*y_[0]*z_[2] - 32.*z_[4] + 3.*x_[2]*(y_[0] - 10.*z_[0]) + x_[0]*(3.*y_[2] - 60.*y_[0]*z_[0] + 80.*z_[2])))*r_[6];
					case 3:
					return (1.2352661552955442*x*(x_[0] - 3.*y_[0])*z*(3.*x_[2] + 3.*y_[2] - 20.*y_[0]*z_[0] + 16.*z_[2] + x_[0]*(6.*y_[0] - 20.*z_[0])))*r_[6];
					case 4:
					return (0.47841652475933066*(x_[2] - 6.*x_[0]*y_[0] + y_[2])*(x_[2] + y_[2] - 24.*y_[0]*z_[0] + 40.*z_[2] + 2.*x_[0]*(y_[0] - 12.*z_[0])))*r_[6];
					case 5:
					return (-3.449910622098108*x*(x_[2] - 10.*x_[0]*y_[0] + 5.*y_[2])*z*(x_[0] + y_[0] - 4.*z_[0]))*r_[6];
					case 6:
					return (-0.5323327660595426*(x_[4] - 15.*x_[2]*y_[0] + 15.*x_[0]*y_[2] - 1.*y_[4])*(x_[0] + y_[0] - 14.*z_[0]))*r_[6];
					case 7:
					return (2.9157066406993195*x*(x_[4] - 21.*x_[2]*y_[0] + 35.*x_[0]*y_[2] - 7.*y_[4])*z)*r_[6];
					case 8:
					return (0.7289266601748299*(x_[6] - 28.*x_[4]*y_[0] + 70.*x_[2]*y_[2] - 28.*x_[0]*y_[4] + y_[6]))*r_[6];
					default:
					throw; return 0.0;
				}
				case 9:
				switch (m){
					case -9:
					return (0.7489009518531883*y*(9.*x_[6] - 84.*x_[4]*y_[0] + 126.*x_[2]*y_[2] - 36.*x_[0]*y_[4] + y_[6]))*r_[7];
					case -8:
					return (-25.41854119163758*x*y*(x_[4] - 7.*x_[2]*y_[0] + 7.*x_[0]*y_[2] - 1.*y_[4])*z)*r_[7];
					case -7:
					return (0.5449054813440533*y*(-7.*x_[4] + 35.*x_[2]*y_[0] - 21.*x_[0]*y_[2] + y_[4])*(x_[0] + y_[0] - 16.*z_[0]))*r_[7];
					case -6:
					return (2.516810610695134*x*y*(3.*x_[2] - 10.*x_[0]*y_[0] + 3.*y_[2])*z*(3.*x_[0] + 3.*y_[0] - 14.*z_[0]))*r_[7];
					case -5:
					return (0.4873782790390186*y*(5.*x_[2] - 10.*x_[0]*y_[0] + y_[2])*(x_[2] + y_[2] - 28.*y_[0]*z_[0] + 56.*z_[2] + 2.*x_[0]*(y_[0] - 14.*z_[0])))*r_[7];
					case -4:
					return (-16.310796954916693*x*y*(x_[0] - 1.*y_[0])*z*(x_[2] + y_[2] - 8.*y_[0]*z_[0] + 8.*z_[2] + 2.*x_[0]*(y_[0] - 4.*z_[0])))*r_[7];
					case -3:
					return (0.46170852001619445*y*(-3.*x_[0] + y_[0])*(x_[4] + y_[4] - 36.*y_[2]*z_[0] + 120.*y_[0]*z_[2] - 64.*z_[4] + 3.*x_[2]*(y_[0] - 12.*z_[0]) + 3.*x_[0]*(y_[2] - 24.*y_[0]*z_[0] + 40.*z_[2])))*r_[7];
					case -2:
					return (1.209036709702997*x*y*z*(7.*x_[4] + 7.*y_[4] - 70.*y_[2]*z_[0] + 112.*y_[0]*z_[2] - 32.*z_[4] + 7.*x_[2]*(3.*y_[0] - 10.*z_[0]) + 7.*x_[0]*(3.*y_[2] - 20.*y_[0]*z_[0] + 16.*z_[2])))*r_[7];
					case -1:
					return (0.0644418731522273*y*(7.*x_[6] + 7.*y_[6] - 280.*y_[4]*z_[0] + 1120.*y_[2]*z_[2] - 896.*y_[0]*z_[4] + 128.*z_[6] + 28.*x_[4]*(y_[0] - 10.*z_[0]) + 14.*x_[2]*(3.*y_[2] - 60.*y_[0]*z_[0] + 80.*z_[2]) + 28.*x_[0]*(y_[4] - 30.*y_[2]*z_[0] + 80.*y_[0]*z_[2] - 32.*z_[4])))*r_[7];
					case 0:
					return (0.009606427264386591*z*(315.*x_[6] + 315.*y_[6] - 3360.*y_[4]*z_[0] + 6048.*y_[2]*z_[2] - 2304.*y_[0]*z_[4] + 128.*z_[6] + 420.*x_[4]*(3.*y_[0] - 8.*z_[0]) + 126.*x_[2]*(15.*y_[2] - 80.*y_[0]*z_[0] + 48.*z_[2]) + 36.*x_[0]*(35.*y_[4] - 280.*y_[2]*z_[0] + 336.*y_[0]*z_[2] - 64.*z_[4])))*r_[7];
					case 1:
					return (0.0644418731522273*x*(7.*x_[6] + 7.*y_[6] - 280.*y_[4]*z_[0] + 1120.*y_[2]*z_[2] - 896.*y_[0]*z_[4] + 128.*z_[6] + 28.*x_[4]*(y_[0] - 10.*z_[0]) + 14.*x_[2]*(3.*y_[2] - 60.*y_[0]*z_[0] + 80.*z_[2]) + 28.*x_[0]*(y_[4] - 30.*y_[2]*z_[0] + 80.*y_[0]*z_[2] - 32.*z_[4])))*r_[7];
					case 2:
					return (-0.6045183548514985*(x_[0] - 1.*y_[0])*z*(7.*x_[4] + 7.*y_[4] - 70.*y_[2]*z_[0] + 112.*y_[0]*z_[2] - 32.*z_[4] + 7.*x_[2]*(3.*y_[0] - 10.*z_[0]) + 7.*x_[0]*(3.*y_[2] - 20.*y_[0]*z_[0] + 16.*z_[2])))*r_[7];
					case 3:
					return (-0.46170852001619445*x*(x_[0] - 3.*y_[0])*(x_[4] + y_[4] - 36.*y_[2]*z_[0] + 120.*y_[0]*z_[2] - 64.*z_[4] + 3.*x_[2]*(y_[0] - 12.*z_[0]) + 3.*x_[0]*(y_[2] - 24.*y_[0]*z_[0] + 40.*z_[2])))*r_[7];
					case 4:
					return (4.077699238729173*(x_[2] - 6.*x_[0]*y_[0] + y_[2])*z*(x_[2] + y_[2] - 8.*y_[0]*z_[0] + 8.*z_[2] + 2.*x_[0]*(y_[0] - 4.*z_[0])))*r_[7];
					case 5:
					return (0.4873782790390186*x*(x_[2] - 10.*x_[0]*y_[0] + 5.*y_[2])*(x_[2] + y_[2] - 28.*y_[0]*z_[0] + 56.*z_[2] + 2.*x_[0]*(y_[0] - 14.*z_[0])))*r_[7];
					case 6:
					return (-1.258405305347567*(x_[4] - 15.*x_[2]*y_[0] + 15.*x_[0]*y_[2] - 1.*y_[4])*z*(3.*x_[0] + 3.*y_[0] - 14.*z_[0]))*r_[7];
					case 7:
					return (-0.5449054813440533*x*(x_[4] - 21.*x_[2]*y_[0] + 35.*x_[0]*y_[2] - 7.*y_[4])*(x_[0] + y_[0] - 16.*z_[0]))*r_[7];
					case 8:
					return (3.1773176489546975*(x_[6] - 28.*x_[4]*y_[0] + 70.*x_[2]*y_[2] - 28.*x_[0]*y_[4] + y_[6])*z)*r_[7];
					case 9:
					return (0.7489009518531883*x*(x_[6] - 36.*x_[4]*y_[0] + 126.*x_[2]*y_[2] - 84.*x_[0]*y_[4] + 9.*y_[6]))*r_[7];
					default:
					throw; return 0.0;
				}
				case 10:
				switch (m){
					case -10:
					return (-1.53479023644398*x*y*(5.*x_[6] - 60.*x_[4]*y_[0] + 126.*x_[2]*y_[2] - 60.*x_[0]*y_[4] + 5.*y_[6]))*r_[8];
					case -9:
					return (3.4318952998917145*y*(9.*x_[6] - 84.*x_[4]*y_[0] + 126.*x_[2]*y_[2] - 36.*x_[0]*y_[4] + y_[6])*z)*r_[8];
					case -8:
					return (4.453815461763347*x*y*(x_[4] - 7.*x_[2]*y_[0] + 7.*x_[0]*y_[2] - 1.*y_[4])*(x_[0] + y_[0] - 18.*z_[0]))*r_[8];
					case -7:
					return (1.3636969112298054*y*(-7.*x_[4] + 35.*x_[2]*y_[0] - 21.*x_[0]*y_[2] + y_[4])*z*(3.*x_[0] + 3.*y_[0] - 16.*z_[0]))*r_[8];
					case -6:
					return (-0.33074508272523756*x*y*(3.*x_[2] - 10.*x_[0]*y_[0] + 3.*y_[2])*(3.*x_[2] + 3.*y_[2] - 96.*y_[0]*z_[0] + 224.*z_[2] + 6.*x_[0]*(y_[0] - 16.*z_[0])))*r_[8];
					case -5:
					return (0.295827395278969*y*(5.*x_[2] - 10.*x_[0]*y_[0] + y_[2])*z*(15.*x_[2] + 15.*y_[2] - 140.*y_[0]*z_[0] + 168.*z_[2] + 10.*x_[0]*(3.*y_[0] - 14.*z_[0])))*r_[8];
					case -4:
					return (1.8709767267129689*x*y*(x_[0] - 1.*y_[0])*(x_[4] + y_[4] - 42.*y_[2]*z_[0] + 168.*y_[0]*z_[2] - 112.*z_[4] + 3.*x_[2]*(y_[0] - 14.*z_[0]) + 3.*x_[0]*(y_[2] - 28.*y_[0]*z_[0] + 56.*z_[2])))*r_[8];
					case -3:
					return (0.6614901654504751*y*(-3.*x_[0] + y_[0])*z*(7.*x_[4] + 7.*y_[4] - 84.*y_[2]*z_[0] + 168.*y_[0]*z_[2] - 64.*z_[4] + 21.*x_[2]*(y_[0] - 4.*z_[0]) + 21.*x_[0]*(y_[2] - 8.*y_[0]*z_[0] + 8.*z_[2])))*r_[8];
					case -2:
					return (-0.1297288946800651*x*y*(7.*x_[6] + 7.*y_[6] - 336.*y_[4]*z_[0] + 1680.*y_[2]*z_[2] - 1792.*y_[0]*z_[4] + 384.*z_[6] + 28.*x_[4]*(y_[0] - 12.*z_[0]) + 42.*x_[2]*(y_[2] - 24.*y_[0]*z_[0] + 40.*z_[2]) + 28.*x_[0]*(y_[4] - 36.*y_[2]*z_[0] + 120.*y_[0]*z_[2] - 64.*z_[4])))*r_[8];
					case -1:
					return (0.07489901226520819*y*z*(63.*x_[6] + 63.*y_[6] - 840.*y_[4]*z_[0] + 2016.*y_[2]*z_[2] - 1152.*y_[0]*z_[4] + 128.*z_[6] + 84.*x_[4]*(3.*y_[0] - 10.*z_[0]) + 126.*x_[2]*(3.*y_[2] - 20.*y_[0]*z_[0] + 16.*z_[2]) + 36.*x_[0]*(7.*y_[4] - 70.*y_[2]*z_[0] + 112.*y_[0]*z_[2] - 32.*z_[4])))*r_[8];
					case 0:
					return (-0.005049690376783604*(63.*x_[8] + 63.*y_[8] - 3150.*y_[6]*z_[0] + 16800.*y_[4]*z_[2] - 20160.*y_[2]*z_[4] + 5760.*y_[0]*z_[6] - 256.*z_[8] + 315.*x_[6]*(y_[0] - 10.*z_[0]) + 210.*x_[4]*(3.*y_[2] - 60.*y_[0]*z_[0] + 80.*z_[2]) + 630.*x_[2]*(y_[4] - 30.*y_[2]*z_[0] + 80.*y_[0]*z_[2] - 32.*z_[4]) + 45.*x_[0]*(7.*y_[6] - 280.*y_[4]*z_[0] + 1120.*y_[2]*z_[2] - 896.*y_[0]*z_[4] + 128.*z_[6])))*r_[8];
					case 1:
					return (0.07489901226520819*x*z*(63.*x_[6] + 63.*y_[6] - 840.*y_[4]*z_[0] + 2016.*y_[2]*z_[2] - 1152.*y_[0]*z_[4] + 128.*z_[6] + 84.*x_[4]*(3.*y_[0] - 10.*z_[0]) + 126.*x_[2]*(3.*y_[2] - 20.*y_[0]*z_[0] + 16.*z_[2]) + 36.*x_[0]*(7.*y_[4] - 70.*y_[2]*z_[0] + 112.*y_[0]*z_[2] - 32.*z_[4])))*r_[8];
					case 2:
					return (0.06486444734003255*(x_[0] - 1.*y_[0])*(7.*x_[6] + 7.*y_[6] - 336.*y_[4]*z_[0] + 1680.*y_[2]*z_[2] - 1792.*y_[0]*z_[4] + 384.*z_[6] + 28.*x_[4]*(y_[0] - 12.*z_[0]) + 42.*x_[2]*(y_[2] - 24.*y_[0]*z_[0] + 40.*z_[2]) + 28.*x_[0]*(y_[4] - 36.*y_[2]*z_[0] + 120.*y_[0]*z_[2] - 64.*z_[4])))*r_[8];
					case 3:
					return (-0.6614901654504751*x*(x_[0] - 3.*y_[0])*z*(7.*x_[4] + 7.*y_[4] - 84.*y_[2]*z_[0] + 168.*y_[0]*z_[2] - 64.*z_[4] + 21.*x_[2]*(y_[0] - 4.*z_[0]) + 21.*x_[0]*(y_[2] - 8.*y_[0]*z_[0] + 8.*z_[2])))*r_[8];
					case 4:
					return (-0.4677441816782422*(x_[2] - 6.*x_[0]*y_[0] + y_[2])*(x_[4] + y_[4] - 42.*y_[2]*z_[0] + 168.*y_[0]*z_[2] - 112.*z_[4] + 3.*x_[2]*(y_[0] - 14.*z_[0]) + 3.*x_[0]*(y_[2] - 28.*y_[0]*z_[0] + 56.*z_[2])))*r_[8];
					case 5:
					return (0.295827395278969*x*(x_[2] - 10.*x_[0]*y_[0] + 5.*y_[2])*z*(15.*x_[2] + 15.*y_[2] - 140.*y_[0]*z_[0] + 168.*z_[2] + 10.*x_[0]*(3.*y_[0] - 14.*z_[0])))*r_[8];
					case 6:
					return (0.16537254136261878*(x_[4] - 15.*x_[2]*y_[0] + 15.*x_[0]*y_[2] - 1.*y_[4])*(3.*x_[2] + 3.*y_[2] - 96.*y_[0]*z_[0] + 224.*z_[2] + 6.*x_[0]*(y_[0] - 16.*z_[0])))*r_[8];
					case 7:
					return (-1.3636969112298054*x*(x_[4] - 21.*x_[2]*y_[0] + 35.*x_[0]*y_[2] - 7.*y_[4])*z*(3.*x_[0] + 3.*y_[0] - 16.*z_[0]))*r_[8];
					case 8:
					return (-0.5567269327204184*(x_[6] - 28.*x_[4]*y_[0] + 70.*x_[2]*y_[2] - 28.*x_[0]*y_[4] + y_[6])*(x_[0] + y_[0] - 18.*z_[0]))*r_[8];
					case 9:
					return (3.4318952998917145*x*(x_[6] - 36.*x_[4]*y_[0] + 126.*x_[2]*y_[2] - 84.*x_[0]*y_[4] + 9.*y_[6])*z)*r_[8];
					case 10:
					return (0.76739511822199*(x_[8] - 45.*x_[6]*y_[0] + 210.*x_[4]*y_[2] - 210.*x_[2]*y_[4] + 45.*x_[0]*y_[6] - 1.*y_[8]))*r_[8];
					default:
					throw; return 0.0;
				}
				case 11:
				switch (m){
					case -11:
					return (-0.7846421057871968*y*(-11.*x_[8] + 165.*x_[6]*y_[0] - 462.*x_[4]*y_[2] + 330.*x_[2]*y_[4] - 55.*x_[0]*y_[6] + y_[8]))*r_[9];
					case -10:
					return (-7.360595397610622*x*y*(5.*x_[6] - 60.*x_[4]*y_[0] + 126.*x_[2]*y_[2] - 60.*x_[0]*y_[4] + 5.*y_[6])*z)*r_[9];
					case -9:
					return (-0.5678822637834374*y*(9.*x_[6] - 84.*x_[4]*y_[0] + 126.*x_[2]*y_[2] - 36.*x_[0]*y_[4] + y_[6])*(x_[0] + y_[0] - 20.*z_[0]))*r_[9];
					case -8:
					return (35.190376803837125*x*y*(x_[4] - 7.*x_[2]*y_[0] + 7.*x_[0]*y_[2] - 1.*y_[4])*z*(x_[0] + y_[0] - 6.*z_[0]))*r_[9];
					case -7:
					return (-0.504576632477118*y*(-7.*x_[4] + 35.*x_[2]*y_[0] - 21.*x_[0]*y_[2] + y_[4])*(x_[2] + y_[2] - 36.*y_[0]*z_[0] + 96.*z_[2] + 2.*x_[0]*(y_[0] - 18.*z_[0])))*r_[9];
					case -6:
					return (-0.6382445650901524*x*y*(3.*x_[2] - 10.*x_[0]*y_[0] + 3.*y_[2])*z*(15.*x_[2] + 15.*y_[2] - 160.*y_[0]*z_[0] + 224.*z_[2] + 10.*x_[0]*(3.*y_[0] - 16.*z_[0])))*r_[9];
					case -5:
					return (-0.09479344319133458*y*(5.*x_[2] - 10.*x_[0]*y_[0] + y_[2])*(5.*x_[4] + 5.*y_[4] - 240.*y_[2]*z_[0] + 1120.*y_[0]*z_[2] - 896.*z_[4] + 15.*x_[2]*(y_[0] - 16.*z_[0]) + 5.*x_[0]*(3.*y_[2] - 96.*y_[0]*z_[0] + 224.*z_[2])))*r_[9];
					case -4:
					return (4.012798025660803*x*y*(x_[0] - 1.*y_[0])*z*(5.*x_[4] + 5.*y_[4] - 70.*y_[2]*z_[0] + 168.*y_[0]*z_[2] - 80.*z_[4] + 5.*x_[2]*(3.*y_[0] - 14.*z_[0]) + x_[0]*(15.*y_[2] - 140.*y_[0]*z_[0] + 168.*z_[2])))*r_[9];
					case -3:
					return (-0.4578958327847118*y*(-3.*x_[0] + y_[0])*(x_[6] + y_[6] - 56.*y_[4]*z_[0] + 336.*y_[2]*z_[2] - 448.*y_[0]*z_[4] + 128.*z_[6] + 4.*x_[4]*(y_[0] - 14.*z_[0]) + 6.*x_[2]*(y_[2] - 28.*y_[0]*z_[0] + 56.*z_[2]) + 4.*x_[0]*(y_[4] - 42.*y_[2]*z_[0] + 168.*y_[0]*z_[2] - 112.*z_[4])))*r_[9];
					case -2:
					return (-0.48951123574626354*x*y*z*(21.*x_[6] + 21.*y_[6] - 336.*y_[4]*z_[0] + 1008.*y_[2]*z_[2] - 768.*y_[0]*z_[4] + 128.*z_[6] + 84.*x_[4]*(y_[0] - 4.*z_[0]) + 126.*x_[2]*(y_[2] - 8.*y_[0]*z_[0] + 8.*z_[2]) + 12.*x_[0]*(7.*y_[4] - 84.*y_[2]*z_[0] + 168.*y_[0]*z_[2] - 64.*z_[4])))*r_[9];
					case -1:
					return (-0.021466487742607707*y*(21.*x_[8] + 21.*y_[8] - 1260.*y_[6]*z_[0] + 8400.*y_[4]*z_[2] - 13440.*y_[2]*z_[4] + 5760.*y_[0]*z_[6] - 512.*z_[8] + 105.*x_[6]*(y_[0] - 12.*z_[0]) + 210.*x_[4]*(y_[2] - 24.*y_[0]*z_[0] + 40.*z_[2]) + 210.*x_[2]*(y_[4] - 36.*y_[2]*z_[0] + 120.*y_[0]*z_[2] - 64.*z_[4]) + 15.*x_[0]*(7.*y_[6] - 336.*y_[4]*z_[0] + 1680.*y_[2]*z_[2] - 1792.*y_[0]*z_[4] + 384.*z_[6])))*r_[9];
					case 0:
					return (0.005284683964654306*z*(-693.*x_[8] - 693.*y_[8] + 11550.*y_[6]*z_[0] - 36960.*y_[4]*z_[2] + 31680.*y_[2]*z_[4] - 7040.*y_[0]*z_[6] + 256.*z_[8] - 1155.*x_[6]*(3.*y_[0] - 10.*z_[0]) - 2310.*x_[4]*(3.*y_[2] - 20.*y_[0]*z_[0] + 16.*z_[2]) - 990.*x_[2]*(7.*y_[4] - 70.*y_[2]*z_[0] + 112.*y_[0]*z_[2] - 32.*z_[4]) - 55.*x_[0]*(63.*y_[6] - 840.*y_[4]*z_[0] + 2016.*y_[2]*z_[2] - 1152.*y_[0]*z_[4] + 128.*z_[6])))*r_[9];
					case 1:
					return (-0.021466487742607707*x*(21.*x_[8] + 21.*y_[8] - 1260.*y_[6]*z_[0] + 8400.*y_[4]*z_[2] - 13440.*y_[2]*z_[4] + 5760.*y_[0]*z_[6] - 512.*z_[8] + 105.*x_[6]*(y_[0] - 12.*z_[0]) + 210.*x_[4]*(y_[2] - 24.*y_[0]*z_[0] + 40.*z_[2]) + 210.*x_[2]*(y_[4] - 36.*y_[2]*z_[0] + 120.*y_[0]*z_[2] - 64.*z_[4]) + 15.*x_[0]*(7.*y_[6] - 336.*y_[4]*z_[0] + 1680.*y_[2]*z_[2] - 1792.*y_[0]*z_[4] + 384.*z_[6])))*r_[9];
					case 2:
					return (0.24475561787313177*(x_[0] - 1.*y_[0])*z*(21.*x_[6] + 21.*y_[6] - 336.*y_[4]*z_[0] + 1008.*y_[2]*z_[2] - 768.*y_[0]*z_[4] + 128.*z_[6] + 84.*x_[4]*(y_[0] - 4.*z_[0]) + 126.*x_[2]*(y_[2] - 8.*y_[0]*z_[0] + 8.*z_[2]) + 12.*x_[0]*(7.*y_[4] - 84.*y_[2]*z_[0] + 168.*y_[0]*z_[2] - 64.*z_[4])))*r_[9];
					case 3:
					return (0.4578958327847118*x*(x_[0] - 3.*y_[0])*(x_[6] + y_[6] - 56.*y_[4]*z_[0] + 336.*y_[2]*z_[2] - 448.*y_[0]*z_[4] + 128.*z_[6] + 4.*x_[4]*(y_[0] - 14.*z_[0]) + 6.*x_[2]*(y_[2] - 28.*y_[0]*z_[0] + 56.*z_[2]) + 4.*x_[0]*(y_[4] - 42.*y_[2]*z_[0] + 168.*y_[0]*z_[2] - 112.*z_[4])))*r_[9];
					case 4:
					return (-1.0031995064152008*(x_[2] - 6.*x_[0]*y_[0] + y_[2])*z*(5.*x_[4] + 5.*y_[4] - 70.*y_[2]*z_[0] + 168.*y_[0]*z_[2] - 80.*z_[4] + 5.*x_[2]*(3.*y_[0] - 14.*z_[0]) + x_[0]*(15.*y_[2] - 140.*y_[0]*z_[0] + 168.*z_[2])))*r_[9];
					case 5:
					return (-0.09479344319133458*x*(x_[2] - 10.*x_[0]*y_[0] + 5.*y_[2])*(5.*x_[4] + 5.*y_[4] - 240.*y_[2]*z_[0] + 1120.*y_[0]*z_[2] - 896.*z_[4] + 15.*x_[2]*(y_[0] - 16.*z_[0]) + 5.*x_[0]*(3.*y_[2] - 96.*y_[0]*z_[0] + 224.*z_[2])))*r_[9];
					case 6:
					return (0.3191222825450762*(x_[4] - 15.*x_[2]*y_[0] + 15.*x_[0]*y_[2] - 1.*y_[4])*z*(15.*x_[2] + 15.*y_[2] - 160.*y_[0]*z_[0] + 224.*z_[2] + 10.*x_[0]*(3.*y_[0] - 16.*z_[0])))*r_[9];
					case 7:
					return (0.504576632477118*x*(x_[4] - 21.*x_[2]*y_[0] + 35.*x_[0]*y_[2] - 7.*y_[4])*(x_[2] + y_[2] - 36.*y_[0]*z_[0] + 96.*z_[2] + 2.*x_[0]*(y_[0] - 18.*z_[0])))*r_[9];
					case 8:
					return (-4.398797100479641*(x_[6] - 28.*x_[4]*y_[0] + 70.*x_[2]*y_[2] - 28.*x_[0]*y_[4] + y_[6])*z*(x_[0] + y_[0] - 6.*z_[0]))*r_[9];
					case 9:
					return (-0.5678822637834374*x*(x_[6] - 36.*x_[4]*y_[0] + 126.*x_[2]*y_[2] - 84.*x_[0]*y_[4] + 9.*y_[6])*(x_[0] + y_[0] - 20.*z_[0]))*r_[9];
					case 10:
					return (3.680297698805311*(x_[8] - 45.*x_[6]*y_[0] + 210.*x_[4]*y_[2] - 210.*x_[2]*y_[4] + 45.*x_[0]*y_[6] - 1.*y_[8])*z)*r_[9];
					case 11:
					return (0.7846421057871968*x*(x_[8] - 55.*x_[6]*y_[0] + 330.*x_[4]*y_[2] - 462.*x_[2]*y_[4] + 165.*x_[0]*y_[6] - 11.*y_[8]))*r_[9];
					default:
					throw; return 0.0;
				}
				default:
				return 0.0;
			}
		}
		//
		// \int_0^\infty Exp[-(x-r)^2/(2 sigma^2)]*Exp[-(x-rp)^2/(2 sigmap^2)] dx
		//
		double GOverlap(double r, double rp, double sigma, double sigmap )
		{
			return (sqrt(PI/2.)*(1 + erf((rp*pow(sigma,2) + r*pow(sigmap,2))/(sqrt(2)*pow(sigma,2)*sqrt(pow(sigma,-2) + pow(sigmap,-2))*pow(sigmap,2)))))/(exp(pow(r - rp,2)/(2.*(pow(sigma,2) + pow(sigmap,2))))*sqrt(pow(sigma,-2) + pow(sigmap,-2)));
		}

		double Gau(double r, double r0, double sigma)
		{
			double toexp=pow(r - r0,2.0)/(-2.*sigma*sigma);
			if (toexp < -25.0)
			return 0.0;
			else
			return exp(toexp);
		}


		//
		// Projects a delta function at x,y,z onto
		//  Exp[-(x-r)^2/(2 sigma^2)]*Y_{LM}(theta,phi)
		//
		// which will occupy a vector of length Nrad*(1+lmax)**2

		void RadSHProjection(double x, double y, double z, double* output, double fac=1.0)
		{
			//double r = sqrt(x*x+y*y+z*z);
			double r = 1/sqrt(x*x+y*y+z*z);
			// Populate tables...
			double x_[SH_LMAX-1];
			double y_[SH_LMAX-1];
			double z_[SH_LMAX-1];
			double r_[SH_LMAX-1];

			x_[0] = x*x;
			y_[0] = y*y;
			z_[0] = z*z;
			r_[0] = r*r;

			for (int i=2; i<(SH_LMAX-1) ; i+=2)
			{
				x_[i] = x_[i-2]*x*x;
				y_[i] = y_[i-2]*y*y;
				z_[i] = z_[i-2]*z*z;
			}

			for (int i=1; i<(SH_LMAX-1) ; ++i)
				r_[i] = r_[i-1]*r;


			// for (int i=0; i<(SH_LMAX-1) ; ++i)
			// {
			// 	x_[i] = pow(x,i+2.);
			// 	y_[i] = pow(y,i+2.);
			// 	z_[i] = pow(z,i+2.);
			// 	r_[i] = pow(r,i+2.);
			// }

			//if (r<pow(10.0,-9))
			if (r>pow(10.0,9))
			return;
			// double theta = acos(z/r);
			// double theta = acos(z*r);
			// double phi = atan2(y,x);
			#pragma omp parallel for
			for (int i=0; i<SH_NRAD ; ++i)
			{
				double Gv = Gau(r, RBFS[i][0],RBFS[i][1]);
				for (int l=0; l<SH_LMAX+1 ; ++l)
				{
					for (int m=-l; m<l+1 ; ++m)
					{
						output[i*((SH_LMAX+1)*(SH_LMAX+1)) + (l)*(l) + m+l] += Gv*CartSphericalHarmonic(l,m,x,y,z,r,x_,y_,z_,r_)*fac;
						// if ((RealSphericalHarmonic(l,m,theta,phi) - CartSphericalHarmonic(l,m,x,y,z,r,x_,y_,z_,r_))>0.0000001)
						// 	cout << "Real vs. Cart: " << RealSphericalHarmonic(l,m,theta,phi) << " " << CartSphericalHarmonic(l,m,x,y,z,r,x_,y_,z_,r_) << endl;
					}
				}
			}
		}

		void RadSHProjection_Spherical(double x, double y, double z, double* output, double fac=1.0)
		{
			double r = sqrt(x*x+y*y+z*z);
			if (r<pow(10.0,-9))
			return;
			double theta = acos(z/r);
			double phi = atan2(y,x);
			int op=0;
			for (int i=0; i<SH_NRAD ; ++i)
			{
				double Gv = Gau(r, RBFS[i][0],RBFS[i][1]);
				for (int l=0; l<SH_LMAX+1 ; ++l)
				{
					for (int m=-l; m<l+1 ; ++m)
					{
						output[op] += Gv*RealSphericalHarmonic(l,m,theta,phi)*fac;
						++op;
					}
				}
			}
		}

		//
		// Projects a delta function at x,y,z onto
		//  Exp[-(x-r)^2/(2 sigma^2)]*Y_{LM}(theta,phi)
		//
		// which will occupy a vector of length Nrad*(1+lmax)**2
		void RadInvProjection(double x, double y, double z, double* output, double fac=1.0)
		{
			double r = sqrt(x*x+y*y+z*z);
			if (r<pow(10.0,-9))
			return;
			double theta = acos(z/r);
			double phi = atan2(y,x);
			int op=0;
			double tmp,tmp2;
			for (int i=0; i<SH_NRAD ; ++i)
			{
				double Gv = Gau(r, RBFS[i][0],RBFS[i][1]);
				for (int l=0; l<SH_LMAX+1 ; ++l)
				{
					tmp = 0.0;
					for (int m=-l; m<l+1 ; ++m)
					{
						tmp2 = Gv*RealSphericalHarmonic(l,m,theta,phi);
						tmp += tmp2*tmp2;
					}
					output[op] += tmp*fac;
					++op;
				}
			}
		}
