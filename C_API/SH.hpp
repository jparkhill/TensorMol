#include <math.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>
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

double CartSphericalHarmonic(int l, int m, double x, double y, double z)
{
	double r = sqrt(x*x+y*y+z*z);
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
					return (sqrt(3/PI)*y)/(2.*r);
				case 0:
					return (sqrt(3/PI)*z)/(2.*r);
				case 1:
					return (sqrt(3/PI)*x)/(2.*r);
				default:
					throw; return 0.0;
			}
		case 2:
			switch (m){
				case -2:
					return -(sqrt(15/PI)*x*y)/(2.*pow(r,2));
				case -1:
					return (sqrt(15/PI)*y*z)/(2.*pow(r,2));
				case 0:
					return -(sqrt(5/PI)*(pow(x,2) + pow(y,2) - 2*pow(z,2)))/(4.*pow(r,2));
				case 1:
					return (sqrt(15/PI)*x*z)/(2.*pow(r,2));
				case 2:
					return (sqrt(15/PI)*(pow(x,2) - pow(y,2)))/(4.*pow(r,2));
				default:
					throw; return 0.0;
			}
		case 3:
			switch (m){
				case -3:
					return -(sqrt(35/(2.*PI))*y*(-3*pow(x,2) + pow(y,2)))/(4.*pow(r,3));
				case -2:
					return -(sqrt(105/PI)*x*y*z)/(2.*pow(r,3));
				case -1:
					return -(sqrt(21/(2.*PI))*y*(pow(x,2) + pow(y,2) - 4*pow(z,2)))/(4.*pow(r,3));
				case 0:
					return (sqrt(7/PI)*z*(-3*pow(x,2) - 3*pow(y,2) + 2*pow(z,2)))/(4.*pow(r,3));
				case 1:
					return -(sqrt(21/(2.*PI))*x*(pow(x,2) + pow(y,2) - 4*pow(z,2)))/(4.*pow(r,3));
				case 2:
					return (sqrt(105/PI)*(pow(x,2) - pow(y,2))*z)/(4.*pow(r,3));
				case 3:
					return (sqrt(35/(2.*PI))*x*(pow(x,2) - 3*pow(y,2)))/(4.*pow(r,3));
				default:
					throw; return 0.0;
			}
		case 4:
			switch (m){
				case -4:
					return (3*sqrt(35/PI)*x*y*(-pow(x,2) + pow(y,2)))/(4.*pow(r,4));
				case -3:
					return (-3*sqrt(35/(2.*PI))*y*(-3*pow(x,2) + pow(y,2))*z)/(4.*pow(r,4));
				case -2:
					return (3*sqrt(5/PI)*x*y*(pow(x,2) + pow(y,2) - 6*pow(z,2)))/(4.*pow(r,4));
				case -1:
					return (-3*sqrt(5/(2.*PI))*y*z*(3*pow(x,2) + 3*pow(y,2) - 4*pow(z,2)))/(4.*pow(r,4));
				case 0:
					return (3*(3*pow(x,4) + 3*pow(y,4) - 24*pow(y,2)*pow(z,2) + 8*pow(z,4) + 6*pow(x,2)*(pow(y,2) - 4*pow(z,2))))/(16.*sqrt(PI)*pow(r,4));
				case 1:
					return (-3*sqrt(5/(2.*PI))*x*z*(3*pow(x,2) + 3*pow(y,2) - 4*pow(z,2)))/(4.*pow(r,4));
				case 2:
					return (-3*sqrt(5/PI)*(pow(x,2) - pow(y,2))*(pow(x,2) + pow(y,2) - 6*pow(z,2)))/(8.*pow(r,4));
				case 3:
					return (3*sqrt(35/(2.*PI))*x*(pow(x,2) - 3*pow(y,2))*z)/(4.*pow(r,4));
				case 4:
					return (3*sqrt(35/PI)*(pow(x,4) - 6*pow(x,2)*pow(y,2) + pow(y,4)))/(16.*pow(r,4));
				default:
					throw; return 0.0;
			}
		case 5:
			switch (m){
				case -5:
					return (3*sqrt(77/(2.*PI))*y*(5*pow(x,4) - 10*pow(x,2)*pow(y,2) + pow(y,4)))/(16.*pow(r,5));
				case -4:
					return (3*sqrt(385/PI)*x*y*(-pow(x,2) + pow(y,2))*z)/(4.*pow(r,5));
				case -3:
					return (sqrt(385/(2.*PI))*y*(-3*pow(x,2) + pow(y,2))*(pow(x,2) + pow(y,2) - 8*pow(z,2)))/(16.*pow(r,5));
				case -2:
					return (sqrt(1155/PI)*x*y*z*(pow(x,2) + pow(y,2) - 2*pow(z,2)))/(4.*pow(r,5));
				case -1:
					return (sqrt(165/PI)*y*(pow(x,4) + pow(y,4) - 12*pow(y,2)*pow(z,2) + 8*pow(z,4) + 2*pow(x,2)*(pow(y,2) - 6*pow(z,2))))/(16.*pow(r,5));
				case 0:
					return (sqrt(11/PI)*z*(15*pow(x,4) + 15*pow(y,4) - 40*pow(y,2)*pow(z,2) + 8*pow(z,4) + 10*pow(x,2)*(3*pow(y,2) - 4*pow(z,2))))/(16.*pow(r,5));
				case 1:
					return (sqrt(165/PI)*x*(pow(x,4) + pow(y,4) - 12*pow(y,2)*pow(z,2) + 8*pow(z,4) + 2*pow(x,2)*(pow(y,2) - 6*pow(z,2))))/(16.*pow(r,5));
				case 2:
					return -(sqrt(1155/PI)*(pow(x,2) - pow(y,2))*z*(pow(x,2) + pow(y,2) - 2*pow(z,2)))/(8.*pow(r,5));
				case 3:
					return -(sqrt(385/(2.*PI))*x*(pow(x,2) - 3*pow(y,2))*(pow(x,2) + pow(y,2) - 8*pow(z,2)))/(16.*pow(r,5));
				case 4:
					return (3*sqrt(385/PI)*(pow(x,4) - 6*pow(x,2)*pow(y,2) + pow(y,4))*z)/(16.*pow(r,5));
				case 5:
					return (3*sqrt(77/(2.*PI))*x*(pow(x,4) - 10*pow(x,2)*pow(y,2) + 5*pow(y,4)))/(16.*pow(r,5));
				default:
					throw; return 0.0;
			}
		case 6:
			switch (m){
				case -6:
					return -(sqrt(3003/(2.*PI))*x*y*(3*pow(x,4) - 10*pow(x,2)*pow(y,2) + 3*pow(y,4)))/(16.*pow(r,6));
				case -5:
					return (3*sqrt(1001/(2.*PI))*y*(5*pow(x,4) - 10*pow(x,2)*pow(y,2) + pow(y,4))*z)/(16.*pow(r,6));
				case -4:
					return (3*sqrt(91/PI)*x*y*(pow(x,2) - pow(y,2))*(pow(x,2) + pow(y,2) - 10*pow(z,2)))/(8.*pow(r,6));
				case -3:
					return (sqrt(1365/(2.*PI))*y*(-3*pow(x,2) + pow(y,2))*z*(3*pow(x,2) + 3*pow(y,2) - 8*pow(z,2)))/(16.*pow(r,6));
				case -2:
					return -(sqrt(1365/(2.*PI))*x*y*(pow(x,4) + pow(y,4) - 16*pow(y,2)*pow(z,2) + 16*pow(z,4) + 2*pow(x,2)*(pow(y,2) - 8*pow(z,2))))/(16.*pow(r,6));
				case -1:
					return (sqrt(273/PI)*y*z*(5*pow(x,4) + 5*pow(y,4) - 20*pow(y,2)*pow(z,2) + 8*pow(z,4) + 10*pow(x,2)*(pow(y,2) - 2*pow(z,2))))/(16.*pow(r,6));
				case 0:
					return -(sqrt(13/PI)*(5*pow(x,6) + 5*pow(y,6) - 90*pow(y,4)*pow(z,2) + 120*pow(y,2)*pow(z,4) - 16*pow(z,6) + 15*pow(x,4)*(pow(y,2) - 6*pow(z,2)) + 15*pow(x,2)*(pow(y,4) - 12*pow(y,2)*pow(z,2) + 8*pow(z,4))))/(32.*pow(r,6));
				case 1:
					return (sqrt(273/PI)*x*z*(5*pow(x,4) + 5*pow(y,4) - 20*pow(y,2)*pow(z,2) + 8*pow(z,4) + 10*pow(x,2)*(pow(y,2) - 2*pow(z,2))))/(16.*pow(r,6));
				case 2:
					return (sqrt(1365/(2.*PI))*(pow(x,2) - pow(y,2))*(pow(x,4) + pow(y,4) - 16*pow(y,2)*pow(z,2) + 16*pow(z,4) + 2*pow(x,2)*(pow(y,2) - 8*pow(z,2))))/(32.*pow(r,6));
				case 3:
					return -(sqrt(1365/(2.*PI))*x*(pow(x,2) - 3*pow(y,2))*z*(3*pow(x,2) + 3*pow(y,2) - 8*pow(z,2)))/(16.*pow(r,6));
				case 4:
					return (-3*sqrt(91/PI)*(pow(x,4) - 6*pow(x,2)*pow(y,2) + pow(y,4))*(pow(x,2) + pow(y,2) - 10*pow(z,2)))/(32.*pow(r,6));
				case 5:
					return (3*sqrt(1001/(2.*PI))*x*(pow(x,4) - 10*pow(x,2)*pow(y,2) + 5*pow(y,4))*z)/(16.*pow(r,6));
				case 6:
					return (sqrt(3003/(2.*PI))*(pow(x,6) - 15*pow(x,4)*pow(y,2) + 15*pow(x,2)*pow(y,4) - pow(y,6)))/(32.*pow(r,6));
				default:
					throw; return 0.0;
			}
		case 7:
			switch (m){
				case -7:
					return (-3*sqrt(715/PI)*y*(-7*pow(x,6) + 35*pow(x,4)*pow(y,2) - 21*pow(x,2)*pow(y,4) + pow(y,6)))/(64.*pow(r,7));
				case -6:
					return (-3*sqrt(5005/(2.*PI))*x*y*(3*pow(x,4) - 10*pow(x,2)*pow(y,2) + 3*pow(y,4))*z)/(16.*pow(r,7));
				case -5:
					return (-3*sqrt(385/PI)*y*(5*pow(x,4) - 10*pow(x,2)*pow(y,2) + pow(y,4))*(pow(x,2) + pow(y,2) - 12*pow(z,2)))/(64.*pow(r,7));
				case -4:
					return (3*sqrt(385/PI)*x*y*(pow(x,2) - pow(y,2))*z*(3*pow(x,2) + 3*pow(y,2) - 10*pow(z,2)))/(8.*pow(r,7));
				case -3:
					return (-3*sqrt(35/PI)*y*(-3*pow(x,2) + pow(y,2))*(3*pow(x,4) + 3*pow(y,4) - 60*pow(y,2)*pow(z,2) + 80*pow(z,4) + 6*pow(x,2)*(pow(y,2) - 10*pow(z,2))))/(64.*pow(r,7));
				case -2:
					return (-3*sqrt(35/(2.*PI))*x*y*z*(15*pow(x,4) + 15*pow(y,4) - 80*pow(y,2)*pow(z,2) + 48*pow(z,4) + 10*pow(x,2)*(3*pow(y,2) - 8*pow(z,2))))/(16.*pow(r,7));
				case -1:
					return -(sqrt(105/PI)*y*(5*pow(x,6) + 5*pow(y,6) - 120*pow(y,4)*pow(z,2) + 240*pow(y,2)*pow(z,4) - 64*pow(z,6) + 15*pow(x,4)*(pow(y,2) - 8*pow(z,2)) + 15*pow(x,2)*(pow(y,4) - 16*pow(y,2)*pow(z,2) + 16*pow(z,4))))/(64.*pow(r,7));
				case 0:
					return (sqrt(15/PI)*z*(-35*pow(x,6) - 35*pow(y,6) + 210*pow(y,4)*pow(z,2) - 168*pow(y,2)*pow(z,4) + 16*pow(z,6) - 105*pow(x,4)*(pow(y,2) - 2*pow(z,2)) - 21*pow(x,2)*(5*pow(y,4) - 20*pow(y,2)*pow(z,2) + 8*pow(z,4))))/(32.*pow(r,7));
				case 1:
					return -(sqrt(105/PI)*x*(5*pow(x,6) + 5*pow(y,6) - 120*pow(y,4)*pow(z,2) + 240*pow(y,2)*pow(z,4) - 64*pow(z,6) + 15*pow(x,4)*(pow(y,2) - 8*pow(z,2)) + 15*pow(x,2)*(pow(y,4) - 16*pow(y,2)*pow(z,2) + 16*pow(z,4))))/(64.*pow(r,7));
				case 2:
					return (3*sqrt(35/(2.*PI))*(pow(x,2) - pow(y,2))*z*(15*pow(x,4) + 15*pow(y,4) - 80*pow(y,2)*pow(z,2) + 48*pow(z,4) + 10*pow(x,2)*(3*pow(y,2) - 8*pow(z,2))))/(32.*pow(r,7));
				case 3:
					return (3*sqrt(35/PI)*x*(pow(x,2) - 3*pow(y,2))*(3*pow(x,4) + 3*pow(y,4) - 60*pow(y,2)*pow(z,2) + 80*pow(z,4) + 6*pow(x,2)*(pow(y,2) - 10*pow(z,2))))/(64.*pow(r,7));
				case 4:
					return (-3*sqrt(385/PI)*(pow(x,4) - 6*pow(x,2)*pow(y,2) + pow(y,4))*z*(3*pow(x,2) + 3*pow(y,2) - 10*pow(z,2)))/(32.*pow(r,7));
				case 5:
					return (-3*sqrt(385/PI)*x*(pow(x,4) - 10*pow(x,2)*pow(y,2) + 5*pow(y,4))*(pow(x,2) + pow(y,2) - 12*pow(z,2)))/(64.*pow(r,7));
				case 6:
					return (3*sqrt(5005/(2.*PI))*(pow(x,6) - 15*pow(x,4)*pow(y,2) + 15*pow(x,2)*pow(y,4) - pow(y,6))*z)/(32.*pow(r,7));
				case 7:
					return (3*sqrt(715/PI)*x*(pow(x,6) - 21*pow(x,4)*pow(y,2) + 35*pow(x,2)*pow(y,4) - 7*pow(y,6)))/(64.*pow(r,7));
				default:
					throw; return 0.0;
			}
		case 8:
			switch (m){
				case -8:
					return (-3*sqrt(12155/PI)*x*y*(pow(x,6) - 7*pow(x,4)*pow(y,2) + 7*pow(x,2)*pow(y,4) - pow(y,6)))/(32.*pow(r,8));
				case -7:
					return (-3*sqrt(12155/PI)*y*(-7*pow(x,6) + 35*pow(x,4)*pow(y,2) - 21*pow(x,2)*pow(y,4) + pow(y,6))*z)/(64.*pow(r,8));
				case -6:
					return (sqrt(7293/(2.*PI))*x*y*(3*pow(x,4) - 10*pow(x,2)*pow(y,2) + 3*pow(y,4))*(pow(x,2) + pow(y,2) - 14*pow(z,2)))/(32.*pow(r,8));
				case -5:
					return (-3*sqrt(17017/PI)*y*(5*pow(x,4) - 10*pow(x,2)*pow(y,2) + pow(y,4))*z*(pow(x,2) + pow(y,2) - 4*pow(z,2)))/(64.*pow(r,8));
				case -4:
					return (-3*sqrt(1309/PI)*x*y*(pow(x,2) - pow(y,2))*(pow(x,4) + pow(y,4) - 24*pow(y,2)*pow(z,2) + 40*pow(z,4) + 2*pow(x,2)*(pow(y,2) - 12*pow(z,2))))/(32.*pow(r,8));
				case -3:
					return -(sqrt(19635/PI)*y*(-3*pow(x,2) + pow(y,2))*z*(3*pow(x,4) + 3*pow(y,4) - 20*pow(y,2)*pow(z,2) + 16*pow(z,4) + pow(x,2)*(6*pow(y,2) - 20*pow(z,2))))/(64.*pow(r,8));
				case -2:
					return (3*sqrt(595/(2.*PI))*x*y*(pow(x,6) + pow(y,6) - 30*pow(y,4)*pow(z,2) + 80*pow(y,2)*pow(z,4) - 32*pow(z,6) + 3*pow(x,4)*(pow(y,2) - 10*pow(z,2)) + pow(x,2)*(3*pow(y,4) - 60*pow(y,2)*pow(z,2) + 80*pow(z,4))))/(32.*pow(r,8));
				case -1:
					return (-3*sqrt(17/PI)*y*z*(35*pow(x,6) + 35*pow(y,6) - 280*pow(y,4)*pow(z,2) + 336*pow(y,2)*pow(z,4) - 64*pow(z,6) + 35*pow(x,4)*(3*pow(y,2) - 8*pow(z,2)) + 7*pow(x,2)*(15*pow(y,4) - 80*pow(y,2)*pow(z,2) + 48*pow(z,4))))/(64.*pow(r,8));
				case 0:
					return (sqrt(17/PI)*(35*pow(x,8) + 35*pow(y,8) - 1120*pow(y,6)*pow(z,2) + 3360*pow(y,4)*pow(z,4) - 1792*pow(y,2)*pow(z,6) + 128*pow(z,8) + 140*pow(x,6)*(pow(y,2) - 8*pow(z,2)) + 210*pow(x,4)*(pow(y,4) - 16*pow(y,2)*pow(z,2) + 16*pow(z,4)) + 28*pow(x,2)*(5*pow(y,6) - 120*pow(y,4)*pow(z,2) + 240*pow(y,2)*pow(z,4) - 64*pow(z,6))))/(256.*pow(r,8));
				case 1:
					return (-3*sqrt(17/PI)*x*z*(35*pow(x,6) + 35*pow(y,6) - 280*pow(y,4)*pow(z,2) + 336*pow(y,2)*pow(z,4) - 64*pow(z,6) + 35*pow(x,4)*(3*pow(y,2) - 8*pow(z,2)) + 7*pow(x,2)*(15*pow(y,4) - 80*pow(y,2)*pow(z,2) + 48*pow(z,4))))/(64.*pow(r,8));
				case 2:
					return (-3*sqrt(595/(2.*PI))*(pow(x,2) - pow(y,2))*(pow(x,6) + pow(y,6) - 30*pow(y,4)*pow(z,2) + 80*pow(y,2)*pow(z,4) - 32*pow(z,6) + 3*pow(x,4)*(pow(y,2) - 10*pow(z,2)) + pow(x,2)*(3*pow(y,4) - 60*pow(y,2)*pow(z,2) + 80*pow(z,4))))/(64.*pow(r,8));
				case 3:
					return (sqrt(19635/PI)*x*(pow(x,2) - 3*pow(y,2))*z*(3*pow(x,4) + 3*pow(y,4) - 20*pow(y,2)*pow(z,2) + 16*pow(z,4) + pow(x,2)*(6*pow(y,2) - 20*pow(z,2))))/(64.*pow(r,8));
				case 4:
					return (3*sqrt(1309/PI)*(pow(x,4) - 6*pow(x,2)*pow(y,2) + pow(y,4))*(pow(x,4) + pow(y,4) - 24*pow(y,2)*pow(z,2) + 40*pow(z,4) + 2*pow(x,2)*(pow(y,2) - 12*pow(z,2))))/(128.*pow(r,8));
				case 5:
					return (-3*sqrt(17017/PI)*x*(pow(x,4) - 10*pow(x,2)*pow(y,2) + 5*pow(y,4))*z*(pow(x,2) + pow(y,2) - 4*pow(z,2)))/(64.*pow(r,8));
				case 6:
					return -(sqrt(7293/(2.*PI))*(pow(x,6) - 15*pow(x,4)*pow(y,2) + 15*pow(x,2)*pow(y,4) - pow(y,6))*(pow(x,2) + pow(y,2) - 14*pow(z,2)))/(64.*pow(r,8));
				case 7:
					return (3*sqrt(12155/PI)*x*(pow(x,6) - 21*pow(x,4)*pow(y,2) + 35*pow(x,2)*pow(y,4) - 7*pow(y,6))*z)/(64.*pow(r,8));
				case 8:
					return (3*sqrt(12155/PI)*(pow(x,8) - 28*pow(x,6)*pow(y,2) + 70*pow(x,4)*pow(y,4) - 28*pow(x,2)*pow(y,6) + pow(y,8)))/(256.*pow(r,8));
				default:
					throw; return 0.0;
			}
		case 9:
			switch (m){
				case -9:
					return (sqrt(230945/(2.*PI))*y*(9*pow(x,8) - 84*pow(x,6)*pow(y,2) + 126*pow(x,4)*pow(y,4) - 36*pow(x,2)*pow(y,6) + pow(y,8)))/(256.*pow(r,9));
				case -8:
					return (-3*sqrt(230945/PI)*x*y*(pow(x,6) - 7*pow(x,4)*pow(y,2) + 7*pow(x,2)*pow(y,4) - pow(y,6))*z)/(32.*pow(r,9));
				case -7:
					return (3*sqrt(13585/(2.*PI))*y*(-7*pow(x,6) + 35*pow(x,4)*pow(y,2) - 21*pow(x,2)*pow(y,4) + pow(y,6))*(pow(x,2) + pow(y,2) - 16*pow(z,2)))/(256.*pow(r,9));
				case -6:
					return (sqrt(40755/(2.*PI))*x*y*(3*pow(x,4) - 10*pow(x,2)*pow(y,2) + 3*pow(y,4))*z*(3*pow(x,2) + 3*pow(y,2) - 14*pow(z,2)))/(32.*pow(r,9));
				case -5:
					return (3*sqrt(2717/(2.*PI))*y*(5*pow(x,4) - 10*pow(x,2)*pow(y,2) + pow(y,4))*(pow(x,4) + pow(y,4) - 28*pow(y,2)*pow(z,2) + 56*pow(z,4) + 2*pow(x,2)*(pow(y,2) - 14*pow(z,2))))/(128.*pow(r,9));
				case -4:
					return (-3*sqrt(95095/PI)*x*y*(pow(x,2) - pow(y,2))*z*(pow(x,4) + pow(y,4) - 8*pow(y,2)*pow(z,2) + 8*pow(z,4) + 2*pow(x,2)*(pow(y,2) - 4*pow(z,2))))/(32.*pow(r,9));
				case -3:
					return (sqrt(21945/(2.*PI))*y*(-3*pow(x,2) + pow(y,2))*(pow(x,6) + pow(y,6) - 36*pow(y,4)*pow(z,2) + 120*pow(y,2)*pow(z,4) - 64*pow(z,6) + 3*pow(x,4)*(pow(y,2) - 12*pow(z,2)) + 3*pow(x,2)*(pow(y,4) - 24*pow(y,2)*pow(z,2) + 40*pow(z,4))))/(128.*pow(r,9));
				case -2:
					return (3*sqrt(1045/(2.*PI))*x*y*z*(7*pow(x,6) + 7*pow(y,6) - 70*pow(y,4)*pow(z,2) + 112*pow(y,2)*pow(z,4) - 32*pow(z,6) + 7*pow(x,4)*(3*pow(y,2) - 10*pow(z,2)) + 7*pow(x,2)*(3*pow(y,4) - 20*pow(y,2)*pow(z,2) + 16*pow(z,4))))/(32.*pow(r,9));
				case -1:
					return (3*sqrt(95/PI)*y*(7*pow(x,8) + 7*pow(y,8) - 280*pow(y,6)*pow(z,2) + 1120*pow(y,4)*pow(z,4) - 896*pow(y,2)*pow(z,6) + 128*pow(z,8) + 28*pow(x,6)*(pow(y,2) - 10*pow(z,2)) + 14*pow(x,4)*(3*pow(y,4) - 60*pow(y,2)*pow(z,2) + 80*pow(z,4)) + 28*pow(x,2)*(pow(y,6) - 30*pow(y,4)*pow(z,2) + 80*pow(y,2)*pow(z,4) - 32*pow(z,6))))/(256.*pow(r,9));
				case 0:
					return (sqrt(19/PI)*z*(315*pow(x,8) + 315*pow(y,8) - 3360*pow(y,6)*pow(z,2) + 6048*pow(y,4)*pow(z,4) - 2304*pow(y,2)*pow(z,6) + 128*pow(z,8) + 420*pow(x,6)*(3*pow(y,2) - 8*pow(z,2)) + 126*pow(x,4)*(15*pow(y,4) - 80*pow(y,2)*pow(z,2) + 48*pow(z,4)) + 36*pow(x,2)*(35*pow(y,6) - 280*pow(y,4)*pow(z,2) + 336*pow(y,2)*pow(z,4) - 64*pow(z,6))))/(256.*pow(r,9));
				case 1:
					return (3*sqrt(95/PI)*x*(7*pow(x,8) + 7*pow(y,8) - 280*pow(y,6)*pow(z,2) + 1120*pow(y,4)*pow(z,4) - 896*pow(y,2)*pow(z,6) + 128*pow(z,8) + 28*pow(x,6)*(pow(y,2) - 10*pow(z,2)) + 14*pow(x,4)*(3*pow(y,4) - 60*pow(y,2)*pow(z,2) + 80*pow(z,4)) + 28*pow(x,2)*(pow(y,6) - 30*pow(y,4)*pow(z,2) + 80*pow(y,2)*pow(z,4) - 32*pow(z,6))))/(256.*pow(r,9));
				case 2:
					return (-3*sqrt(1045/(2.*PI))*(pow(x,2) - pow(y,2))*z*(7*pow(x,6) + 7*pow(y,6) - 70*pow(y,4)*pow(z,2) + 112*pow(y,2)*pow(z,4) - 32*pow(z,6) + 7*pow(x,4)*(3*pow(y,2) - 10*pow(z,2)) + 7*pow(x,2)*(3*pow(y,4) - 20*pow(y,2)*pow(z,2) + 16*pow(z,4))))/(64.*pow(r,9));
				case 3:
					return -(sqrt(21945/(2.*PI))*x*(pow(x,2) - 3*pow(y,2))*(pow(x,6) + pow(y,6) - 36*pow(y,4)*pow(z,2) + 120*pow(y,2)*pow(z,4) - 64*pow(z,6) + 3*pow(x,4)*(pow(y,2) - 12*pow(z,2)) + 3*pow(x,2)*(pow(y,4) - 24*pow(y,2)*pow(z,2) + 40*pow(z,4))))/(128.*pow(r,9));
				case 4:
					return (3*sqrt(95095/PI)*(pow(x,4) - 6*pow(x,2)*pow(y,2) + pow(y,4))*z*(pow(x,4) + pow(y,4) - 8*pow(y,2)*pow(z,2) + 8*pow(z,4) + 2*pow(x,2)*(pow(y,2) - 4*pow(z,2))))/(128.*pow(r,9));
				case 5:
					return (3*sqrt(2717/(2.*PI))*x*(pow(x,4) - 10*pow(x,2)*pow(y,2) + 5*pow(y,4))*(pow(x,4) + pow(y,4) - 28*pow(y,2)*pow(z,2) + 56*pow(z,4) + 2*pow(x,2)*(pow(y,2) - 14*pow(z,2))))/(128.*pow(r,9));
				case 6:
					return -(sqrt(40755/(2.*PI))*(pow(x,6) - 15*pow(x,4)*pow(y,2) + 15*pow(x,2)*pow(y,4) - pow(y,6))*z*(3*pow(x,2) + 3*pow(y,2) - 14*pow(z,2)))/(64.*pow(r,9));
				case 7:
					return (-3*sqrt(13585/(2.*PI))*x*(pow(x,6) - 21*pow(x,4)*pow(y,2) + 35*pow(x,2)*pow(y,4) - 7*pow(y,6))*(pow(x,2) + pow(y,2) - 16*pow(z,2)))/(256.*pow(r,9));
				case 8:
					return (3*sqrt(230945/PI)*(pow(x,8) - 28*pow(x,6)*pow(y,2) + 70*pow(x,4)*pow(y,4) - 28*pow(x,2)*pow(y,6) + pow(y,8))*z)/(256.*pow(r,9));
				case 9:
					return (sqrt(230945/(2.*PI))*x*(pow(x,8) - 36*pow(x,6)*pow(y,2) + 126*pow(x,4)*pow(y,4) - 84*pow(x,2)*pow(y,6) + 9*pow(y,8)))/(256.*pow(r,9));
				default:
					throw; return 0.0;
			}
		case 10:
			switch (m){
				case -10:
					return -(sqrt(969969/(2.*PI))*x*y*(5*pow(x,8) - 60*pow(x,6)*pow(y,2) + 126*pow(x,4)*pow(y,4) - 60*pow(x,2)*pow(y,6) + 5*pow(y,8)))/(256.*pow(r,10));
				case -9:
					return (sqrt(4849845/(2.*PI))*y*(9*pow(x,8) - 84*pow(x,6)*pow(y,2) + 126*pow(x,4)*pow(y,4) - 36*pow(x,2)*pow(y,6) + pow(y,8))*z)/(256.*pow(r,10));
				case -8:
					return (sqrt(255255/PI)*x*y*(pow(x,6) - 7*pow(x,4)*pow(y,2) + 7*pow(x,2)*pow(y,4) - pow(y,6))*(pow(x,2) + pow(y,2) - 18*pow(z,2)))/(64.*pow(r,10));
				case -7:
					return (3*sqrt(85085/(2.*PI))*y*(-7*pow(x,6) + 35*pow(x,4)*pow(y,2) - 21*pow(x,2)*pow(y,4) + pow(y,6))*z*(3*pow(x,2) + 3*pow(y,2) - 16*pow(z,2)))/(256.*pow(r,10));
				case -6:
					return (-3*sqrt(5005/(2.*PI))*x*y*(3*pow(x,4) - 10*pow(x,2)*pow(y,2) + 3*pow(y,4))*(3*pow(x,4) + 3*pow(y,4) - 96*pow(y,2)*pow(z,2) + 224*pow(z,4) + 6*pow(x,2)*(pow(y,2) - 16*pow(z,2))))/(256.*pow(r,10));
				case -5:
					return (3*sqrt(1001/(2.*PI))*y*(5*pow(x,4) - 10*pow(x,2)*pow(y,2) + pow(y,4))*z*(15*pow(x,4) + 15*pow(y,4) - 140*pow(y,2)*pow(z,2) + 168*pow(z,4) + 10*pow(x,2)*(3*pow(y,2) - 14*pow(z,2))))/(128.*pow(r,10));
				case -4:
					return (3*sqrt(5005/PI)*x*y*(pow(x,2) - pow(y,2))*(pow(x,6) + pow(y,6) - 42*pow(y,4)*pow(z,2) + 168*pow(y,2)*pow(z,4) - 112*pow(z,6) + 3*pow(x,4)*(pow(y,2) - 14*pow(z,2)) + 3*pow(x,2)*(pow(y,4) - 28*pow(y,2)*pow(z,2) + 56*pow(z,4))))/(64.*pow(r,10));
				case -3:
					return (3*sqrt(5005/(2.*PI))*y*(-3*pow(x,2) + pow(y,2))*z*(7*pow(x,6) + 7*pow(y,6) - 84*pow(y,4)*pow(z,2) + 168*pow(y,2)*pow(z,4) - 64*pow(z,6) + 21*pow(x,4)*(pow(y,2) - 4*pow(z,2)) + 21*pow(x,2)*(pow(y,4) - 8*pow(y,2)*pow(z,2) + 8*pow(z,4))))/(128.*pow(r,10));
				case -2:
					return (-3*sqrt(385/PI)*x*y*(7*pow(x,8) + 7*pow(y,8) - 336*pow(y,6)*pow(z,2) + 1680*pow(y,4)*pow(z,4) - 1792*pow(y,2)*pow(z,6) + 384*pow(z,8) + 28*pow(x,6)*(pow(y,2) - 12*pow(z,2)) + 42*pow(x,4)*(pow(y,4) - 24*pow(y,2)*pow(z,2) + 40*pow(z,4)) + 28*pow(x,2)*(pow(y,6) - 36*pow(y,4)*pow(z,2) + 120*pow(y,2)*pow(z,4) - 64*pow(z,6))))/(256.*pow(r,10));
				case -1:
					return (sqrt(1155/PI)*y*z*(63*pow(x,8) + 63*pow(y,8) - 840*pow(y,6)*pow(z,2) + 2016*pow(y,4)*pow(z,4) - 1152*pow(y,2)*pow(z,6) + 128*pow(z,8) + 84*pow(x,6)*(3*pow(y,2) - 10*pow(z,2)) + 126*pow(x,4)*(3*pow(y,4) - 20*pow(y,2)*pow(z,2) + 16*pow(z,4)) + 36*pow(x,2)*(7*pow(y,6) - 70*pow(y,4)*pow(z,2) + 112*pow(y,2)*pow(z,4) - 32*pow(z,6))))/(256.*pow(r,10));
				case 0:
					return -(sqrt(21/PI)*(63*pow(x,10) + 63*pow(y,10) - 3150*pow(y,8)*pow(z,2) + 16800*pow(y,6)*pow(z,4) - 20160*pow(y,4)*pow(z,6) + 5760*pow(y,2)*pow(z,8) - 256*pow(z,10) + 315*pow(x,8)*(pow(y,2) - 10*pow(z,2)) + 210*pow(x,6)*(3*pow(y,4) - 60*pow(y,2)*pow(z,2) + 80*pow(z,4)) + 630*pow(x,4)*(pow(y,6) - 30*pow(y,4)*pow(z,2) + 80*pow(y,2)*pow(z,4) - 32*pow(z,6)) + 45*pow(x,2)*(7*pow(y,8) - 280*pow(y,6)*pow(z,2) + 1120*pow(y,4)*pow(z,4) - 896*pow(y,2)*pow(z,6) + 128*pow(z,8))))/(512.*pow(r,10));
				case 1:
					return (sqrt(1155/PI)*x*z*(63*pow(x,8) + 63*pow(y,8) - 840*pow(y,6)*pow(z,2) + 2016*pow(y,4)*pow(z,4) - 1152*pow(y,2)*pow(z,6) + 128*pow(z,8) + 84*pow(x,6)*(3*pow(y,2) - 10*pow(z,2)) + 126*pow(x,4)*(3*pow(y,4) - 20*pow(y,2)*pow(z,2) + 16*pow(z,4)) + 36*pow(x,2)*(7*pow(y,6) - 70*pow(y,4)*pow(z,2) + 112*pow(y,2)*pow(z,4) - 32*pow(z,6))))/(256.*pow(r,10));
				case 2:
					return (3*sqrt(385/PI)*(pow(x,2) - pow(y,2))*(7*pow(x,8) + 7*pow(y,8) - 336*pow(y,6)*pow(z,2) + 1680*pow(y,4)*pow(z,4) - 1792*pow(y,2)*pow(z,6) + 384*pow(z,8) + 28*pow(x,6)*(pow(y,2) - 12*pow(z,2)) + 42*pow(x,4)*(pow(y,4) - 24*pow(y,2)*pow(z,2) + 40*pow(z,4)) + 28*pow(x,2)*(pow(y,6) - 36*pow(y,4)*pow(z,2) + 120*pow(y,2)*pow(z,4) - 64*pow(z,6))))/(512.*pow(r,10));
				case 3:
					return (-3*sqrt(5005/(2.*PI))*x*(pow(x,2) - 3*pow(y,2))*z*(7*pow(x,6) + 7*pow(y,6) - 84*pow(y,4)*pow(z,2) + 168*pow(y,2)*pow(z,4) - 64*pow(z,6) + 21*pow(x,4)*(pow(y,2) - 4*pow(z,2)) + 21*pow(x,2)*(pow(y,4) - 8*pow(y,2)*pow(z,2) + 8*pow(z,4))))/(128.*pow(r,10));
				case 4:
					return (-3*sqrt(5005/PI)*(pow(x,4) - 6*pow(x,2)*pow(y,2) + pow(y,4))*(pow(x,6) + pow(y,6) - 42*pow(y,4)*pow(z,2) + 168*pow(y,2)*pow(z,4) - 112*pow(z,6) + 3*pow(x,4)*(pow(y,2) - 14*pow(z,2)) + 3*pow(x,2)*(pow(y,4) - 28*pow(y,2)*pow(z,2) + 56*pow(z,4))))/(256.*pow(r,10));
				case 5:
					return (3*sqrt(1001/(2.*PI))*x*(pow(x,4) - 10*pow(x,2)*pow(y,2) + 5*pow(y,4))*z*(15*pow(x,4) + 15*pow(y,4) - 140*pow(y,2)*pow(z,2) + 168*pow(z,4) + 10*pow(x,2)*(3*pow(y,2) - 14*pow(z,2))))/(128.*pow(r,10));
				case 6:
					return (3*sqrt(5005/(2.*PI))*(pow(x,6) - 15*pow(x,4)*pow(y,2) + 15*pow(x,2)*pow(y,4) - pow(y,6))*(3*pow(x,4) + 3*pow(y,4) - 96*pow(y,2)*pow(z,2) + 224*pow(z,4) + 6*pow(x,2)*(pow(y,2) - 16*pow(z,2))))/(512.*pow(r,10));
				case 7:
					return (-3*sqrt(85085/(2.*PI))*x*(pow(x,6) - 21*pow(x,4)*pow(y,2) + 35*pow(x,2)*pow(y,4) - 7*pow(y,6))*z*(3*pow(x,2) + 3*pow(y,2) - 16*pow(z,2)))/(256.*pow(r,10));
				case 8:
					return -(sqrt(255255/PI)*(pow(x,8) - 28*pow(x,6)*pow(y,2) + 70*pow(x,4)*pow(y,4) - 28*pow(x,2)*pow(y,6) + pow(y,8))*(pow(x,2) + pow(y,2) - 18*pow(z,2)))/(512.*pow(r,10));
				case 9:
					return (sqrt(4849845/(2.*PI))*x*(pow(x,8) - 36*pow(x,6)*pow(y,2) + 126*pow(x,4)*pow(y,4) - 84*pow(x,2)*pow(y,6) + 9*pow(y,8))*z)/(256.*pow(r,10));
				case 10:
					return (sqrt(969969/(2.*PI))*(pow(x,10) - 45*pow(x,8)*pow(y,2) + 210*pow(x,6)*pow(y,4) - 210*pow(x,4)*pow(y,6) + 45*pow(x,2)*pow(y,8) - pow(y,10)))/(512.*pow(r,10));
				default:
					throw; return 0.0;
			}
		case 11:
			switch (m){
				case -11:
					return -(sqrt(2028117/PI)*y*(-11*pow(x,10) + 165*pow(x,8)*pow(y,2) - 462*pow(x,6)*pow(y,4) + 330*pow(x,4)*pow(y,6) - 55*pow(x,2)*pow(y,8) + pow(y,10)))/(1024.*pow(r,11));
				case -10:
					return -(sqrt(22309287/(2.*PI))*x*y*(5*pow(x,8) - 60*pow(x,6)*pow(y,2) + 126*pow(x,4)*pow(y,4) - 60*pow(x,2)*pow(y,6) + 5*pow(y,8))*z)/(256.*pow(r,11));
				case -9:
					return -(sqrt(1062347/PI)*y*(9*pow(x,8) - 84*pow(x,6)*pow(y,2) + 126*pow(x,4)*pow(y,4) - 36*pow(x,2)*pow(y,6) + pow(y,8))*(pow(x,2) + pow(y,2) - 20*pow(z,2)))/(1024.*pow(r,11));
				case -8:
					return (sqrt(15935205/PI)*x*y*(pow(x,6) - 7*pow(x,4)*pow(y,2) + 7*pow(x,2)*pow(y,4) - pow(y,6))*z*(pow(x,2) + pow(y,2) - 6*pow(z,2)))/(64.*pow(r,11));
				case -7:
					return -(sqrt(838695/PI)*y*(-7*pow(x,6) + 35*pow(x,4)*pow(y,2) - 21*pow(x,2)*pow(y,4) + pow(y,6))*(pow(x,4) + pow(y,4) - 36*pow(y,2)*pow(z,2) + 96*pow(z,4) + 2*pow(x,2)*(pow(y,2) - 18*pow(z,2))))/(1024.*pow(r,11));
				case -6:
					return -(sqrt(167739/(2.*PI))*x*y*(3*pow(x,4) - 10*pow(x,2)*pow(y,2) + 3*pow(y,4))*z*(15*pow(x,4) + 15*pow(y,4) - 160*pow(y,2)*pow(z,2) + 224*pow(z,4) + 10*pow(x,2)*(3*pow(y,2) - 16*pow(z,2))))/(256.*pow(r,11));
				case -5:
					return (-3*sqrt(3289/PI)*y*(5*pow(x,4) - 10*pow(x,2)*pow(y,2) + pow(y,4))*(5*pow(x,6) + 5*pow(y,6) - 240*pow(y,4)*pow(z,2) + 1120*pow(y,2)*pow(z,4) - 896*pow(z,6) + 15*pow(x,4)*(pow(y,2) - 16*pow(z,2)) + 5*pow(x,2)*(3*pow(y,4) - 96*pow(y,2)*pow(z,2) + 224*pow(z,4))))/(1024.*pow(r,11));
				case -4:
					return (3*sqrt(23023/PI)*x*y*(pow(x,2) - pow(y,2))*z*(5*pow(x,6) + 5*pow(y,6) - 70*pow(y,4)*pow(z,2) + 168*pow(y,2)*pow(z,4) - 80*pow(z,6) + 5*pow(x,4)*(3*pow(y,2) - 14*pow(z,2)) + pow(x,2)*(15*pow(y,4) - 140*pow(y,2)*pow(z,2) + 168*pow(z,4))))/(64.*pow(r,11));
				case -3:
					return -(sqrt(345345/(2.*PI))*y*(-3*pow(x,2) + pow(y,2))*(pow(x,8) + pow(y,8) - 56*pow(y,6)*pow(z,2) + 336*pow(y,4)*pow(z,4) - 448*pow(y,2)*pow(z,6) + 128*pow(z,8) + 4*pow(x,6)*(pow(y,2) - 14*pow(z,2)) + 6*pow(x,4)*(pow(y,4) - 28*pow(y,2)*pow(z,2) + 56*pow(z,4)) + 4*pow(x,2)*(pow(y,6) - 42*pow(y,4)*pow(z,2) + 168*pow(y,2)*pow(z,4) - 112*pow(z,6))))/(512.*pow(r,11));
				case -2:
					return -(sqrt(49335/PI)*x*y*z*(21*pow(x,8) + 21*pow(y,8) - 336*pow(y,6)*pow(z,2) + 1008*pow(y,4)*pow(z,4) - 768*pow(y,2)*pow(z,6) + 128*pow(z,8) + 84*pow(x,6)*(pow(y,2) - 4*pow(z,2)) + 126*pow(x,4)*(pow(y,4) - 8*pow(y,2)*pow(z,2) + 8*pow(z,4)) + 12*pow(x,2)*(7*pow(y,6) - 84*pow(y,4)*pow(z,2) + 168*pow(y,2)*pow(z,4) - 64*pow(z,6))))/(256.*pow(r,11));
				case -1:
					return -(sqrt(759/(2.*PI))*y*(21*pow(x,10) + 21*pow(y,10) - 1260*pow(y,8)*pow(z,2) + 8400*pow(y,6)*pow(z,4) - 13440*pow(y,4)*pow(z,6) + 5760*pow(y,2)*pow(z,8) - 512*pow(z,10) + 105*pow(x,8)*(pow(y,2) - 12*pow(z,2)) + 210*pow(x,6)*(pow(y,4) - 24*pow(y,2)*pow(z,2) + 40*pow(z,4)) + 210*pow(x,4)*(pow(y,6) - 36*pow(y,4)*pow(z,2) + 120*pow(y,2)*pow(z,4) - 64*pow(z,6)) + 15*pow(x,2)*(7*pow(y,8) - 336*pow(y,6)*pow(z,2) + 1680*pow(y,4)*pow(z,4) - 1792*pow(y,2)*pow(z,6) + 384*pow(z,8))))/(512.*pow(r,11));
				case 0:
					return (sqrt(23/PI)*z*(-693*pow(x,10) - 693*pow(y,10) + 11550*pow(y,8)*pow(z,2) - 36960*pow(y,6)*pow(z,4) + 31680*pow(y,4)*pow(z,6) - 7040*pow(y,2)*pow(z,8) + 256*pow(z,10) - 1155*pow(x,8)*(3*pow(y,2) - 10*pow(z,2)) - 2310*pow(x,6)*(3*pow(y,4) - 20*pow(y,2)*pow(z,2) + 16*pow(z,4)) - 990*pow(x,4)*(7*pow(y,6) - 70*pow(y,4)*pow(z,2) + 112*pow(y,2)*pow(z,4) - 32*pow(z,6)) - 55*pow(x,2)*(63*pow(y,8) - 840*pow(y,6)*pow(z,2) + 2016*pow(y,4)*pow(z,4) - 1152*pow(y,2)*pow(z,6) + 128*pow(z,8))))/(512.*pow(r,11));
				case 1:
					return -(sqrt(759/(2.*PI))*x*(21*pow(x,10) + 21*pow(y,10) - 1260*pow(y,8)*pow(z,2) + 8400*pow(y,6)*pow(z,4) - 13440*pow(y,4)*pow(z,6) + 5760*pow(y,2)*pow(z,8) - 512*pow(z,10) + 105*pow(x,8)*(pow(y,2) - 12*pow(z,2)) + 210*pow(x,6)*(pow(y,4) - 24*pow(y,2)*pow(z,2) + 40*pow(z,4)) + 210*pow(x,4)*(pow(y,6) - 36*pow(y,4)*pow(z,2) + 120*pow(y,2)*pow(z,4) - 64*pow(z,6)) + 15*pow(x,2)*(7*pow(y,8) - 336*pow(y,6)*pow(z,2) + 1680*pow(y,4)*pow(z,4) - 1792*pow(y,2)*pow(z,6) + 384*pow(z,8))))/(512.*pow(r,11));
				case 2:
					return (sqrt(49335/PI)*(pow(x,2) - pow(y,2))*z*(21*pow(x,8) + 21*pow(y,8) - 336*pow(y,6)*pow(z,2) + 1008*pow(y,4)*pow(z,4) - 768*pow(y,2)*pow(z,6) + 128*pow(z,8) + 84*pow(x,6)*(pow(y,2) - 4*pow(z,2)) + 126*pow(x,4)*(pow(y,4) - 8*pow(y,2)*pow(z,2) + 8*pow(z,4)) + 12*pow(x,2)*(7*pow(y,6) - 84*pow(y,4)*pow(z,2) + 168*pow(y,2)*pow(z,4) - 64*pow(z,6))))/(512.*pow(r,11));
				case 3:
					return (sqrt(345345/(2.*PI))*x*(pow(x,2) - 3*pow(y,2))*(pow(x,8) + pow(y,8) - 56*pow(y,6)*pow(z,2) + 336*pow(y,4)*pow(z,4) - 448*pow(y,2)*pow(z,6) + 128*pow(z,8) + 4*pow(x,6)*(pow(y,2) - 14*pow(z,2)) + 6*pow(x,4)*(pow(y,4) - 28*pow(y,2)*pow(z,2) + 56*pow(z,4)) + 4*pow(x,2)*(pow(y,6) - 42*pow(y,4)*pow(z,2) + 168*pow(y,2)*pow(z,4) - 112*pow(z,6))))/(512.*pow(r,11));
				case 4:
					return (-3*sqrt(23023/PI)*(pow(x,4) - 6*pow(x,2)*pow(y,2) + pow(y,4))*z*(5*pow(x,6) + 5*pow(y,6) - 70*pow(y,4)*pow(z,2) + 168*pow(y,2)*pow(z,4) - 80*pow(z,6) + 5*pow(x,4)*(3*pow(y,2) - 14*pow(z,2)) + pow(x,2)*(15*pow(y,4) - 140*pow(y,2)*pow(z,2) + 168*pow(z,4))))/(256.*pow(r,11));
				case 5:
					return (-3*sqrt(3289/PI)*x*(pow(x,4) - 10*pow(x,2)*pow(y,2) + 5*pow(y,4))*(5*pow(x,6) + 5*pow(y,6) - 240*pow(y,4)*pow(z,2) + 1120*pow(y,2)*pow(z,4) - 896*pow(z,6) + 15*pow(x,4)*(pow(y,2) - 16*pow(z,2)) + 5*pow(x,2)*(3*pow(y,4) - 96*pow(y,2)*pow(z,2) + 224*pow(z,4))))/(1024.*pow(r,11));
				case 6:
					return (sqrt(167739/(2.*PI))*(pow(x,6) - 15*pow(x,4)*pow(y,2) + 15*pow(x,2)*pow(y,4) - pow(y,6))*z*(15*pow(x,4) + 15*pow(y,4) - 160*pow(y,2)*pow(z,2) + 224*pow(z,4) + 10*pow(x,2)*(3*pow(y,2) - 16*pow(z,2))))/(512.*pow(r,11));
				case 7:
					return (sqrt(838695/PI)*x*(pow(x,6) - 21*pow(x,4)*pow(y,2) + 35*pow(x,2)*pow(y,4) - 7*pow(y,6))*(pow(x,4) + pow(y,4) - 36*pow(y,2)*pow(z,2) + 96*pow(z,4) + 2*pow(x,2)*(pow(y,2) - 18*pow(z,2))))/(1024.*pow(r,11));
				case 8:
					return -(sqrt(15935205/PI)*(pow(x,8) - 28*pow(x,6)*pow(y,2) + 70*pow(x,4)*pow(y,4) - 28*pow(x,2)*pow(y,6) + pow(y,8))*z*(pow(x,2) + pow(y,2) - 6*pow(z,2)))/(512.*pow(r,11));
				case 9:
					return -(sqrt(1062347/PI)*x*(pow(x,8) - 36*pow(x,6)*pow(y,2) + 126*pow(x,4)*pow(y,4) - 84*pow(x,2)*pow(y,6) + 9*pow(y,8))*(pow(x,2) + pow(y,2) - 20*pow(z,2)))/(1024.*pow(r,11));
				case 10:
					return (sqrt(22309287/(2.*PI))*(pow(x,10) - 45*pow(x,8)*pow(y,2) + 210*pow(x,6)*pow(y,4) - 210*pow(x,4)*pow(y,6) + 45*pow(x,2)*pow(y,8) - pow(y,10))*z)/(512.*pow(r,11));
				case 11:
					return (sqrt(2028117/PI)*x*(pow(x,10) - 55*pow(x,8)*pow(y,2) + 330*pow(x,6)*pow(y,4) - 462*pow(x,4)*pow(y,6) + 165*pow(x,2)*pow(y,8) - 11*pow(y,10)))/(1024.*pow(r,11));
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
	double r = sqrt(x*x+y*y+z*z);
	if (r<pow(10.0,-9))
		return;
	int op=0;
	for (int i=0; i<SH_NRAD ; ++i)
	{
		double Gv = Gau(r, RBFS[i][0],RBFS[i][1]);
		for (int l=0; l<SH_LMAX+1 ; ++l)
		{
			for (int m=-l; m<l+1 ; ++m)
			{
				output[op] += Gv*CartSphericalHarmonic(l,m,x,y,z)*fac;
				//cout << "Real vs. Cart: " << RealSphericalHarmonic(l,m,theta,phi) << " " << CartSphericalHarmonic(l,m,x,y,z) << endl;
				++op;
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


