#include <pycuda-complex.hpp>
#define pi 3.14159265
#define phi 1.6180339
#include <stdio.h>
typedef  pycuda::complex<cudaPres> pyComplex;
__device__ cudaPres gValue(cudaPres t, cudaPres x, cudaPres y, cudaPres z, cudaPres constG, cudaPres ampG, cudaPres omegaG ){
return constG ;
}
__device__ cudaPres MfieldBX( cudaPres x, cudaPres y, cudaPres z, cudaPres t, cudaPres constMag, cudaPres ampMag, cudaPres omegaMag ){
//   return 6000.0f*cos(2*pi*x/6.0f)*cos(2*pi*x/6.0f) + 2000;
return x;//+2*sin(pi*t);
//   return -cos(0.8*y)*sin(0.8*x);

}
__device__ cudaPres MfieldBY( cudaPres x, cudaPres y, cudaPres z, cudaPres t, cudaPres constMag, cudaPres ampMag, cudaPres omegaMag ){
  //   return 6000.0f*cos(2*pi*x/6.0f)*cos(2*pi*x/6.0f) + 2000;
  return -(y - constMag +ampMag*sin(t*omegaMag));
  //   return cos(0.8*x)*sin(0.8*y);
  }
  __device__ cudaPres MfieldBZ( cudaPres x, cudaPres y, cudaPres z, cudaPres t ){
    //   return 6000.0f*cos(2*pi*x/6.0f)*cos(2*pi*x/6.0f) + 2000;
    return 0;
    }

__device__ cudaPres KspaceFFT(int tid, int nPoint, cudaPres L){
cudaPres Kfft;
if (tid < nPoint/2){
  Kfft = 2.0f*pi*(tid)/L;
  }
else {
Kfft = 2.0f*pi*(tid-nPoint)/L;
}
return Kfft;
}

__global__ void getAlphas_kernel( cudaPres dx, cudaPres dy, cudaPres dz,
cudaPres xMin, cudaPres yMin, cudaPres zMin,
cudaPres gammaX, cudaPres gammaY, cudaPres gammaZ, int nSpinor, cudaPres constG,
pyComplex *psi1, pyComplex *alphas){
int tid_x = blockIdx.x*blockDim.x + threadIdx.x;
int tid_y = blockIdx.y*blockDim.y + threadIdx.y;
int tid_z = blockIdx.z*blockDim.z + threadIdx.z;
int tid   = gridDim.z * blockDim.z * gridDim.y * blockDim.y * tid_x + gridDim.z * blockDim.z * tid_y + tid_z;
cudaPres result = 0.0;
cudaPres ri = tid_x*dx + xMin;
result += 0.5*gammaX*gammaX*ri*ri;
// result += 20.0*0.28633747*sin(phi*ri)*sin(phi*ri);
// result += 20.0*0.62177085*sin(pi*ri)*sin(pi*ri);
ri = tid_y*dy + yMin;
result += 0.5*gammaY*gammaY*ri*ri;
// result += 20.0*0.50630326*sin(sqrt(3.0)*ri)*sin(sqrt(3.0)*ri);
// result += 20.0*0.51316782*sin(sqrt(2.0)*ri)*sin(sqrt(2.0)*ri);
ri = tid_z*dz + zMin;
result += 0.5*gammaZ*gammaZ*ri*ri;
// result += 20.0*0.7949122*sin(sqrt(pi)*ri)*sin(sqrt(pi)*ri);
// result += 20.0*0.6949122*sin(sqrt(phi)*ri)*sin(sqrt(phi)*ri);
//ri=0.0;
//ri+=;
//ri+=0.50630326*sin(sqrt(3.0)*y)*sin(sqrt(3.0)*y) + 0.51316782*sin(sqrt(2.0)*y)*sin(sqrt(2.0)*y);
//ri+=0.28633747*sin(sqrt(pi)*z)*sin(sqrt(pi)*z) + 0.62177085*sin(sqrt(phi)*z)*sin(sqrt(phi)*z);
//result += 20.0*ri;
ri = abs(psi1[tid]);
alphas[tid] =  result + constG*ri*ri;
//cudaPres x = tid_x*dx + xMin;
//cudaPres y = tid_y*dy + yMin;
//cudaPres z = tid_z*dz + zMin;
//cudaPres randPot = 0.0;
//randPot+=0.28633747*sin(phi*x)*sin(phi*x) + 0.62177085*sin(pi*x)*sin(pi*x);
//randPot+=0.50630326*sin(sqrt(3.0)*y)*sin(sqrt(3.0)*y) + 0.51316782*sin(sqrt(2.0)*y)*sin(sqrt(2.0)*y);
//randPot+=0.28633747*sin(sqrt(pi)*z)*sin(sqrt(pi)*z) + 0.62177085*sin(sqrt(phi)*z)*sin(sqrt(phi)*z);
//cudaPres g = constG;//gValue(0.0f,x,y,z,constG, 0, 0);
//cudaPres psi_mod = abs(psi1[tid]);
//alphas[tid] =  0.5f*( gammaX*x*x + gammaY*y*y + gammaZ*z*z +40.0*randPot) + constG*psi_mod*psi_mod;
}
__global__ void curandinit_kernel( cudaPres dx, cudaPres dy, cudaPres dz,
cudaPres xMin, cudaPres yMin, cudaPres zMin,
cudaPres gammaX, cudaPres gammaY, cudaPres gammaZ,
cudaPres *rndReal, cudaPres *rndImag,  pyComplex *psi){
int tid_x = blockIdx.x*blockDim.x + threadIdx.x;
int tid_y = blockIdx.y*blockDim.y + threadIdx.y;
int tid_z = blockIdx.z*blockDim.z + threadIdx.z;
int tid   = gridDim.z * blockDim.z * gridDim.y * blockDim.y * tid_x + gridDim.z * blockDim.z * tid_y + tid_z;
cudaPres x = tid_x*dx + xMin;
cudaPres y = tid_y*dy + yMin;
cudaPres z = tid_z*dz + zMin;
pyComplex auxC;
cudaPres aux = exp(-gammaX*x*x-gammaY*y*y-gammaZ*z*z);
auxC._M_re = aux + aux*(rndReal[tid]-0.5);
auxC._M_im = aux + aux*(rndImag[tid]-0.5);
psi[tid] = auxC;
}
__global__ void implicitStep1_kernel( cudaPres xMin, cudaPres yMin, cudaPres zMin, cudaPres dx, cudaPres dy, cudaPres dz, int nSpinor,
cudaPres alpha, cudaPres omega, cudaPres gammaX, cudaPres gammaY, cudaPres gammaZ, cudaPres kappa, cudaPres Bz, cudaPres constG,
cudaPres constMag, pyComplex *psi1_d){
int tid_x = blockIdx.x*blockDim.x + threadIdx.x;
int tid_y = blockIdx.y*blockDim.y + threadIdx.y;
int tid_z = blockIdx.z*blockDim.z + threadIdx.z;
int tid   = gridDim.z * blockDim.z * gridDim.y * blockDim.y * tid_x + gridDim.z * blockDim.z * tid_y + tid_z;
cudaPres result = 0.0;
cudaPres ri = tid_x*dx + xMin;
result += 0.5*gammaX*gammaX*ri*ri;
// result += 20.0*0.28633747*sin(phi*ri)*sin(phi*ri);
// result += 20.0*0.62177085*sin(pi*ri)*sin(pi*ri);
ri = tid_y*dy + yMin;
result += 0.5*gammaY*gammaY*ri*ri;
// result += 20.0*0.50630326*sin(sqrt(3.0)*ri)*sin(sqrt(3.0)*ri);
// result += 20.0*0.51316782*sin(sqrt(2.0)*ri)*sin(sqrt(2.0)*ri);
ri = tid_z*dz + zMin;
result += 0.5*gammaZ*gammaZ*ri*ri;
// result += 20.0*0.7949122*sin(sqrt(pi)*ri)*sin(sqrt(pi)*ri);
// result += 20.0*0.6949122*sin(sqrt(phi)*ri)*sin(sqrt(phi)*ri);
// pyComplex iComplex(0.0f, 1.0f);
// pyComplex complex1(1.0f, 0.0f);
pyComplex psi1; //,Vtrap,torque;//psi1, psi2, psi3, partialX, partialY, Vtrap, torque, lz, result;
// cudaPres g = constG;//gValue(0.0f,x,y,z,constG,0.0f,0.0f);

psi1 = psi1_d[tid];
ri = abs(psi1);
result *= -1;
result += alpha;
result -= constG*ri*ri;
// cudaPres psiMod = abs(psi1);
// Vtrap = psi1*(gammaX*x*x + gammaY*y*y + gammaZ*z*z + 40*randPot )*0.5f;
// torque = psi1*g*psiMod*psiMod;
//     partialX = partialX_d[tid];
//     partialY = partialY_d[tid];
//     lz = iComplex * omega * (partialY*(x-x0) - partialX*(y-y0));
//     G1_d[tid] = psi1*alpha - Vtrap - torque - lz;
// psi1_d[tid] = psi1*alpha - Vtrap - torque;
psi1_d[tid] = psi1*result;
}
__global__ void implicitStep2_kernel( cudaPres dt, cudaPres alpha, int nSpinor,
int nPointX, int nPointY, int nPointZ, cudaPres Lx, cudaPres Ly, cudaPres Lz,
pyComplex *psiTransf, pyComplex *GTranf){
int tid_x = blockIdx.x*blockDim.x + threadIdx.x;
int tid_y = blockIdx.y*blockDim.y + threadIdx.y;
int tid_z = blockIdx.z*blockDim.z + threadIdx.z;
int tid   = gridDim.z * blockDim.z * gridDim.y * blockDim.y * tid_x + gridDim.z * blockDim.z * tid_y + tid_z;
cudaPres k2 = 0.0;
cudaPres kAux = KspaceFFT(tid_x,nPointX, Lx);//kx[tid_y];
k2 += kAux*kAux;
kAux = KspaceFFT(tid_y,nPointY, Ly);//ky[tid_x];
k2 += kAux*kAux;
kAux = KspaceFFT(tid_z,nPointZ, Lz);//kz[tid_z];
k2 += kAux*kAux;
//cudaPres kX = KspaceFFT(tid_x,nPointX, Lx);//kx[tid_y];
//cudaPres kY = KspaceFFT(tid_y,nPointY, Ly);//ky[tid_x];
//cudaPres kZ = KspaceFFT(tid_z,nPointZ, Lz);//kz[tid_z];

pyComplex factor, psiT, Gt;
// factor = 2.0 / ( 2.0 + dt*(k2 + 2.0*alpha) );
kAux = 2.0 / ( 2.0 + dt*(k2 + 2.0*alpha) );
psiT = psiTransf[tid];
Gt = GTranf[tid];
psiTransf[tid] = kAux * ( psiT + Gt*dt);

//   if (nSpinor > 1){
//     psiT = psi2Transf[tid];
//     Gt = G2Tranf[tid];
//     psi2Transf[tid] = factor * ( psiT + Gt*dt);
//   }
//
//   if (nSpinor == 3){
//     psiT = psi3Transf[tid];
//     Gt = G3Tranf[tid];
//     psi3Transf[tid] = factor * ( psiT + Gt*dt);
//   }
}



// KERNELS EN 2D
__global__ void getAlphas_kernel2D( int Sigma, cudaPres dx, cudaPres dy,
cudaPres xMin, cudaPres yMin, cudaPres kappa,
cudaPres gammaX, cudaPres gammaY, cudaPres z, int nSpinor, cudaPres constG,
pyComplex *psi1, pyComplex *alphas){
int tid_x = blockIdx.x*blockDim.x + threadIdx.x;
int tid_y = blockIdx.y*blockDim.y + threadIdx.y;
int tid   = gridDim.y * blockDim.y * tid_x + tid_y ;
int sigma;
cudaPres R, q2;
cudaPres result = 0.0;
cudaPres x = tid_x*dx + xMin;
cudaPres y = tid_y*dy + yMin;
pyComplex ri;

sigma = 2*Sigma - 1;
R = sqrt(x*x + y*y + z*z);
q2 = x*x + y*y;
result += 0.5*(q2 + (1.0+z*z/(R*R))/(4.0*R*R)+(1.0 - 2.0*z/R + z*z/(R*R))/(4.0*q2)) + sigma*kappa*R;
ri = abs(psi1[tid]);
alphas[tid] =  result + constG*ri*ri;
}

__global__ void getAlphas_kernel2DReal( cudaPres dx, cudaPres dy,
cudaPres xMin, cudaPres yMin, cudaPres kappa,
cudaPres gammaX, cudaPres gammaY, cudaPres z, cudaPres constG,
pyComplex *psi1, pyComplex *psi2, pyComplex *alphas){
int tid_x = blockIdx.x*blockDim.x + threadIdx.x;
int tid_y = blockIdx.y*blockDim.y + threadIdx.y;
int tid   = gridDim.y * blockDim.y * tid_x + tid_y ;
cudaPres result = 0.0;
cudaPres x = tid_x*dx + xMin;
cudaPres y = tid_y*dy + yMin;
cudaPres ri;

result += 0.5*(gammaX*x*x + gammaY*y*y);
ri = abs(psi1[tid]);
result += constG*ri*ri;
ri = abs(psi2[tid]);
result += constG*ri*ri;
alphas[tid] =  result;
}


__global__ void curandinit_kernel2D( cudaPres dx, cudaPres dy,
cudaPres xMin, cudaPres yMin,
cudaPres gammaX, cudaPres gammaY,
cudaPres *rndReal, cudaPres *rndImag,  pyComplex *psi){
int tid_x = blockIdx.x*blockDim.x + threadIdx.x;
int tid_y = blockIdx.y*blockDim.y + threadIdx.y;
int tid   = gridDim.y * blockDim.y * tid_x +  tid_y;
cudaPres x = tid_x*dx + xMin;
cudaPres y = tid_y*dy + yMin;
pyComplex auxC; //aqui de seguro es la condicion inicial, para nuestro caso como es una red la condicion inicial debe ser una costante  
//cudaPres aux = 2 // ya que ahora tengo a una red mi condición inicial tiene que ser diferente en este caso una constante, elegimos un numero en este caso 2  
cudaPres aux = exp(-gammaX*x*x-gammaY*y*y);
auxC._M_re = aux + aux*(rndReal[tid]-0.5);
auxC._M_im = aux + aux*(rndImag[tid]-0.5);
psi[tid] = auxC;
}
__global__ void implicitStep1_kernel2DReal( cudaPres xMin, cudaPres yMin, cudaPres dx, cudaPres dy, 
cudaPres alpha, cudaPres gammaX, cudaPres gammaY, cudaPres kappa, cudaPres z_0, cudaPres constG, pyComplex *psi1_d, pyComplex *psi2_d){
int tid_x = blockIdx.x*blockDim.x + threadIdx.x;
int tid_y = blockIdx.y*blockDim.y + threadIdx.y;
int tid   = gridDim.y * blockDim.y * tid_x + tid_y;
cudaPres ri, result;
cudaPres x = tid_x*dx + xMin;
cudaPres y = tid_y*dy + yMin;
pyComplex psi1, psi2, iComplex(0.,1.), B;

psi1 = psi1_d[tid];
psi2 = psi2_d[tid];
result = -0.5*(gammaX*x*x+gammaY*y*y);
ri = abs(psi1);
result -= constG*ri*ri;
ri = abs(psi2);
result -= constG*ri*ri;
result += alpha;
result -= kappa*z_0;
B = kappa * (x - iComplex * y);
psi1_d[tid] = psi1*result - psi2*B;
result += 2.0*kappa*z_0;
B = kappa * (x + iComplex * y);
psi2_d[tid] = psi2*result - psi1*B;
}

__global__ void implicitStep1_kernel2D( int Sigma, cudaPres xMin, cudaPres yMin, cudaPres dx, cudaPres dy, int nSpinor,
cudaPres alpha, cudaPres omega, cudaPres gammaX, cudaPres gammaY, cudaPres kappa, cudaPres z, cudaPres constG,
cudaPres constMag, pyComplex *partialX_d, pyComplex *partialY_d, pyComplex *psi1_d){
int tid_x = blockIdx.x*blockDim.x + threadIdx.x;
int tid_y = blockIdx.y*blockDim.y + threadIdx.y;
int tid   = gridDim.y * blockDim.y * tid_x + tid_y;
cudaPres R, q2, ri;
cudaPres x = tid_x*dx + xMin;
cudaPres y = tid_y*dy + yMin;
int sigma = 2*Sigma - 1;
pyComplex psi1, partialX, partialY, result;

R = sqrt(x*x + y*y + z*z);
q2 = x*x + y*y;
result = -(0.5*(q2 + (1.0+z*z/(R*R))/(4.0*R*R)+(1.0 - 2.0*z/R + z*z/(R*R))/(4.0*q2)) + sigma*kappa*R);
psi1 = psi1_d[tid];
partialX = partialX_d[tid];
partialY = partialY_d[tid];
ri = sigma*(1.0 - z/R)/(2.0*q2);
result += ri*x*partialY;
result -= ri*y*partialX;
ri = abs(psi1);
result += alpha;
result -= constG*ri*ri;
psi1_d[tid] = psi1*result;
}



__global__ void implicitStep2_kernel2DReal( cudaPres dt, cudaPres alpha,
int nPointX, int nPointY, cudaPres Lx, cudaPres Ly,
pyComplex *psi1Transf, pyComplex *psi2Transf, pyComplex *G1Tranf, pyComplex *G2Tranf){
int tid_x = blockIdx.x*blockDim.x + threadIdx.x;
int tid_y = blockIdx.y*blockDim.y + threadIdx.y;
int tid   = gridDim.y * blockDim.y * tid_x + tid_y;
cudaPres k2 = 0.0;
cudaPres kAux = KspaceFFT(tid_x,nPointX, Lx);
k2 += kAux*kAux;
kAux = KspaceFFT(tid_y,nPointY, Ly);
k2 += kAux*kAux;

pyComplex psi1T, psi2T, Gt1, Gt2;
kAux = 2.0cString / ( 2.0cString + dt*(k2 + 2.0*alpha) );
psi1T = psi1Transf[tid];
psi2T = psi2Transf[tid];
Gt1 = G1Tranf[tid];
Gt2 = G2Tranf[tid];
psi1Transf[tid] = kAux * ( psi1T + Gt1*dt);
psi2Transf[tid] = kAux * ( psi2T + Gt2*dt);
}

__global__ void implicitStep2_kernel2D( cudaPres dt, cudaPres alpha, int nSpinor,
int nPointX, int nPointY, cudaPres Lx, cudaPres Ly, pyComplex *psiTransf, pyComplex *GTranf){
int tid_x = blockIdx.x*blockDim.x + threadIdx.x;
int tid_y = blockIdx.y*blockDim.y + threadIdx.y;
int tid   = gridDim.y * blockDim.y * tid_x + tid_y;
cudaPres k2 = 0.0;
cudaPres kAux = KspaceFFT(tid_x,nPointX, Lx);
k2 += kAux*kAux;
kAux = KspaceFFT(tid_y,nPointY, Ly);
k2 += kAux*kAux;

pyComplex factor, psiT, psiTn, Gt;
kAux = 2.0 / ( 2.0 + dt*(k2 + 2.0*alpha) );
psiT = psiTransf[tid];
Gt = GTranf[tid];
psiTransf[tid] = kAux * ( psiT + Gt*dt);
}

__global__ void unitaryTransfInvSm1_kernel2D(cudaPres xMin, cudaPres yMin, cudaPres dx, cudaPres dy, cudaPres z_0, pyComplex *psi_d, pyComplex *psi1_d, pyComplex *psi2_d){
int tid_x = blockIdx.x*blockDim.x + threadIdx.x;
int tid_y = blockIdx.y*blockDim.y + threadIdx.y;
int tid   = gridDim.y * blockDim.y * tid_x + tid_y;
cudaPres x = tid_x*dx + xMin;
cudaPres y = tid_y*dy + yMin;
pyComplex psi, exp, i_complex(0.,1.0);
cudaPres R, q, ri;
R = sqrt(x*x + y*y + z_0*z_0);
q = sqrt(x*x + y*y);
psi=psi_d[tid];

exp = x/q + i_complex*y/q;
ri=sqrt((z_0 + R)/(2.0cString*R));
psi1_d[tid]=ri*psi;
ri=-q/sqrt(2.0cString*R*(z_0+R));
psi2_d[tid]=ri*exp*psi;

}

__global__ void unitaryTransfInvS1_kernel2D(cudaPres xMin, cudaPres yMin, cudaPres dx, cudaPres dy, cudaPres z_0, 
pyComplex *psi_d, pyComplex *psi1_d, pyComplex *psi2_d){
int tid_x = blockIdx.x*blockDim.x + threadIdx.x;
int tid_y = blockIdx.y*blockDim.y + threadIdx.y;
int tid   = gridDim.y * blockDim.y * tid_x + tid_y;
cudaPres x = tid_x*dx + xMin;
cudaPres y = tid_y*dy + yMin;
pyComplex psi, exp, i_complex(0.,1.0);
cudaPres R, q, ri;
R = sqrt(x*x + y*y + z_0*z_0);
q = sqrt(x*x + y*y);
psi=psi_d[tid];

exp = x/q - i_complex*y/q;
ri=sqrt((z_0 + R)/(2.0cString*R));
psi2_d[tid]=ri*psi;
ri=q/sqrt(2.0cString*R*(z_0+R));
psi1_d[tid]=ri*exp*psi;
}

__global__ void unitaryTransf_kernel2D(cudaPres xMin, cudaPres yMin, cudaPres dx, cudaPres dy, cudaPres z_0, 
pyComplex *Upsi1_d, pyComplex *Upsi2_d, pyComplex *psiU1_d, pyComplex *psiU2_d){
int tid_x = blockIdx.x*blockDim.x + threadIdx.x;
int tid_y = blockIdx.y*blockDim.y + threadIdx.y;
int tid   = gridDim.y * blockDim.y * tid_x + tid_y;
cudaPres x = tid_x*dx + xMin;
cudaPres y = tid_y*dy + yMin;
pyComplex Upsi1, Upsi2, exp, i_complex(0.,1.0);
cudaPres R, q, ri1, ri2;
R = sqrt(x*x + y*y + z_0*z_0);
q = sqrt(x*x + y*y);
Upsi1=Upsi1_d[tid];
Upsi2=Upsi2_d[tid];

exp = x/q - i_complex*y/q;
ri1=sqrt((z_0 + R)/(2.0cString*R));
ri2=q/sqrt(2.0cString*R*(z_0+R));
psiU1_d[tid] = ri1*Upsi1 - ri2*exp*Upsi2;
exp = x/q + i_complex*y/q;
psiU2_d[tid] = ri1*Upsi2 + ri2*exp*Upsi1;

}

/*
__global__ void unitaryTransfInv_kernel2D(cudaPres xMin, cudaPres yMin, cudaPres dx, cudaPres dy, cudaPres z_0, pyComplex *psi1_d, pyComplex *psi2_d){
int tid_x = blockIdx.x*blockDim.x + threadIdx.x;
int tid_y = blockIdx.y*blockDim.y + threadIdx.y;
int tid   = gridDim.y * blockDim.y * tid_x + tid_y;
cudaPres x = tid_x*dx + xMin;
cudaPres y = tid_y*dy + yMin;
pyComplex psi1, psi2, exp, i_complex(0.,1.0);
cudaPres R, q, ri1, ri2;
R = sqrt(x*x + y*y + z_0*z_0);
q = sqrt(x*x + y*y);
psi1=psi1_d[tid];
psi2=psi2_d[tid];

exp = x/q - i_complex*y/q;
ri1=sqrt((z_0 + R)/(2.0cString*R));
ri2=q/sqrt(2.0cString*R*(z_0+R));
psi1_d[tid] = ri1*Upsi1 + ri2*exp*Upsi2;
exp = x/q + i_complex*y/q;
psi2_d[tid] = ri1*Upsi2 - ri2*exp*Upsi1;

}
*/


__global__ void applyObservable_kernel2D(int Sigma, int nPointX, int nPointY,
cudaPres Lx, cudaPres Ly,
cudaPres dx, cudaPres dy,
cudaPres xMin, cudaPres yMin,
cudaPres constG, cudaPres kappa, cudaPres z_0, cudaPres gammaX, cudaPres gammaY,
pyComplex *PartialX, pyComplex *PartialY, pyComplex *psiState_d, pyComplex *OTranf_d, int typeObservable){
//  typeObservable = {Chemicla Potential:0, Angular Mom:2 ..}
int tid_x = blockIdx.x*blockDim.x + threadIdx.x;
int tid_y = blockIdx.y*blockDim.y + threadIdx.y;
int tid   = gridDim.y * blockDim.y * tid_x + tid_y;
cudaPres x = tid_x*dx + xMin;
cudaPres y = tid_y*dy + yMin;
// cudaPres randPot = 0.0;
pyComplex psi1,Vtrap,torque,laPsi,partialX,partialY,Lz;
int sigma = 2*Sigma - 1;
cudaPres q2 = x*x + y*y;
cudaPres R = sqrt(q2 + z_0*z_0);

if (typeObservable == 0){
psi1 = psiState_d[tid];
laPsi = OTranf_d[tid]; //It suppose that in this array we have the Laplacian of Psi
partialX = PartialX[tid];
partialY = PartialY[tid];
cudaPres ri = sigma*(1.cString-z_0/R)/(2.cString*q2);
cudaPres psiMod = abs(psi1);
Lz = psi1*(x*partialY - y*partialX)*ri;
//potencial de la trampa armónica, en nuestro caso nosotros queremos una red óptica  con y sin desorden
//Vtrap = sin(x) + sin(y)
Vtrap = psi1*((gammaX*gammaX*q2+(1.cString+z_0*z_0/(R*R))+(1.cString-z_0/R)*(1.cString-z_0/R)/(4.cString*q2))*0.5cString+sigma*kappa*R);
torque = psi1*constG*psiMod*psiMod;
OTranf_d[tid] = -0.5cString*laPsi-Lz+Vtrap+torque;
}
}

__global__ void applyObservablereal_kernel2D( int nPointX, int nPointY,
cudaPres Lx, cudaPres Ly,
cudaPres dx, cudaPres dy,
cudaPres xMin, cudaPres yMin,
cudaPres constG, cudaPres gammaX, cudaPres gammaY, cudaPres z_0, cudaPres kappa,
pyComplex *psi1_d, pyComplex *psi2_d, pyComplex *OTranf1_d, pyComplex *Otranf2_d, int typeObservable){

//  typeObservable = {Chemicla Potential:0, Angular Mom:2 ..}

int tid_x = blockIdx.x*blockDim.x + threadIdx.x;
int tid_y = blockIdx.y*blockDim.y + threadIdx.y;
int tid   = gridDim.y * blockDim.y * tid_x + tid_y;
cudaPres x = tid_x*dx + xMin;
cudaPres y = tid_y*dy + yMin;
// cudaPres randPot = 0.0;
// randPot+=0.28633747*sin(phi*x)*sin(phi*x) + 0.62177085*sin(pi*x)*sin(pi*x);
// randPot+=0.50630326*sin(sqrt(3.0)*y)*sin(sqrt(3.0)*y) + 0.51316782*sin(sqrt(2.0)*y)*sin(sqrt(2.0)*y);
// randPot+=0.28633747*sin(sqrt(pi)*z)*sin(sqrt(pi)*z) + 0.62177085*sin(sqrt(phi)*z)*sin(sqrt(phi)*z);
pyComplex psi1,psi2,Vtrap,torque,laPsi1,laPsi2,B,iComplex(0.0,1.0);

if (typeObservable == 0){
psi1 = psi1_d[tid];
psi2 = psi2_d[tid];
laPsi1 = OTranf1_d[tid];
laPsi2 = Otranf2_d[tid]; //It suppose that in this array we have the Laplacian of Psi
cudaPres psiMod = abs(psi1);
Vtrap = (gammaX*gammaX*x*x + gammaY*gammaY*y*y)*0.5cString;
torque = constG*psiMod*psiMod;
psiMod = abs(psi2);
torque += constG*psiMod*psiMod;
B = kappa*(x-iComplex*y);
OTranf1_d[tid] = -0.5cString*laPsi1 + psi1*Vtrap + psi1*torque + psi1*kappa*z_0 + psi2*B;
B = kappa*(x+iComplex*y);
Otranf2_d[tid] = -0.5cString*laPsi2 + psi2*Vtrap + psi2*torque - psi2*kappa*z_0 + psi1*B;
}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void solvePartialX_kernel2D( cudaPres Lx,int nPointX, pyComplex *fftTrnf, pyComplex *partialxfft){
int tid_x = blockIdx.x*blockDim.x + threadIdx.x;
int tid_y = blockIdx.y*blockDim.y + threadIdx.y;
int tid   = gridDim.y * blockDim.y * tid_x + tid_y;

cudaPres kx = KspaceFFT(tid_x,nPointX, Lx);
partialxfft[tid] = kx*fftTrnf[tid];
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void solvePartialY_kernel2D( cudaPres Ly,int nPointY, pyComplex *fftTrnf, pyComplex *partialyfft){
int tid_x = blockIdx.x*blockDim.x + threadIdx.x;
int tid_y = blockIdx.y*blockDim.y + threadIdx.y;
int tid   = gridDim.y * blockDim.y * tid_x + tid_y;

cudaPres ky = KspaceFFT(tid_y,nPointY, Ly);
partialyfft[tid] = ky*fftTrnf[tid];
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void applyLaplacian_kernel2D( cudaPres Lx,cudaPres Ly,
int nPointX,int nPointY,
pyComplex *fftTrnf){
int tid_x = blockIdx.x*blockDim.x + threadIdx.x;
int tid_y = blockIdx.y*blockDim.y + threadIdx.y;
int tid   = gridDim.y * blockDim.y * tid_x + tid_y;

cudaPres kX = KspaceFFT(tid_x,nPointX, Lx);//kx[tid_y];
cudaPres kY = KspaceFFT(tid_y,nPointY, Ly);//ky[tid_x];
cudaPres k2 = kX*kX + kY*kY;
pyComplex value = fftTrnf[tid];
fftTrnf[tid] = -k2*value;
}
