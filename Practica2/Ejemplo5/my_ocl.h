#ifndef _OCL_H_

#define _OCL_H_

typedef struct { float m, x, y, z, vx, vy, vz; } body;

int nbodiesOCL(body * data, const int nBodies, const int nIters, const float dt);
#endif // _OCL_H_
