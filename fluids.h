//#ifdef _FLUIDS_H_
//#define _FLUIDS_H_

#include <string>
#include <iostream>
#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <GL/glut.h>
#include <GL/freeglut.h>
#include "glm/glm.hpp"
#include "glm/gtx/transform.hpp"

#define N 62
#define SIZE ((N+2)*(N+2)*(N+2))
#define IX(i,j,k) ((i) + (N+2) * (j) + (N+2) * (N+2) * (k))
#define SWAP(x0,x) {float *tmp=x0; x0=x; x=tmp;}
#define DIFF 0.0000005
#define VISC 0.0025
#define RAY_STEP 0.2;
#define K 10



//#endif