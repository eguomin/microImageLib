#ifndef _NR_H_
#define _NR_H_

#ifndef _FCOMPLEX_DECLARE_T_
typedef struct FCOMPLEX { float r, i; } fcomplex;
#define _FCOMPLEX_DECLARE_T_
#endif /* _FCOMPLEX_DECLARE_T_ */

#ifndef _ARITHCODE_DECLARE_T_
typedef struct {
	unsigned long *ilob, *iupb, *ncumfq, jdif, nc, minint, nch, ncum, nrad;
} arithcode;
#define _ARITHCODE_DECLARE_T_
#endif /* _ARITHCODE_DECLARE_T_ */

#ifndef _HUFFCODE_DECLARE_T_
typedef struct {
	unsigned long *icod, *ncod, *left, *right, nch, nodemax;
} huffcode;
#define _HUFFCODE_DECLARE_T_
#endif /* _HUFFCODE_DECLARE_T_ */

#include <stdio.h>

#if defined(__STDC__) || defined(ANSI) || defined(NRANSI) /* ANSI */


float brent(float ax, float bx, float cx,
	float(*f)(float), float tol, float *xmin);
float f1dim(float x);
void linmin(float p[], float xi[], int n, float *fret,
	float(*func)(float[]));
void mnbrak(float *ax, float *bx, float *cx, float *fa, float *fb,
	float *fc, float(*func)(float));
//void powell(float p[], float **xi, int n, float ftol, int *iter, float *fret,
//	float(*func)(float[]));
void powell(float p[], float **xi, int n, float ftol, int *iter, float *fret,
	float(*func)(float[]), int *totalIt, int itLimit);

#else /* ANSI */
/* traditional - K&R */

float brent();
float f1dim();
void linmin();
void mnbrak();
void powell()

#endif /* ANSI */

#endif /* _NR_H_ */
