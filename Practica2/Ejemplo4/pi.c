#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/resource.h>

#include "my_ocl.h"

double get_time(){
	static struct timeval 	tv0;
	double time_, time;

	gettimeofday(&tv0,(struct timezone*)0);
	time_=(double)((tv0.tv_usec + (tv0.tv_sec)*1000000));
	time = time_/1000000;
	return(time);
}

double calc_pi(int n)
{
	int i;
	double x, area, pi;

	area= 0.0;
	for(i=1; i<n; i++) {
		x = (i+0.5)/n;
		area += 4.0/(1.0 + x*x);
	}
	pi = area/n;

	return(pi);
}

int main(int argc, char **argv)
{
	int n;
	double t0, t1;
	double pi;

	if (argc==2){
		n = atoi(argv[1]);
	} else {
		printf("./exec n [c,g] \n");
		return(-1);
	}

	switch (argv[2][0]) {
		case 'c':
			t0 = get_time();
			pi = calc_pi(n);
			t1 = get_time();
			printf("CPU Exection time %f ms. PI=%3.10f\n", t1-t0, pi);
			break;
		case 'g':
			t0 = get_time();
			pi= calc_piOCL(n);
			t1 = get_time();
			printf("OCL Exection time %f ms. PI=%3.10f\n", t1-t0, pi);
			break;
		default:
			printf("Not Implemented yet!!\n");


	}

	return 0;
}
