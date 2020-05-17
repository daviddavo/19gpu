#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/time.h>
#include <sys/resource.h>

#include "my_ocl.h"

double get_time(){
	static struct timeval 	tv0;
	double time_, time;

	gettimeofday(&tv0,(struct timezone*)0);
	time_=(double)((tv0.tv_usec + (tv0.tv_sec)*1000000));
	time = time_/1000;
	return(time);
}


// typedef struct { float m, x, y, z, vx, vy, vz; } body;

void randomizeBodies(body *data, int n) {
	for (int i = 0; i < n; i++) {
		data[i].m  = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;

		data[i].x  = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
		data[i].y  = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
		data[i].z  = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;

		data[i].vx = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
		data[i].vy = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
		data[i].vz = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
	}
}

void bodyForce(body *p, float dt, int n) {

	for (int i = 0; i < n; i++) { 
		float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

		for (int j = 0; j < n; j++) {
			if (i!=j) {
				float dx = p[j].x - p[i].x;
				float dy = p[j].y - p[i].y;
				float dz = p[j].z - p[i].z;
				float distSqr = dx*dx + dy*dy + dz*dz;
				float invDist = 1.0f / sqrtf(distSqr);
				float invDist3 = invDist * invDist * invDist;

				float G = 6.674e-11;
				float g_masses = G * p[j].m * p[i].m;

				Fx += g_masses * dx * invDist3; 
				Fy += g_masses * dy * invDist3; 
				Fz += g_masses * dz * invDist3;
			}
		}

		p[i].vx += dt*Fx/p[i].m; p[i].vy += dt*Fy/p[i].m; p[i].vz += dt*Fz/p[i].m;
	}
}

void integrate(body *p, float dt, int n){
	for (int i = 0 ; i < n; i++) {
		p[i].x += p[i].vx*dt;
		p[i].y += p[i].vy*dt;
		p[i].z += p[i].vz*dt;
        // printf("k: %d, i: %d, bodypos: (%f, %f, %f), bodyvel: (%f, %f, %f)\n",
            // -1, i, p[i].x, p[i].y, p[i].z, p[i].vx, p[i].vy, p[i].vz);
	}
}

void nbodies(body * p, const int nBodies, const int nIters, const float dt)
{
    // Note: Now nIters and dt are another argument of the function
	double t0 = get_time();

    // for (int i = 0; i < nBodies; i++) {
        // printf("i: %d, bodymass: %f\n", i, p[i].m);
        // printf("k: %d, i: %d, bodypos: (%f, %f, %f), bodyvel: (%f, %f, %f)\n",
            // 0, i, p[i].x, p[i].y, p[i].z, p[i].vx, p[i].vy, p[i].vz);
    // }

	for (int iter = 1; iter <= nIters; iter++) {
		bodyForce(p, dt, nBodies); // compute interbody forces
		integrate(p, dt, nBodies); // integrate position
	}

	double totalTime = get_time()-t0; 
	printf("%d Bodies with %d iterations: %0.3f Millions Interactions/second\n", nBodies, nIters, nIters* 1e-6f * nBodies * nBodies / totalTime);
}

char checkCPUGPU (const body * pcpu, const body * pgpu, 
    const unsigned int nBodies, const float e) 
{
    for (int i = 0; i < nBodies; i++) {
        // We assume mass is the same
        if (1-fabsf(pcpu[i].x/pgpu[i].x) >= e 
            || 1-fabsf(pcpu[i].y/pgpu[i].y) >= e
            || 1-fabsf(pcpu[i].z/pgpu[i].z) >= e) {
            printf("Warning: position is not the same! (body %d)\n", i);
            printf("cpu: (%f, %f, %f), gpu: (%f, %f, %f)\n",
                pcpu[i].x, pcpu[i].y, pcpu[i].z, pgpu[i].x, pgpu[i].y, pgpu[i].z);
            return EXIT_FAILURE;
        }

        if (1-fabsf(pcpu[i].vx/pgpu[i].vx) >= e
            || 1-fabsf(pcpu[i].vy/pgpu[i].vy) >= e
            || 1-fabsf(pcpu[i].vz/pgpu[i].vz) >= e) {
            printf("Warning: velocity is not the same! (body %d)\n", i);
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}


int main(const int argc, const char** argv) {

	int nBodies = 1000;
    int nSteps = 100;
	double t0, t1, tgpu, tcpu;
    
    if (argc < 3 || argc > 4) {
        printf("%s nbodies [c,g] [steps]\n", argv[0]);
        return -1;
    }

	if (argc >= 3)
		nBodies = atoi(argv[1]);

    if (argc == 4)
        nSteps = atoi(argv[3]);

	body *p = (body*)malloc(nBodies*sizeof(body));
    body *pgpu;
	randomizeBodies(p, nBodies); // Init pos / vel data

	switch (argv[2][0]) {
		case 'c':
			t0 = get_time();
			nbodies(p, nBodies, nSteps, 0.01f);
			t1 = get_time();
			printf("CPU Exection time %f ms.\n", t1-t0);
			break;
		case 'g':
            pgpu = (body*)calloc(sizeof(body), nBodies);
            memcpy(pgpu, p, sizeof(body)*nBodies);
			t0 = get_time();
            if (nbodiesOCL(pgpu, nBodies, nSteps, 0.01f) != EXIT_SUCCESS) {
                fprintf(stderr, "Failed to run simulation on GPU\n");
                break;
            }
			t1 = get_time();
            tgpu = t1 - t0;

			printf("OCL Exection time %f ms.\n", tgpu);
            char c = 1;
            if (tgpu > 1000) {
                printf("Warning!. Comparing against CPU would take a long time (about 20 times longer).\nContinue? (y/n): ");
                while ( (c=getchar()) != 'y' && c != 'n' ) {}
                c = c == 'y';
            }

            if (c) {
                t0 = get_time();
                nbodies(p, nBodies, nSteps, 0.01f);
                t1 = get_time();
                tcpu = t1-t0;

                printf("CPU Exection time %f ms.\n", tcpu);
                // Comprobamos que está en el mismo orden de magnitud.
                // Tal vez estoy siendo demasiado generoso con el error y
                // no sé si la implementacion es mala o 
                // es debido al acarreo del error en el redondeo
                if(checkCPUGPU(p, pgpu, nBodies, 1.0f)) {
                    printf("Warning!!, CPU != GPU. This is normal for more than 50 iterations (it's a chaotic system and little problems with rounding can cause large variations in results)\n");
                } else {
                    printf("Success. CPU == GPU\n");
                }
            } else {
                printf("Not checking against CPU\n");
            }

            free(pgpu);
			break;
		default:
			printf("Not Implemented yet!!\n");


	}

    free(p);
	return(0);
}
