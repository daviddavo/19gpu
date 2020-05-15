#ifndef G
#define G 6.674e-11f
#endif

__kernel void nbodies_naive(
    const __global float * bodymass,
    __global float3 * bodyvel,
    __global float3 * bodypos,
    const float dt,
    const int nsteps
    ) {
    /* La memoria cache es tan buena a partir de Fermi, que lo de usar
     * memoria local ya lo implementaré si me da tiempo
     */
    // __local float4 * lbodyvec[BLOCK_DIM];
    // __local float4 * lbodypos[BLOCK_DIM];

    // lbodyvec[get_local_id(0)] = bodyvec[get_global_id()];
    // lbodypos[get_local_id(0)] = bodypos[get_global_id()];
    
    int i = get_global_id(0);
    /* printf("i: %d, bodymass: let mapleader = "."%f\n", i, bodymass[i]); */

    // barrier(CLK_LOCAL_MEM_FENCE);

    /* printf("k: %d, i: %d, bodypos: (%f, %f, %f), bodyvel: (%f, %f, %f)\n", */
         /* 0, i, bodypos[i].x, bodypos[i].y, bodypos[i].z, bodyvel[i].x, bodyvel[i].y, bodyvel[i].z); */
    for (int k = 0; k < nsteps; k++) {
        barrier(CLK_GLOBAL_MEM_FENCE);

        // primero calculamos las fuerzas
        float3 F = 0.0f;
        for (int j = 0; j < get_global_size(0); j++) {
            if (i != j) {
                float3 d = bodypos[j] - bodypos[i];
                float3 invd = 1.0f / sqrt(dot(d, d)); 
                float3 invd3 = invd*invd*invd;
                
                F += G * bodymass[i] * bodymass[j] * d * invd3;
            }
        }

        // F=m*a
        bodyvel[i] += dt*F/bodymass[i];
        bodypos[i] += bodyvel[i] * dt; 
        /* printf("k: %d, i: %d, bodypos: (%f, %f, %f), bodyvel: (%f, %f, %f)\n", */
            /* k+1, i, bodypos[i].x, bodypos[i].y, bodypos[i].z, bodyvel[i].x, bodyvel[i].y, bodyvel[i].z); */
    }
}
