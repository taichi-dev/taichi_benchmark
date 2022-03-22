#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <math.h>

#include <sTimer.h>
#include <sRandom.h>
#include <helper_math.h>
#include <svpng.inc>

#define PI 3.14159265359f

// -----------------------------------GPU Func-----------------------------------
// From [smallpt](http://www.kevinbeason.com/smallpt/)

// Rendering parameters
// max_depth = 10, sphere, image size 1024x768

enum materialType
{ 
    DIFFUSE = 0, 
    MIRROR, 
    GLASS
};

struct Ray
{
    __device__ Ray(float3 origin, float3 direction) 
        : origin(origin), direction(direction) {}

    float3 origin;
    float3 direction;
};

struct sphere
{
    float radius;
    float3 center, emission, reflectance;
    materialType type;

    __device__ double intersect(const Ray &r) const
    {

        float3 op = center - r.origin;
        float t, epsilon = 0.0001f;  // epsilon required to prevent floating point precision artefacts
        float b = dot(op, r.direction);    // b in quadratic equation
        float disc = b*b - dot(op, op) + radius*radius;  // discriminant quadratic equation
        if(disc < 0) return 0;       // if disc < 0, no real solution (we're not interested in complex roots) 
        else disc = sqrtf(disc);    // if disc >= 0, check for solutions using negative and positive discriminant
        return (t = b - disc) > epsilon ? t : ((t = b + disc) > epsilon ? t : 0); // pick closest point in front of ray origin
    }
};

__constant__ sphere spheres[] = {
    {1e5f,{1e5f + 1.0f, 40.8f, 81.6f},{0.0f, 0.0f, 0.0f},{0.75f, 0.25f, 0.25f}, DIFFUSE}, //Left 
    {1e5f,{-1e5f + 99.0f, 40.8f, 81.6f},{0.0f, 0.0f, 0.0f},{.25f, .25f, .75f}, DIFFUSE}, //Rght 
    {1e5f,{50.0f, 40.8f, 1e5f},{0.0f, 0.0f, 0.0f},{.75f, .75f, .75f}, DIFFUSE}, //Back 
    {1e5f,{50.0f, 40.8f, -1e5f + 170.0f},{0.0f, 0.0f, 0.0f},{0.0f, 0.0f, 0.0f}, DIFFUSE}, //Frnt 
    {1e5f,{50.0f, 1e5f, 81.6f},{0.0f, 0.0f, 0.0f},{.75f, .75f, .75f}, DIFFUSE}, //Botm 
    {1e5f,{50.0f, -1e5f + 81.6f, 81.6f},{0.0f, 0.0f, 0.0f},{.75f, .75f, .75f}, DIFFUSE}, //Top 
    {16.5f,{27.0f, 16.5f, 47.0f},{0.0f, 0.0f, 0.0f},{1, 1, 1}, MIRROR},//Mirr
    {16.5f,{73.0f, 16.5f, 78.0f},{0.0f, 0.0f, 0.0f},{1, 1, 1}, GLASS},//Glas
    {600.0f,{50.0f, 681.6f-.27f, 81.6f},{12, 12, 12},{0.0f, 0.0f, 0.0f}, DIFFUSE}  // Light
};

__device__ float rgbToLuminance(const float3& rgb)
{
    const float YWeight[3] = {0.212671f, 0.715160f, 0.072169f};
    return YWeight[0] * rgb.x + YWeight[1] * rgb.y + YWeight[2] * rgb.z;
}

__device__ bool intersectScene(const Ray &r, float &t, int &id)
{
    float n = sizeof(spheres) / sizeof(sphere), d, inf = t = 1e20;  // t is distance to closest intersection, initialise t to a huge number outside scene
    for(int i = int(n); i--;)
    {
        // find closest hit object and point
        if((d = spheres[i].intersect(r)) && d < t)
        {
            t = d;
            id = i;
        }
    }
        
    return t < inf; // returns true if an intersection with the scene occurred, false when no hit
}

__device__ float clamp(float x) { return x < 0 ? 0 : x>1 ? 1 : x; }

__device__ float gammaCorrection(float x)
{
    return pow(clamp(x), 1 / 2.2f);
}

__device__ float3 radiance(Ray &r, curandState* rs)
{
    float3 L = make_float3(0.0f, 0.0f, 0.0f); // accumulates ray colour with each iteration through bounce loop
    float3 throughput = make_float3(1.0f, 1.0f, 1.0f);
    int depth = 0;

    // ray bounce loop
    for (int i=0; i<10/*max_depth*/; i++) {
        float t;    
        int id = 0;         

        // find closest intersection with object's index
        if(!intersectScene(r, t, id))
            break;

        const sphere &obj = spheres[id];
        float3 hitpoint = r.origin + r.direction * t; 
        float3 normal = normalize(hitpoint - obj.center);
        float3 nl = dot(normal, r.direction) < 0 ? normal : normal * -1; // front facing normal

        // prevent self-intersection
        r.origin = hitpoint + nl * 0.05f;

        //float pdf = 1.0f;

        // add emission
        L += throughput * obj.emission;

        // different material
        if(obj.type == DIFFUSE)
        {        
            // uniform sampling hemisphere
            float r1 = 2 * PI * curand_uniform(rs);
            float r2 = curand_uniform(rs);
            float r2s = sqrtf(r2);

            // compute local coordinate on the hit point
            float3 w = nl;
            float3 u = normalize(cross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
            float3 v = cross(w, u);

            // local to world convert
            r.direction = normalize(u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrtf(1 - r2));
            //pdf = 1.0f / PI;

            // importance sampling no need costheta
            //throughput *= obj.reflectance * dot(r.direction, nl);
            throughput *= obj.reflectance;
        }
        else if(obj.type == MIRROR)
        {
            r.direction = r.direction - normal * 2 * dot(normal, r.direction);
            throughput *= obj.reflectance;
            //pdf = 1.0f;
        }
        else
        {
            r.origin = hitpoint;

            // Ideal dielectric REFRACTION
            float3 reflectDir = r.direction - normal * 2 * dot(normal, r.direction);
            // Ray from outside going in?
            bool into = dot(normal, nl) > 0;
            float nc = 1, nt = 1.5, nnt = into ? nc / nt : nt / nc, ddn = dot(r.direction, nl), cos2t;
            
            // total internal reflection
            if((cos2t = 1 - nnt*nnt*(1 - ddn*ddn)) < 0)
            {
                r.direction = reflectDir;
                throughput *= obj.reflectance;
            }
            else
            {
                // refract or reflect
                float3 tdir = normalize(r.direction*nnt - normal*((into ? 1 : -1)*(ddn*nnt + sqrt(cos2t))));

                float a = nt - nc, b = nt + nc, R0 = a*a / (b*b), c = 1 - (into ? -ddn : dot(tdir, normal));

                float Re = R0 + (1 - R0)*c*c*c*c*c, Tr = 1 - Re, P = .25 + .5*Re, RP = Re / P, TP = Tr / (1 - P);
                
                if(curand_uniform(rs) < P)
                {
                    // reflect
                    r.direction = reflectDir;
                    throughput *= obj.reflectance * RP;
                }
                else
                {
                    //refract
                    r.direction = tdir;
                    throughput *= obj.reflectance * TP;
                    //throughput *= make_float3(1, 0, 0);
                }
            }
        }

        // Russian roulette Stop with at least some probability to avoid getting stuck
        if(depth++ >= 5)
        {
            float q = min(0.95f, rgbToLuminance(throughput));
            if(curand_uniform(rs) >= q)
                break;
            throughput /= q;
        }
    }

    return L;
}

__global__ void render(int spp, int width, int height, float3* output)
{
    // position of current pixel
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // index of current pixel
    //int i = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    int i = (height - y - 1) * width + x;

    curandState rs;
    curand_init(i, 0, 0, &rs);

    Ray cam(make_float3(50, 52, 295.6), normalize(make_float3(0, -0.042612, -1))); // cam pos, dir 
    float3 cx = make_float3(width * 0.5135f / height, 0.0f, 0.0f);
    // .5135 is field of view angle
    float3 cy = normalize(cross(cx, cam.direction)) * 0.5135f;
    float3 color = make_float3(0.0f);

    for (int sy = 0; sy < 2; sy++)
    {
        for (int sx = 0; sx < 2; sx++)
        { 
            for(int s = 0; s < spp; s++)
            {
                float r1 = curand_uniform(&rs);
                float dx = r1 < 1 ? sqrtf(r1) - 1 : 1-sqrtf(2 - r1);
                float r2 = curand_uniform(&rs);
                float dy = r2 < 1 ? sqrtf(r2) - 1 : 1-sqrtf(2 - r2);
                //--! super sampling
                float3 d = cam.direction + cx*((((sx + dx + .5) / 2) + x) / width - .5) + 
                                           cy*((((sy + dy + .5) / 2) + y) / height - .5);

                Ray tRay = Ray(cam.origin + d * 140, normalize(d));
                color += radiance(tRay, &rs) *(.25f / spp);
            }
        }
    }

    // output to the cache
    output[i] = make_float3(clamp(color.x, 0.0f, 1.0f), clamp(color.y, 0.0f, 1.0f), clamp(color.z, 0.0f, 1.0f));
}

// -----------------------------------CPU Func-----------------------------------
int toInt(float x)
{
    return (int) (pow(clamp(x, 0.0f, 1.0f), 1.0f / 2.2f) * 255 + 0.5f);
}

void save(const char* fileName, int width, int height, float3* data)
{
    FILE *fp = fopen(fileName, "wb");

    // Convert from float3 array to uchar array
    unsigned char* output = new unsigned char[width * height * 3];

    for(int i = 0; i < width * height; i++)
    {
        //printf_s("%f %f %f \n", data[i].x, data[i].y, data[i].z);
        output[i * 3 + 0] = toInt(data[i].x);
        output[i * 3 + 1] = toInt(data[i].y);
        output[i * 3 + 2] = toInt(data[i].z);
    }

    svpng(fp, width, height, output, 0);
    fclose(fp);
    delete[] output;
}

int main(int argc, char *argv[])
{
    // Image Size
    int width = 1024, height = 768;
    int spp_in = argc==2 ? atoi(argv[1]) : 512;
    int spp = spp_in / 4;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
 
    // Memory on CPU
    float3* outputCPU = new float3[width * height];
    float3* outputGPU;
    cudaMalloc(&outputGPU, width * height * sizeof(float3));

    // Ray Pool
    dim3 blockSize(128, 1, 1);
    dim3 gridSize(width / blockSize.x, height / blockSize.y, 1);

    cudaEventRecord(start);

    render <<<gridSize, blockSize>>>(spp, width, height, outputGPU);
    cudaDeviceSynchronize();
    // Copy Mem from GPU to CPU
    cudaMemcpy(outputCPU, outputGPU, width * height * sizeof(float3), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // free CUDA memory
    cudaFree(outputGPU);

    float milliseconds = 0;

    cudaEventElapsedTime(&milliseconds, start, stop);

    //float fps = 1.0f / (milliseconds / 1000.0f);
    //printf("{\"spp\":%d, \"fps\": %d}\n", spp_in, static_cast<int>(fps));
    printf("{\"spp\":%d, \"time_ms\": %f}\n", spp_in, milliseconds);
    //save("test.png", width, height, outputCPU);

    //getchar();
    return 0;
}
