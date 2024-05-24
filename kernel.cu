#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <cmath>
#include "./include/lodepng.h"

#define WIDTH 1024
#define HEIGHT 1024
#define NUM_SEEDS 10

struct Point {
    float x, y;
};

__global__ void voronoiKernel(Point* seeds, int num_seeds, unsigned char* output) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * WIDTH + x;

    if (x >= WIDTH || y >= HEIGHT) return;

    float minDist = 1e20;
    int minSeedIdx = -1;

    for (int i = 0; i < num_seeds; ++i) {
        float dist = (x - seeds[i].x) * (x - seeds[i].x) + (y - seeds[i].y) * (y - seeds[i].y);
        if (dist < minDist) {
            minDist = dist;
            minSeedIdx = i;
        }
    }

    int color = (minSeedIdx % 256);  // Simple coloring based on seed index
    output[4 * idx + 0] = color;
    output[4 * idx + 1] = color;
    output[4 * idx + 2] = color;
    output[4 * idx + 3] = 255;
}

void saveImage(const std::vector<unsigned char>& image, const std::string& filename) {
    unsigned error = lodepng::encode(filename, image, WIDTH, HEIGHT);
    if (error) {
        std::cerr << "Error encoding PNG: " << lodepng_error_text(error) << std::endl;
    }
}

int main() {
    std::vector<Point> h_seeds(NUM_SEEDS);
    std::vector<unsigned char> h_output(WIDTH * HEIGHT * 4);

    // Initialize seeds with some values
    for (int i = 0; i < NUM_SEEDS; ++i) {
        h_seeds[i] = { float(rand() % WIDTH), float(rand() % HEIGHT) };
    }

    Point* d_seeds;
    unsigned char* d_output;

    cudaMalloc(&d_seeds, NUM_SEEDS * sizeof(Point));
    cudaMalloc(&d_output, WIDTH * HEIGHT * 4 * sizeof(unsigned char));

    cudaMemcpy(d_seeds, h_seeds.data(), NUM_SEEDS * sizeof(Point), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, (HEIGHT + blockSize.y - 1) / blockSize.y);

    voronoiKernel << <gridSize, blockSize >> > (d_seeds, NUM_SEEDS, d_output);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output.data(), d_output, WIDTH * HEIGHT * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Save the image using lodepng
    saveImage(h_output, "voronoi.png");

    // Clean up
    cudaFree(d_seeds);
    cudaFree(d_output);

    return 0;
}
