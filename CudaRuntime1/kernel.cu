#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>

// Estructura para representar un punto en 2D
struct Punto {
    float x, y;
    __host__ __device__ Punto(float _x = 0, float _y = 0) : x(_x), y(_y) {}
};

// Estructura para representar un vector en 2D
struct Vector {
    float x, y;
    __host__ __device__ Vector(float _x = 0, float _y = 0) : x(_x), y(_y) {}
    __host__ __device__ Vector(Punto p1, Punto p2) : x(p2.x - p1.x), y(p2.y - p1.y) {}
};

// Función para comparar dos floats con una precisión dada
__device__ bool iguales(float a, float b, float eps = 1e-3) {
    return fabs(a - b) < eps;
}

// Función para calcular la distancia entre dos puntos
__device__ float distancia(float x1, float y1, float x2, float y2) {
    float dx = x1 - x2;
    float dy = y1 - y2;
    return hypotf(dx, dy);
}

// Función para calcular la intersección de dos círculos
__device__ void interseccion(float x0, float y0, float r0, float x1, float y1, float r1, float& x3, float& y3, float& x4, float& y4) {
    float d = distancia(x0, y0, x1, y1);
    float a = (r0 * r0 - r1 * r1 + d * d) / (2 * d);
    float h = sqrtf(r0 * r0 - a * a);
    float x2 = x0 + a * (x1 - x0) / d;
    float y2 = y0 + a * (y1 - y0) / d;
    x3 = x2 + h * (y1 - y0) / d;
    y3 = y2 - h * (x1 - x0) / d;
    x4 = x2 - h * (y1 - y0) / d;
    y4 = y2 + h * (x1 - x0) / d;
}

// Kernel CUDA para calcular las intersecciones de los círculos de Voronoi
__global__ void voronoi(Punto* comb, Punto* puntos, float* x, float* y, int numComb, int numPuntos, int PPT) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x; // ID global del hilo
    int radio = 3 * gid; // Radio de los círculos
    int i = 0;

    if (gid < numComb) {
        for (int j = 0; j < numComb; ++j) {
            Punto C1 = comb[2 * j];
            Punto C2 = comb[2 * j + 1];
            bool check1 = true, check2 = true;

            if (distancia(C1.x, C1.y, C2.x, C2.y) <= radio * 2) {
                float x1, y1, x2, y2;
                interseccion(C1.x, C1.y, radio, C2.x, C2.y, radio, x1, y1, x2, y2);

                for (int k = 0; k < numPuntos; ++k) {
                    Punto P = puntos[k];
                    if (!(P.x == C1.x && P.y == C1.y) && !(P.x == C2.x && P.y == C2.y)) {
                        if (distancia(x1, y1, P.x, P.y) < distancia(x1, y1, C1.x, C1.y))
                            check1 = false;
                        if (distancia(x2, y2, P.x, P.y) < distancia(x2, y2, C1.x, C1.y))
                            check2 = false;
                    }
                }

                if (check1) {
                    x[gid * PPT + i] = x1;
                    y[gid * PPT + i] = y1;
                    i++;
                }
                if (check2) {
                    x[gid * PPT + i] = x2;
                    y[gid * PPT + i] = y2;
                    i++;
                }
            }
        }
    }
}

int main() {
    const int cantidad = 50; // Número de puntos
    const int PPT = 200; // Puntos por thread
    const int radius = 100; // Número de hilos

    // Generar puntos aleatorios
    std::vector<Punto> Puntos(cantidad);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1000);

    for (int i = 0; i < cantidad; ++i) {
        Puntos[i] = Punto(dis(gen), dis(gen));
    }

    // Crear combinaciones de pares de puntos
    std::vector<std::pair<Punto, Punto>> combinations;
    for (size_t i = 0; i < Puntos.size(); ++i) {
        for (size_t j = i + 1; j < Puntos.size(); ++j) {
            combinations.emplace_back(Puntos[i], Puntos[j]);
        }
    }

    // Copiar combinaciones a un vector de Thrust en el host
    thrust::host_vector<Punto> h_combinations(combinations.size() * 2);
    for (size_t i = 0; i < combinations.size(); ++i) {
        h_combinations[2 * i] = combinations[i].first;
        h_combinations[2 * i + 1] = combinations[i].second;
    }

    // Copiar puntos a un vector de Thrust en el host
    thrust::host_vector<Punto> h_Puntos = Puntos;

    // Copiar datos del host a la GPU
    thrust::device_vector<Punto> d_combinations = h_combinations;
    thrust::device_vector<Punto> d_Puntos = h_Puntos;
    thrust::device_vector<float> d_x(combinations.size() * PPT, 0);
    thrust::device_vector<float> d_y(combinations.size() * PPT, 0);

    // Lanzar kernel CUDA
    voronoi << <1, radius >> > (
        thrust::raw_pointer_cast(d_combinations.data()),
        thrust::raw_pointer_cast(d_Puntos.data()),
        thrust::raw_pointer_cast(d_x.data()),
        thrust::raw_pointer_cast(d_y.data()),
        combinations.size(),
        Puntos.size(),
        PPT
        );

    // Copiar resultados de vuelta al host
    thrust::host_vector<float> h_x = d_x;
    thrust::host_vector<float> h_y = d_y;

    // Crear una imagen usando OpenCV
    cv::Mat image(1000, 1000, CV_8UC3, cv::Scalar(255, 255, 255));

    // Dibujar los puntos originales
    for (int i = 0; i < cantidad; ++i) {
        cv::circle(image, cv::Point(Puntos[i].x, Puntos[i].y), 5, cv::Scalar(0, 0, 0), -1);
    }

    // Dibujar las intersecciones
    for (int i = 0; i < h_x.size(); ++i) {
        if (h_x[i] != 0 || h_y[i] != 0) {
            cv::circle(image, cv::Point(h_x[i], h_y[i]), 3, cv::Scalar(0, 0, 255), -1);
        }
    }

    // Guardar la imagen como archivo PNG
    cv::imwrite("voronoi.png", image);
    std::cout << "Image saved as voronoi.png" << std::endl;

    return 0;
}
