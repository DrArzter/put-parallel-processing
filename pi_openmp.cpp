#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <omp.h>

int main() {
    const int N = 1000000;
    int count = 0;
    srand(time(0));

    #pragma omp parallel for reduction(+:count)
    for (int i = 0; i < N; i++) {
        double x = static_cast<double>(rand()) / RAND_MAX;
        double y = static_cast<double>(rand()) / RAND_MAX;
        if (x * x + y * y <= 1.0) {
            count++;
        }
    }

    double pi = 4.0 * count / N;
    std::cout << "Approximate value of pi: " << pi << std::endl;
    return 0;
}
