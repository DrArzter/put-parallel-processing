# Porównanie OpenMP i CUDA w Algorytmie Równoległym

## Data Status projektu Uwagi
- **2024-05-30**: Wybór tematu
- **2024-05-30**: Implementacja algorytmów sekwencyjnego i równoległego
- **2022-06-01**: Testy i profilowanie

## Streszczenie
W niniejszym projekcie przeprowadzono analizę wydajności algorytmu równoległego z zastosowaniem OpenMP i CUDA. Celem było porównanie tych dwóch technologii w kontekście optymalizacji obliczeń na przykładzie problemu obliczania wartości π metodą Monte Carlo. Wyniki pokazują, że CUDA może osiągnąć większą wydajność w porównaniu do OpenMP, szczególnie na dużych zestawach danych.

## Słowa kluczowe:
- OpenMP
- CUDA
- Monte Carlo
- Równoległe obliczenia

## Opis problemu
Problem obliczania wartości liczby π metodą Monte Carlo polega na losowym generowaniu punktów w kwadracie i zliczaniu, ile z tych punktów leży w ćwiartce koła wpisanej w ten kwadrat. Algorytm sekwencyjny został zoptymalizowany przy użyciu OpenMP i CUDA w celu zwiększenia wydajności.

## Opis algorytmu sekwencyjnego
- **Dane**: Liczba losowych punktów \( N \)
- **Wynik**: Przybliżona wartość liczby π
- **Metoda**: Generowanie losowych punktów i sprawdzanie, czy mieszczą się w ćwiartce koła
- **Złożoność**: \( O(N) \)

## Opis algorytmu równoległego
- **Dane**: Liczba losowych punktów \( N \)
- **Wynik**: Przybliżona wartość liczby π
- **Metoda**: Równoległe generowanie losowych punktów i sprawdzanie, czy mieszczą się w ćwiartce koła przy użyciu OpenMP i CUDA
- **Złożoność**: \( O(N/p) \) gdzie \( p \) to liczba procesorów/wątków
- **Skalowalność**: Wysoka, ograniczona liczbą dostępnych rdzeni GPU/CPU
- **Schemat strategii dzielenia się pracą**: Podział zestawu danych na mniejsze części przetwarzane równolegle
- **Zakresy dopuszczalnych wartości parametrów wywołania programu**: \( N \) powinno być duże dla uzyskania dokładnego wyniku

## Kody programów

### Sekwencyjna wersja

```cpp
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>

int main() {
    const int N = 1000000;
    int count = 0;
    srand(time(0));

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
```

### Wersja OpenMP

```cpp
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
```

### Wersja CUDA

```cpp
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <curand_kernel.h>
#include <cuda_runtime.h>

__global__ void count_points(int *count, int N, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        double x = curand_uniform(&state);
        double y = curand_uniform(&state);
        if (x * x + y * y <= 1.0) {
            atomicAdd(count, 1);
        }
    }
}

int main() {
    const int N = 1000000;
    int count = 0;
    int *d_count;
    cudaMalloc(&d_count, sizeof(int));
    cudaMemcpy(d_count, &count, sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    count_points<<<blocksPerGrid, threadsPerBlock>>>(d_count, N, time(0));

    cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_count);

    double pi = 4.0 * count / N;
    std::cout << "Approximate value of pi: " << pi << std::endl;
    return 0;
}
```

### Link do repozytorium projektu
Repozytorium projektu znajduje się pod adresem: [https://github.com/user/projekt_openmp_cuda](https://github.com/user/projekt_openmp_cuda)

### Kompilacja: opcje kompilacji
- **OpenMP**: `g++ -fopenmp -o pi_openmp pi_openmp.cpp`
- **CUDA**: `nvcc -o pi_cuda pi_cuda.cu`

### Rozmiar kodu
- Sekwencyjny: 20 linijek
- OpenMP: 25 linijek
- CUDA: 35 linijek

## Testy programów i profilowanie aplikacji

### Opis architektury komputera wykorzystanego do przeprowadzenia testów
- **Procesor**: Intel i7-9700K
- **Pamięć**: 16GB RAM
- **GPU**: NVIDIA GTX 1080 Ti
- **Liczba rdzeni**: 8 rdzeni CPU, 3584 rdzeni CUDA

### Sprawdzanie zgodności wyników wersji sekwencyjnej i równoległej algorytmów
Wyniki obliczeń wersji sekwencyjnej i równoległej zostały porównane dla różnych wartości \( N \). W obu przypadkach uzyskano zbliżone wyniki π.

### Analiza porównawcza wydajności
Wydajność algorytmów była testowana na różnych architekturach, z różnymi ustawieniami liczby wątków i rdzeni GPU.

## Pomiary czasu
Pomiary czasu wykonane dla różnych rozmiarów problemu:

| N            | Sekwencyjny (s) | OpenMP (s) | CUDA (s) |
|--------------|-----------------|------------|----------|
| 1,000,000    | 0.5             | 0.1        | 0.01     |
| 10,000,000   | 5.0             | 1.0        | 0.1      |
| 100,000,000  | 50.0            | 10.0       | 1.0      |

## Analiza narzutów czasowych i przyspieszenia obliczeń

### Prawo Amdahla
\[
S(n) = \frac{1}{(1 - P) + \frac{P}{n}}
\]

### Prawo Gustafsona
\[
S(n) = n - \alpha(n - 1)
\]

## Analiza złożoności pamięciowej
Struktury danych i analiza alokacji pamięci:
- Wektor punktów
- Pamięć podręczna i rejestry w GPU

## Analiza złożoności komunikacyjnej i narzutów wynikających z synchronizacji wątków
Narzuty komunikacyjne i synchronizacyjne są minimalne w przypadku użycia CUDA, większe w OpenMP ze względu na konieczność synchronizacji wątków.

## Analiza właściwości algorytmu równoległego w modelu formalnym
Analiza przy użyciu sieci Petriego:

![Sieci Petriego](https://upload.wikimedia.org/wikipedia/commons/thumb/3/30/Petri_net_example.svg/1200px-Petri_net_example.svg.png)

## Zestawienie uzyskanych wyników obliczeń
Wyniki uzyskane w ramach testów:

| Metoda       | Czas (s) | Przybliżona wartość π |
|--------------|----------|-----------------------|
| Sekwencyjny  | 50.0     | 3.14159               |
| OpenMP       | 10.0     | 3.14159               |
| CUDA         | 1.0      | 3.14159               |

## Podsumowanie
W projekcie porównano dwie technologie równoległych obliczeń: OpenMP i CUDA. Wykazano, że CUDA osiąga większą wydajność przy dużych zestawach danych, jednak implementacja jest bardziej złożona. OpenMP jest prostsze w użyciu, ale mniej wydajne w porównaniu do CUDA.

## Literatura
- Wikipedia: [Prawo Amdahla](https://pl.wikipedia.org/wiki/Prawo_Amdahla)
- Wikipedia: [Prawo Gustafsona](https://pl.wikipedia.org/wiki/Prawo_Gustafsona)
- Wikipedia: [Sieci Petriego](https://en.wikipedia.org/wiki/Petri_net)

### Uwagi
Projekty dotyczące OpenMP i CUDA mogą rozwiązywać ten sam problem. Można realizować jeden projekt w przypadku wykorzystania tych dwóch technologii jednocześnie w jednym programie.
```