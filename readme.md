# Porównanie OpenMP i CUDA w Algorytmie Równoległym

## Opis projektu
Projekt ma na celu porównanie wydajności algorytmu równoległego z zastosowaniem OpenMP i CUDA na przykładzie problemu obliczania wartości liczby π metodą Monte Carlo. 

## Struktura projektu
- `pi_seq.cpp`: Sekwencyjna wersja algorytmu
- `pi_openmp.cpp`: Równoległa wersja z OpenMP
- `pi_cuda.cu`: Równoległa wersja z CUDA

## Kompilacja i uruchomienie

### OpenMP
- g++ -fopenmp -o pi_openmp pi_openmp.cpp
- ./pi_openmp

### CUDA

- nvcc -o pi_cuda pi_cuda.cu
- ./pi_cuda

### Sekwencyjna wersja
```sh
g++ -o pi_seq pi_seq.cpp
./pi_seq
