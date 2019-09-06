CFLAGS = -O3 -march=native -fopenmp-simd -ffp-contract=fast -Wall
LDFLAGS = -lm

dot : dot.c
