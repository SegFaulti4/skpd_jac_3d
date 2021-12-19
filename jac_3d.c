#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define Max(a,b) ((a)>(b)?(a):(b))

double maxeps = 0.1e-7;
int itmax = 100;
int i, j, k;
double eps;
double A[N][N][N], B[N][N][N];

void relax();
void resid();
void init();
void verify();

int main(int argc, char **argv) {
    double avg_time = 0.;

    for (int rep_num = 0; rep_num < REPETITIONS; rep_num++) {
        #ifdef DEBUG
            printf("Started repetition: %d\n", rep_num);
        #endif

        int it;
        init();

        double start = omp_get_wtime();

        for (it = 1; it <= itmax; it++) {
            eps = 0.;
            relax();
            resid();

            #ifdef DEBUG
                if (!(it % 1000)) {
                    printf( "it=%5i   eps=%.15lf\n", it, eps);
                }
            #endif

            if (eps < maxeps) {
                break;
            }
        }
        double end = omp_get_wtime();

        #ifdef DEBUG
            printf("Iterations: %d\n", it);
            printf("Time: %fs\n", end - start);
        #endif

        avg_time += end - start;
        verify();
    }

    avg_time /= REPETITIONS;
    printf("%lf", avg_time);
    return 0;
}

void init() {
    for(i = 0; i <= N - 1; i++) {
        for (j = 0; j <= N - 1; j++) {
            for (k = 0; k <= N - 1; k++) {
                if (i == 0 || i == N - 1 || j == 0 || j == N - 1 || k == 0 || k == N - 1) {
                    A[i][j][k] = 0.;
                } else {
                    A[i][j][k] = (4. + i + j + k);
                }
            }
        }
    }
}

void relax() {
    for(i = 1; i <= N - 2; i++) {
        for (j = 1; j <= N - 2; j++) {
            for (k = 1; k <= N - 2; k++) {
                B[i][j][k] = (A[i - 1][j][k] + A[i + 1][j][k] + A[i][j - 1][k] +
                        A[i][j + 1][k] + A[i][j][k - 1] + A[i][j][k + 1]) / 6.;
            }
        }
    }
}

void resid() {
    for(i = 1; i <= N - 2; i++) {
        for (j = 1; j <= N - 2; j++) {
            for (k = 1; k <= N - 2; k++) {
                double e;
                e = fabs(A[i][j][k] - B[i][j][k]);
                A[i][j][k] = B[i][j][k];
                eps = Max(eps, e);
            }
        }
    }
}

void verify() {
    double s;
    s=0.;
    for(i = 0; i <= N - 1; i++) {
        for (j = 0; j <= N - 1; j++) {
            for (k = 0; k <= N - 1; k++) {
                s = s + A[i][j][k] * (i + 1) * (j + 1) * (k + 1) / (double)(N * N * N);
            }
        }
    }

    #ifdef DEBUG
        printf("\tS = %lf\n", s);
        printf("\teps = %lf\n", eps);
    #endif
}