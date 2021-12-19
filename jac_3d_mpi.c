#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#define  Max(a,b) ((a)>(b)?(a):(b))

double maxeps = 0.1e-7;
int itmax = 100;
int i, j, k;
double eps, sum;
double A[N][N][N], B[N][N][N];

void relax();
void resid();
void init();
void verify();

int wrank, wsize;
int block, startrow, lastrow;
void update();

int main(int argc, char **argv) {
    double avg_time = 0.;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
    MPI_Comm_size(MPI_COMM_WORLD, &wsize);
    MPI_Barrier(MPI_COMM_WORLD); // wait for all process here

    for (int rep_num = 0; rep_num < REPETITIONS; rep_num++) {
        if (!wrank) {
#ifdef DEBUG
            printf("Started repetition: %d\n", rep_num);
#endif
        }

        int it;
        init();

        double start, end;

        if (!wrank) {
            start = MPI_Wtime();
#ifdef DEBUG
            printf("wsize: %d\n", wsize);
#endif
        }
        block = (N - 2) / wsize;
        startrow = block * wrank + 1;
        lastrow = startrow + block - 1;

        for (it = 1; it <= itmax; it++) {
            eps = 0.;
            relax();
            resid();
            update();

            if (!wrank) {
#ifdef DEBUG
                if (!(it % 1000)) {
                    printf("it=%5i   eps=%.15lf\n", it, eps);
                }
#endif
            }

            if (eps < maxeps) {
                break;
            }
        }
        if (!wrank) {
            end = MPI_Wtime();
            avg_time += end - start;
#ifdef DEBUG
            printf("Iterations: %d\n", it);
            printf("Time: %fs\n", end - start);
#endif
        }
        verify();

        MPI_Barrier(MPI_COMM_WORLD);
    }
    if (!wrank) {
        avg_time /= REPETITIONS;
        printf("%lf", avg_time);
    }

    MPI_Finalize();

    return 0;
}

void init() {
    for (i = 0; i <= N - 1; i++) {
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
    for (i = startrow; i <= lastrow; i++) {
        for (j = 1; j <= N - 2; j++) {
            for (k = 1; k <= N - 2; k++) {
                B[i][j][k] = (A[i - 1][j][k] + A[i + 1][j][k] + A[i][j - 1][k] +
                              A[i][j + 1][k] + A[i][j][k - 1] + A[i][j][k + 1]) / 6.;
            }
        }
    }
}

void resid() {
    int start_flag = startrow == 0 ? 1 : 0;
    int last_flag = lastrow == N - 1 ? 1 : 0;

    startrow = start_flag ? startrow + 1 : startrow;
    lastrow = last_flag ? lastrow - 1 : lastrow;

    double local_eps = eps;

    for (i = startrow; i <= lastrow; i++) {
        for (j = 1; j <= N - 2; j++) {
            for (k = 1; k <= N - 2; k++) {
                double e;
                e = fabs(A[i][j][k] - B[i][j][k]);
                A[i][j][k] = B[i][j][k];
                local_eps = Max(local_eps, e);
            }
        }
    }

    MPI_Allreduce(&local_eps, &eps, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    startrow = start_flag ? startrow - 1 : startrow;
    lastrow = last_flag ? lastrow + 1 : lastrow;
}

void verify() {
    double s = 0.0;
    for (i = startrow; i <= lastrow; i++) {
        for (j = 0; j <= N - 1; j++) {
            for (k = 0; k <= N - 1; k++) {
                s = s + A[i][j][k] * (i + 1) * (j + 1) * (k + 1) / (N * N * N);
            }
        }
    }

    MPI_Allreduce(&s, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (!wrank) {
#ifdef DEBUG
        printf("\tS = %lf\n", sum);
        printf("\teps = %lf\n", eps);
#endif
    }
}

void update() {
    MPI_Request request[4];
    MPI_Status status[4];

    if (wrank) {
        MPI_Irecv(&A[startrow - 1][0][0], N * N, MPI_DOUBLE, wrank - 1, 1215, MPI_COMM_WORLD, &request[0]);
        MPI_Isend(&A[startrow][0][0], N * N, MPI_DOUBLE, wrank - 1, 1216, MPI_COMM_WORLD, &request[1]);
    }
    if (wrank != wsize - 1) {
        MPI_Isend(&A[lastrow][0][0], N * N, MPI_DOUBLE, wrank + 1, 1215, MPI_COMM_WORLD, &request[2]);
        MPI_Irecv(&A[lastrow + 1][0][0], N * N, MPI_DOUBLE, wrank + 1, 1216, MPI_COMM_WORLD, &request[3]);
    }

    int ll = 4, shift = 0;
    if (!wrank) {
        ll -= 2;
        shift = 2;
    }
    if (wrank == wsize - 1) {
        ll -= 2;
    }
    if (ll) {
        MPI_Waitall(ll, &request[shift], status);
    }
}
