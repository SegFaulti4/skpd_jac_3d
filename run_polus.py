import subprocess
import logging
import sys

logging.basicConfig(level=logging.DEBUG)

DATA_SIZES = [i + 2 for i in range(16, 129, 8)]
PARALLEL_NUMS = [1, 2, 4, 8] + [i + 2 for i in range(16, 129, 8)]
REPETITIONS = 12
EXPERIMENTS = ['single', 'omp', 'mpi']

COMPILERS = {'single': ['g++', 'jac_3d.c', '-fopenmp'],
             'omp': ['g++', 'jac_3d_omp.c', '-fopenmp'],
             'mpi': ['mpicxx', 'jac_3d_mpi.c']}


def collect(experiment, parallel_num):
    pr = None

    if experiment == 'mpi':
        pr = subprocess.Popen(['mpiexec', '-np', str(parallel_num), './a.out'],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if experiment == 'omp':
        pr = subprocess.Popen(['./a.out', str(parallel_num)],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if experiment == 'single':
        pr = subprocess.Popen(['./a.out'],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        r = float(pr.communicate()[0])
    except ValueError:
        r = 0.0

    pr.wait()
    return r


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in EXPERIMENTS:
        logging.error('Wrong arguments')
        sys.exit(1)
    experiment = sys.argv[1]

    results = []
    for data_size in DATA_SIZES:
        results.append(list())
        proc = subprocess.Popen(COMPILERS[experiment] + ['-DN=' + str(data_size), '-DREPETITIONS=' + str(REPETITIONS)])
        proc.wait()
        logging.info('Compiled for data_size=' + str(data_size))
        for parallel_num in PARALLEL_NUMS:
            logging.info('\tCollecting for parallel_num=' + str(parallel_num))
            results[-1].append(collect(experiment, parallel_num))
            logging.info('\t\tExecuted in ' + str(results[-1][-1]) + 's')
            if experiment == 'single':
                break

    outF = open(experiment + '_res.csv', 'w')
    for res in results:
        outF.write(','.join(map(str, res)) + '\n')
    outF.close()
