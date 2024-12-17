# ising_helpers.pyx

# Cython directives for optimization
# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False
# cython: language_level=3, boundscheck=False

import numpy as np
cimport numpy as np
from libc.math cimport exp
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.time cimport time
from libc.stdint cimport int32_t  # Import int32_t

# Define module-level constants
cdef double J = 1.0
cdef double kB = 1.0

cdef double energy_change(int32_t[:, :] spins, int L, int i, int j):
    cdef int S = spins[i, j]
    cdef int up = spins[i - 1 if i > 0 else L - 1, j]
    cdef int down = spins[i + 1 if i < L - 1 else 0, j]
    cdef int left = spins[i, j - 1 if j > 0 else L - 1]
    cdef int right = spins[i, j + 1 if j < L - 1 else 0]
    cdef int neighbors_sum = up + down + left + right
    return 2.0 * J * S * neighbors_sum

cdef double compute_energy(int32_t[:, :] spins, int L):
    cdef double E = 0.0
    cdef int i, j, S, SR, SD
    for i in range(L):
        for j in range(L):
            S = spins[i, j]
            SR = spins[i, j + 1 if j < L - 1 else 0]
            SD = spins[i + 1 if i < L - 1 else 0, j]
            E += -J * S * SR
            E += -J * S * SD
    return E

cdef double compute_magnetization(int32_t[:, :] spins, int L):
    cdef double M = 0.0
    cdef int i, j
    for i in range(L):
        for j in range(L):
            M += spins[i, j]
    return M

cpdef void metropolis_step(int32_t[:, :] spins, int L, double T):
    """
    Perform a single Metropolis step.
    """
    cdef int i = rand() % L
    cdef int j = rand() % L
    cdef double dE = energy_change(spins, L, i, j)
    
    cdef double r
    cdef double acceptance_threshold
    
    if dE <= 0:
        spins[i, j] = -spins[i, j]
    else:
        r = rand() / <double>RAND_MAX
        acceptance_threshold = exp(-dE / (kB * T))
        if r < acceptance_threshold:
            spins[i, j] = -spins[i, j]

cpdef double total_energy(int32_t[:, :] spins, int L):
    return compute_energy(spins, L)

cpdef double total_magnetization(int32_t[:, :] spins, int L):
    return compute_magnetization(spins, L)

cpdef np.ndarray[int32_t, ndim=2] initialize_spins(int L):
    """
    Initialize spins randomly: returns a typed memoryview for maximum Cython efficiency.
    """
    cdef np.ndarray[int32_t, ndim=2] spins = np.random.choice([-1, 1], size=(L, L)).astype(np.int32)
    return spins

cpdef tuple simulate_measurement(int32_t[:, :] spins, int L, double T, long sweeps, int measure_interval):
    """
    Perform measurement sweeps and collect energy and magnetization samples.
    """
    cdef list E_samples = []
    cdef list M_samples = []
    cdef long sweep
    cdef double E, M
    for sweep in range(sweeps):
        metropolis_step(spins, L, T)
        if sweep % measure_interval == 0:
            E = compute_energy(spins, L)
            M = abs(compute_magnetization(spins, L))
            E_samples.append(E)
            M_samples.append(M)
    return E_samples, M_samples

cpdef void simulate_thermalization(int32_t[:, :] spins, int L, double T, long sweeps):
    """
    Perform thermalization sweeps.
    """
    cdef long sweep
    for sweep in range(sweeps):
        metropolis_step(spins, L, T)

cpdef void seed_rng_custom(unsigned int seed):
    """
    Seed the random number generator with a custom seed.
    """
    srand(seed)
