import numpy as np

# Parameters
L = 20  # system size
T = 2.3 # temperature
J = 1.0 # interaction strength
kB = 1.0 # Boltzmann constant set to 1
steps = 100000 # number of spin flips attempts

# Initialize spins randomly
spins = np.random.choice([-1, 1], size=(L, L))

def energy(spins):
    E = 0
    # sum over each spin and its right & down neighbor (to avoid double counting)
    for i in range(L):
        for j in range(L):
            S = spins[i, j]
            # right neighbor with wrap-around
            SR = spins[i, (j+1)%L]
            # down neighbor with wrap-around
            SD = spins[(i+1)%L, j]
            E += -J * S * SR
            E += -J * S * SD
    return E

def energy_change(spins, i, j):
    # Calculate delta E if spin at (i,j) is flipped
    S = spins[i, j]
    up = spins[(i-1)%L, j]
    down = spins[(i+1)%L, j]
    left = spins[i, (j-1)%L]
    right = spins[i, (j+1)%L]

    # before flip: contribution = -J * S * (up+down+left+right)
    # after flip: contribution = -J * (-S) * (up+down+left+right)
    # deltaE = E_after - E_before = 2 * J * S * (up+down+left+right)
    neighbors_sum = up + down + left + right
    return 2 * J * S * neighbors_sum

# Metropolis updates
def metropolis_step(spins, T):
    # pick a random spin
    i = np.random.randint(L)
    j = np.random.randint(L)
    dE = energy_change(spins, i, j)
    if dE <= 0:
        # good flip
        spins[i, j] = -spins[i, j]
    else:
        # flip with probability exp(-dE/(kB*T))
        if np.random.rand() < np.exp(-dE/(kB*T)):
            spins[i, j] = -spins[i, j]

# Example simulation
# First, thermalize
for _ in range(100000):
    metropolis_step(spins, T)

# Now measure
E_samples = []
M_samples = []
for sweep in range(300000):
    metropolis_step(spins, T)
    if sweep % 10 == 0:
        currE = energy(spins)
        currM = np.mean(spins)
        E_samples.append(currE)
        M_samples.append(currM)

E_mean = np.mean(E_samples)
E2_mean = np.mean([e**2 for e in E_samples])
M_mean = np.mean(M_samples)
M2_mean = np.mean([m**2 for m in M_samples])

N = L*L
C = (E2_mean - E_mean**2) / (N * (T**2)) # specific heat
chi = (M2_mean - M_mean**2) / (N * T)    # susceptibility

print("At T=", T, "E=", E_mean, "C=", C, "M=", M_mean, "chi=", chi)
