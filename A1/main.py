#!/usr/bin/env python

from itertools import count

from numpy import sum, exp, array, zeros, roll, where, mean, linspace, abs, log, sqrt, sinh, var
from numpy.random import choice, seed as srand, rand
from matplotlib import pyplot as plt

srand(1108)

## 2D Lattice generation and Ising Hamiltonian
def random_config(L):
	return choice([-1, 1], (L, L))

## Statistical Description of the Ising Model
# I cannot understand what is the "PDF" in this context.
# The probability distribution is over all configurations
# instead of over different values of a continuous random variable.
# I am interpreting the task as plotting the probability distribution of total magnetization.
def energy(config, J, B):
	return -J * (sum(config[1:, :]*config[:-1, :]) + sum(config[:, 1:]*config[:, :-1])) - B * sum(config)

def enumerate_configs(L):
	for i in range(1<<L**2):
		yield array([[1 - 2*(i >> j*L+k & 1) for j in range(L)] for k in range(L)])

def boltzmann_weight(config, J, B, T):
	return exp(-energy(config, J, B)/T)

def magnetization(config):
	return sum(config) / config.size

L = 4
J = 1
B = 0
T = 1

m = dict()
Z = 0
for config in enumerate_configs(L):
	weight = boltzmann_weight(config, J, B, T)
	mag = magnetization(config)
	if mag in m:
		m[mag] += weight
	else:
		m[mag] = weight
	Z += weight
for mag in m.keys():
	m[mag] /= Z

plt.bar(m.keys(), m.values(), width=1/L**2)
plt.xlabel('Magnetization')
plt.ylabel('Probability')
plt.show()

## Gibbs Sampler on Ising Model
def nearest_neighbors_sum(config):
	result = zeros(config.shape)
	shifted = roll(config, 1, 0)
	shifted[0, :] = 0
	result += shifted
	shifted = roll(config, -1, 0)
	shifted[-1, :] = 0
	result += shifted
	shifted = roll(config, 1, 1)
	shifted[:, 0] = 0
	result += shifted
	shifted = roll(config, -1, 1)
	shifted[:, -1] = 0
	result += shifted
	return result

# In the resulting array, the (i,j) element is the conditional probability
# of the (i,j) spin to be 1 in the next sample.
def conditional(config, J, B, T):
	nn_sum = nearest_neighbors_sum(config)
	weight_1 = exp(-(-J * nn_sum - B)/T)
	return weight_1 / (weight_1 + 1/weight_1)

def educated_random_config(L, J, B, T):
	mag = (1-sinh(2*J/T)**-4)**(1/8) if T/J < 2/log(1+sqrt(2)) else 0
	return where(rand(L, L) < (mag+1)/2, 1, -1)

## Gibbs Iteration
def gibbs_sampler(L, J, B, T, burn_in, taking_period=1):
	config = educated_random_config(L, J, B, T)
	for n in count(-burn_in):
		if n >= 0 and n % taking_period == 0:
			yield config
		config = where(rand(L, L) < conditional(config, J, B, T), 1, -1)

def take(n, iterable):
	for i, x in enumerate(iterable):
		if i == n:
			break
		yield x

L = 30
J = 1
B = 0
T = 1
for burn_in in [0, 100, 1000]:
	m = [magnetization(config) for config in take(100, gibbs_sampler(L, J, B, T, burn_in, 10))]
	plt.hist(m, bins=6, density=True)
	plt.xlabel('Magnetization')
	plt.ylabel('Probability density')
	plt.title('Burn-in = {}'.format(burn_in))
	plt.show()

## magnetization at different temperature
J = 1
B = 0
L_list = [10, 17, 25, 32, 40]
T_list = linspace(0.5, 3.5, 21)

full_cv = []
full_suscep = []
for L in L_list:
	mean_m = []
	cv = []
	suscep = []
	for T in T_list:
		m = []
		e = []
		for config in take(100, gibbs_sampler(L, J, B, T, 100, 10)):
			m.append(magnetization(config))
			e.append(energy(config, J, B))
		mean_m.append(mean(m))
		cv.append(var(e)/L**2/T**2)
		suscep.append(var(m)*L**2/T)
	plt.scatter(T_list, mean_m, label='$L = {}$'.format(L))
	full_cv.append(cv)
	full_suscep.append(suscep)

plt.vlines(2*J/log(1+sqrt(2)), 0, 1, color='r', linestyle='dashed')
plt.xlabel('Temperature')
plt.ylabel('Magnetization')
plt.legend()
plt.show()

## Classical Magnetic Field dependence: magnetization of the 2D Ising model
J = 1
B = 0
T = 3
B_list = linspace(-2, 2, 21)
for L in L_list:
	mag = []
	for B in B_list:
		m = [magnetization(config) for config in take(50, gibbs_sampler(L, J, B, T, 100, 10))]
		mag.append(mean(m))
	plt.scatter(B_list, mag, label='$L = {}$'.format(L))
plt.xlabel('External field')
plt.ylabel('Magnetization')
plt.legend()
plt.show()

## Specific Heat of the 2D Ising Model
for L, cv in zip(L_list, full_cv):
	plt.scatter(T_list, cv, label='$L = {}$'.format(L))
plt.xlabel('Temperature')
plt.ylabel('Specific heat')
plt.legend()
plt.show()

## Magnetic Susceptibility of the 2D Ising model
for L, suscep in zip(L_list, full_suscep):
	plt.scatter(T_list, suscep, label='$L = {}$'.format(L))
plt.xlabel('Temperature')
plt.ylabel('Magnetic susceptibility')
plt.legend()
plt.show()
