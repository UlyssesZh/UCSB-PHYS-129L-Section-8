#!/usr/bin/env python

from math import factorial

from numpy import sum, linspace, zeros_like, vectorize, cos, pi, arange, array, isclose, sqrt, arccos, sin, geomspace, meshgrid, log10, exp, log
from scipy.special import roots_legendre
from scipy.integrate import fixed_quad, quad, romberg
from matplotlib import pyplot as plt

## B1
# A
# B
class Quad:
	def __init__(self, func=lambda x: zeros_like(x), lower=0, upper=1):
		self.func = func
		self.lower = lower
		self.upper = upper

	def midpoint(self, n=1000):
		points = linspace(self.lower, self.upper, n+1)
		midpoints = (points[:-1] + points[1:])/2
		segments = points[1:] - points[:-1]
		return sum(self.func(midpoints) * segments)

	def trapezoidal(self, n=1000):
		points = linspace(self.lower, self.upper, n+1)
		y = self.func(points)
		segments = points[1:] - points[:-1]
		return sum((y[1:] + y[:-1])/2 * segments)

	def simpson(self, n=1000):
		points = linspace(self.lower, self.upper, n+1)
		y = self.func(points)
		midpoints = (points[:-1] + points[1:])/2
		segments = points[1:] - points[:-1]
		return sum((y[:-1] + 4*self.func(midpoints) + y[1:]) * segments/6)

# C
# See README.md

class GaussQuad(Quad):
	def gauss_legendre(self, n=5):
		points, weights = legendre_roots(n)
		points = (self.upper - self.lower)/2 * points + (self.upper + self.lower)/2
		return sum(self.func(points) * weights) * (self.upper - self.lower)/2

# D
def horner(coeffs, x):
	result = 0
	for coeff in reversed(coeffs):
		result = result * x + coeff
	return result

def legendre(x, n):
	if x < 0:
		return (-1)**n * legendre(-x, n)
	if n > 50:
		if x == 1:
			return 1.0
		theta = arccos(x)
		return cos((n+1/2)*theta-pi/4)*sqrt(2/pi/n/sin(theta))
	if x > 3/4:
		coef = [(-1)**k * 2**(n-k) * factorial(n+k) // factorial(k)**2 // factorial(n-k) for k in range(n+1)]
		return horner(coef, 1-x) / 2**n
	coef = [(-1)**k * factorial(2*(n-k)) // factorial(k) // factorial(n-k) // factorial(n-2*k) for k in reversed(range(n//2+1))]
	result = horner(coef, x**2)
	if n % 2:
		result *= x
	return result / 2**n
legendre = vectorize(legendre)

x = linspace(-1, 1, 201)
for M in [1, 2, 3, 4, 5]:
	plt.plot(x, legendre(x, M), label=f'$M={M}$')
plt.ylim(-1, 1)
plt.xlim(-1, 1)
plt.legend()
plt.show()

# E
def legendre_drv(x, n):
	return n/(x**2-1) * (x*legendre(x, n) - legendre(x, n-1))

legendre_roots_cache = {}
def legendre_roots(n):
	if n in legendre_roots_cache:
		return legendre_roots_cache[n]
	if n == 0:
		return array([])
	x = (-1+n**-2.0/8-n**-3.0/8) * cos(pi/(4*n+2)*(4*arange(n)+3)) # Francesco Tricomi
	for _ in range(2):
		x -= legendre(x, n)/legendre_drv(x, n)
	w = 2/(1-x**2)/legendre_drv(x, n)**2
	legendre_roots_cache[n] = x, w
	return x, w

for M in [1, 2, 3, 4, 5]:
	roots, weights = legendre_roots(M)
	roots1, weights1 = roots_legendre(M) # SciPy
	if isclose(roots, roots1).all() and isclose(weights, weights1).all():
		print(f'M={M}: my implementation matches SciPy')
	else:
		print(f'M={M}: my implementation does not match SciPy')

## B2
# A
def error(true_value, approx_value):
	return 2*(true_value - approx_value) / (true_value + approx_value)

def test_quad(func, lower, upper, n, true_value):
	quad = GaussQuad(func, lower, upper)
	return [
		error(true_value, quad.midpoint(n)),
		error(true_value, quad.trapezoidal(n)),
		error(true_value, quad.simpson(n)),
		error(true_value, quad.gauss_legendre(n))
	]

def plot_test_quad(func, true_val):
	n_list, k_list = meshgrid(geomspace(10, 10000, 4, dtype=int), linspace(0, 10, 11))
	errors = [zeros_like(k_list), zeros_like(k_list), zeros_like(k_list), zeros_like(k_list)]
	for i in range(k_list.shape[0]):
		for j in range(k_list.shape[1]):
			k = k_list[i, j]
			n = n_list[i, j]
			f = lambda x: func(x, k)
			true_value = true_val(k)
			midpoint, trapezoidal, simpson, gauss_legendre = test_quad(f, 0, 1, n, true_value)
			errors[0][i, j] = midpoint
			errors[1][i, j] = trapezoidal
			errors[2][i, j] = simpson
			errors[3][i, j] = gauss_legendre

	for error, method in zip(errors, ['Midpoint', 'Trapezoidal', 'Simpson', 'Gauss-Legendre']):
		plt.colorbar(plt.pcolor(log10(n_list), k_list, log10(abs(error))))
		plt.xlabel('$\\log_{10}N$')
		plt.ylabel('$k$')
		plt.title(f'{method} (log10 |error|; white means zero error)')
		plt.show()

plot_test_quad(lambda x, k: x**k, lambda k: 1/(k+1))

# B
plot_test_quad(lambda x, k: 1/(1+exp(-k*x)), lambda k: (log(exp(k)+1)-log(2))/k)

## B3
# A
def oscillator_period(a, n=5):
	return sqrt(8)*GaussQuad(lambda x: 1/sqrt(a**4-x**4), 0, a).gauss_legendre(n)

# B
def oscillator_period_fixed_quad(a, n=5):
	return sqrt(8)*fixed_quad(lambda x: 1/sqrt(a**4-x**4), 0, a, n=n)[0]

print('scipy.integrate.fixed_quad:')
n = 5
old_period = oscillator_period(2, n)
while True:
	new_period = oscillator_period(2, n*2)
	difference = new_period - old_period
	print(f'{n=}: {new_period=}, {difference=}')
	if abs(difference) < 1e-4:
		break
	old_period = new_period
	n *= 2

# C
quad_period, err = quad(lambda x: sqrt(8)/sqrt(2**4-x**4), 0, 2)
print(f'scipy.integrate.quad: period: {quad_period}, error: {err}')
# The error is much smaller with much faster evaluation

# D
period = romberg(lambda x: sqrt(8)/sqrt(2**4-x**4), 0, 2)
print(f'scipy.integrate.romberg: period: {period}')
# Because there is a singularity at x=2, where the sample is infinity.

# E
romberg_period = romberg(lambda x: sqrt(8)/sqrt(2**4-x**4), 0, 2-1e-5, tol=1e-5, show=True, divmax=10)
print(f'period: {romberg_period}, error: {romberg_period-quad_period}')

# F
for divmax in range(10, 16):
	romberg_period = romberg(lambda x: sqrt(8)/sqrt(2**4-x**4), 0, 2-1e-5, tol=1e-5, divmax=divmax)
	error = romberg_period - quad_period
	print(f'{divmax=}: {romberg_period=}, {error=}')

# G
a_list = linspace(0.02, 2, 100)
period_list = [quad(lambda x: sqrt(8)/sqrt(a**4-x**4), 0, a) for a in a_list]
plt.plot(a_list, period_list)
plt.xlabel('$a$')
plt.ylabel('Period')
plt.show()
