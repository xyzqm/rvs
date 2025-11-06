from rv import RV
from scipy import stats
from matplotlib import pyplot as plt
import numpy as np

uf = RV.uniform(0, 1)
# uf.display("Uniform")

# Example 1: Inverse cdf method to do exponential
exp = uf.apply(lambda a: -1 / 7 * np.log(1 - a))
exp.display("Exponential")

print("Exponential EV: ", exp.ev())
print("Exponential var: ", exp.var())

# Example 2: Inverse cdf method to do geometric RV
p = 0.25
geo = uf.apply(lambda a: np.ceil(np.log(1 - a) / np.log(1 - p)))
geo.display("Geometric")

print("Geometric EV: ", geo.ev())
print("Geometric var: ", geo.var())

# Example 3: Box-Muller transform
u1 = RV.uniform(low=0, high=1)
u2 = RV.uniform(low=0, high=1)

z1 = RV.sqrt(RV.log(u1) * -2) * RV.cos(u2 * (2 * np.pi))
z2 = RV.sqrt(RV.log(u1) * -2) * RV.sin(u2 * (2 * np.pi))

z1.display("Box-Muller 1")
z2.display("Box-Muller 2")

# Example 4: Central Limit Theorem (CLT)
def clt(cur, name):
    for i in range(10):
        cur = cur + cur
    cur.display("CLT with 1024 " + name)

clt(RV.uniform(0, 1), "uniform(0, 1)")
clt(exp, "exponential")
clt(RV.from_rv(stats.bernoulli(0.5)), "bernoulli")
clt(RV.from_rv(stats.poisson(0.5)), "poisson")
clt(RV.from_rv(stats.binom(n=10, p=0.2)), "binomial")

# Example 5: Rejection Sampling Method
# Take bounding box (assuming distribution finite support -- finite range), throw darts, see if land
f = lambda x: 2 * x + 0.5 * np.sin(4 * np.pi * x) # PDF, between -0.5 and 2.5
pdf = {
    'func': f,
    'range': (0, 1),
    'max': 2.5
}
rej = RV.from_pdf(pdf)
rej.display("Rejection Sampling")

# Show plot
plt.show()