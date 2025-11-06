import numpy as np
from numbers import Number
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class RV:
    def __init__(self, rv=None, fn=None, lhs=None, rhs=None, cval=None, pdf=None):
        self.rv = rv
        self.fn = fn
        self.lhs = lhs
        self.rhs = rhs
        self.cval = cval
        self.pdf = pdf

    @classmethod
    def from_rv(cls, rv):
        return cls(rv=rv)

    @classmethod
    def uniform(cls, low, high):
        return cls(rv=stats.uniform(loc=low, scale=high - low))

    @classmethod
    def from_operation(cls, fn, lhs, rhs):
        return cls(fn=fn, lhs=lhs, rhs=rhs)

    @classmethod
    def from_constant(cls, value):
        return cls(fn=None, lhs=None, rhs=None, cval=value)

    @classmethod
    def from_pdf(cls, pdf):
        return cls(pdf=pdf)

    def sample(self, samples=100000):
        if self.pdf is not None:
            # Rejection sampling
            x_min, x_max = self.pdf['range']
            y_max = self.pdf['max']
            samples_collected = []
            while len(samples_collected) < samples:
                x_samples = np.random.uniform(x_min, x_max, samples)
                y_samples = np.random.uniform(0, y_max, samples)
                for x, y in zip(x_samples, y_samples):
                    if y <= self.pdf['func'](x):
                        samples_collected.append(x)
                    if len(samples_collected) >= samples:
                        break
            return np.array(samples_collected[:samples])

        if (self.cval is None) == False:
            return np.full(samples, self.cval)

        if self.rv is None:
            left_samples = self.lhs.sample(samples)
            if self.rhs is None:
                return self.fn(left_samples)
            else:
                right_samples = self.rhs.sample(samples)
                return self.fn(left_samples, right_samples)

        return self.rv.rvs(samples)
    
    def ev(self):
        return np.mean(self.sample())
    
    def var(self):
        return np.var(self.sample())

    def display(self, title="RV Distribution"):
        plt.figure(figsize=(10, 6))
        # Plot histogram
        plt.hist(self.sample(), bins=200, edgecolor='black', density=True)  # 'bins' controls number of bars
        plt.title(title)
        plt.xlabel("Value")
        plt.ylabel("Density")
        # sns.kdeplot(self.sample(), color='red', linestyle='--', linewidth=2, label='KDE (Smooth)')

    def __add__(self, other) -> 'RV':
        if isinstance(other, Number):
            return self.apply(lambda a: a + other)

        return RV.from_operation(lambda a, b: a + b, self, other)

    def __sub__(self, other) -> 'RV':
        if isinstance(other, Number):
            return self.apply(lambda a: a - other)
        return RV.from_operation(lambda a, b: a - b, self, other)

    def __mul__(self, other) -> 'RV':
        if isinstance(other, Number):
            return self.apply(lambda a: a * other)
        return RV.from_operation(lambda a, b: a * b, self, other)
    
    def __div__(self, other) -> 'RV':
        if isinstance(other, Number):
            return self.apply(lambda a: a / other)
        return RV.from_operation(lambda a, b: a / b, self, other)
    
    def apply(self, fn) -> 'RV':
        return RV.from_operation(fn, self, None)

    @staticmethod
    def min(a: 'RV', b: 'RV') -> 'RV':
        return RV.from_operation(np.minimum, a, b)

    @staticmethod
    def max(a: 'RV', b: 'RV') -> 'RV':
        return RV.from_operation(np.maximum, a, b)
    
    @staticmethod
    def sin(a: 'RV') -> 'RV':
        return RV.from_operation(np.sin, a, None)
    
    @staticmethod
    def cos(a: 'RV') -> 'RV':
        return RV.from_operation(np.cos, a, None)
    
    @staticmethod
    def tan(a: 'RV') -> 'RV':
        return RV.from_operation(np.tan, a, None)

    @staticmethod
    def cot(a: 'RV') -> 'RV':
        return RV.from_operation(lambda a: 1/np.tan(a), a, None)

    @staticmethod
    def floor(a: 'RV') -> 'RV':
        return RV.from_operation(np.floor, a, None)

    @staticmethod
    def log(a: 'RV') -> 'RV':
        return RV.from_operation(np.log, a, None)

    @staticmethod
    def sqrt(a: 'RV') -> 'RV':
        return RV.from_operation(np.sqrt, a, None)

