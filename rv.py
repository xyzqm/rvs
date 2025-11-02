import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

class RV:
    def __init__(self, rv=None, fn=None, lhs=None, rhs=None):
        self.rv = rv
        self.fn = fn
        self.lhs = lhs
        self.rhs = rhs

    @classmethod
    def from_rv(cls, rv):
        return cls(rv=rv)

    @classmethod
    def from_operation(cls, fn, lhs, rhs):
        return cls(fn=fn, lhs=lhs, rhs=rhs)

    @classmethod
    def constant(cls, value):
        return cls.from_rv(scipy.stats.norm(loc=value, scale=0))

    def sample(self, samples=10000):
        if self.rv is None:
            left_samples = self.lhs.sample(samples)
            if self.rhs is None:
                return self.fn(left_samples)
            else:
                right_samples = self.rhs.sample(samples)
                return self.fn(left_samples, right_samples)

        return self.rv.rvs(samples)

    def display(self):
        plt.figure(figsize=(10, 6))
        # Plot histogram
        plt.hist(self.sample(), bins=100, edgecolor='black', density=True)  # 'bins' controls number of bars
        plt.title("Histogram of Data")
        plt.xlabel("Value")
        plt.ylabel("Density")
        # sns.kdeplot(self.sample(), color='red', linestyle='--', linewidth=2, label='KDE (Smooth)')

    def __add__(self, other) -> 'RV':
        return RV.from_operation(lambda a, b: a + b, self, other)

    def __sub__(self, other) -> 'RV':
        return RV.from_operation(lambda a, b: a - b, self, other)

    def __mul__(self, other) -> 'RV':
        return RV.from_operation(lambda a, b: a * b, self, other)
    
    def __div__(self, other) -> 'RV':
        return RV.from_operation(lambda a, b: a / b, self, other)
    
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

