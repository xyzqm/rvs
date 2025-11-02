from rv import RV
from scipy import stats
import matplotlib.pyplot as plt

normal = RV.from_rv(stats.norm(loc=0, scale=1))
tr = RV.sin(normal)
tr.display()
flr = RV.floor(normal)
flr.display()

x = normal + RV.from_constant(5)
x.display()

# Show plot
plt.show()