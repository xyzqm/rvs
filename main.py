from rv import RV
import scipy
import matplotlib.pyplot as plt

normal = RV.from_rv(scipy.stats.norm(loc=0, scale=1))
tr = RV.sin(normal)
tr.display()
flr = RV.floor(normal)
flr.display()

# Show plot
plt.show()