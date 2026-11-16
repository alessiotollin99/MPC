import numpy as np

#==========================================================================================
#Optimization through casadi
#==========================================================================================
import casadi as cd

opti = cd.Opti()

x = opti.variable()
y = opti.variable()

p1 = opti.parameter()
opti.set_value(p1, 1)
p2 = opti.parameter()
opti.set_value(p2, 100)

obj = (p1-x)**2 + p2*(y-x**2)**2 # usually a nd b (1, and 100 here) - solutions are of the type a and a**2

opti.minimize(obj)
opti.subject_to(x**2+y**2 <=1)

opti.solver('ipopt')
sol = opti.solve()

print(f"The minimum is in : \nx = {sol.value(x): .4f} \ny = {sol.value(y): .4f}")


#==========================================================================================
#Rosenbrock function visualization
#==========================================================================================
import matplotlib.pyplot as plt

a = 1
b = 100

x1 = np.linspace(-2, 2)
y1 = np.linspace(-1, 3)
X, Y = np.meshgrid(x1, y1)
Z = (a-X)**2+b*(Y-X**2)**2
g = X**2+Y**2

fig, ax = plt.subplots()
cs = ax.contour(X, Y, Z)
fig.colorbar(cs)

ax.contour(X, Y, g, levels=[1], colors="k")
ax.contourf(X, Y, g, levels=[1, g.max()], colors="grey", alpha = 0.3)

ax.plot(1, 1, "r*", markersize = 8)
ax.plot(sol.value(x), sol.value(y), "ko")

fig.tight_layout()
plt.show()
