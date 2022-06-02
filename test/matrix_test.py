import numpy as np
# starting with the equation X_dot = F(X) + G(X) * U
# move F(X) over, rename X_dot - F(X) to be just X for simplicity
# X = G(X) * U

x = np.matrix([[1], [2], [3]])
u = np.matrix(([0], [0.001]))
g = np.matmul(x, np.linalg.pinv(u))


print("X =\n", x)
print("U =\n", u)
print("G = X * inv(U) =\n", g)
print("\nCheck that the original equation still holds:\n")
print("X = G * U =\n", np.matmul(g, u))
