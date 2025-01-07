import sympy as sp
import itertools

# Define the symbols for P components and other parameters
P1, P2, P3 = sp.symbols('P1 P2 P3')  # P vector components
e33, e31, e15, em = sp.symbols('e33 e31 e15 em')  # Other symbols
I = sp.eye(3)  # Identity matrix (Kronecker delta)
norm_sq = P1**2 + P2**2 + P3**2
# Define the tensor eP symbolically
eP = sp.MutableDenseNDimArray.zeros(3, 3, 3)

# Populate eP based on the provided formula
for k, i, j in itertools.product(range(3), repeat=3):
    eP[k, i, j] = (
        P1 * P2 * P3 * em +
        norm_sq * (
            e31 * I[i, j] * [P1, P2, P3][k] +
            e15 / 2 * (I[k, i] * [P1, P2, P3][j] + I[k, j] * [P1, P2, P3][i])
        )
    )

# Compute the derivatives of eP with respect to each component of P
# Derivative wrt P1, deP_dP1
for i, j, k in itertools.product(range(3), repeat=3):
    print(f"deP_dP1[{i}, {j}, {k}] = {eP[i,j,k].diff(P1)}")

# Derivative wrt P2, deP_dP2
for i, j, k in itertools.product(range(3), repeat=3):
    print(f"deP_dP2[{i}, {j}, {k}] = {eP[i,j,k].diff(P2)}")

# Derivative wrt P3, deP_dP3
for i, j, k in itertools.product(range(3), repeat=3):
    print(f"deP_dP3[{i}, {j}, {k}] = {eP[i,j,k].diff(P3)}")

