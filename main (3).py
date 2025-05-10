import numpy as np

def is_strictly_diagonally_dominant(A):
    n = A.shape[0]
    for i in range(n):
        diag_val = abs(A[i, i])
        off_diag_sum = np.sum(np.abs(A[i, :])) - diag_val
        if diag_val <= off_diag_sum:
            return False
    return True

def jacobi(A, b, initial_guess, max_iterations=100, tolerance=1e-6):
    n = A.shape[0]
    x = np.array(initial_guess, dtype=float)
    x_new = np.zeros_like(x)
    for k in range(max_iterations):
        for i in range(n):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
        if np.linalg.norm(x_new - x) < tolerance:
            print(f"Jacobi method converged in {k+1} iterations.")
            return x_new
        x[:] = x_new
    print(f"Jacobi method did not converge within {max_iterations} iterations.")
    return x

def gauss_seidel(A, b, initial_guess, max_iterations=100, tolerance=1e-6):
    n = A.shape[0]
    x = np.array(initial_guess, dtype=float)
    for k in range(max_iterations):
        x_old = np.copy(x)
        for i in range(n):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x[i] = (b[i] - s1 - s2) / A[i, i]
        if np.linalg.norm(x - x_old) < tolerance:
            print(f"Gauss-Seidel method converged in {k+1} iterations.")
            return x
    print(f"Gauss-Seidel method did not converge within {max_iterations} iterations.")
    return x

def sor(A, b, initial_guess, omega, max_iterations=100, tolerance=1e-6):
    n = A.shape[0]
    x = np.array(initial_guess, dtype=float)
    for k in range(max_iterations):
        x_old = np.copy(x)
        for i in range(n):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x[i] = (1 - omega) * x_old[i] + (omega / A[i, i]) * (b[i] - s1 - s2)
        if np.linalg.norm(x - x_old) < tolerance:
            print(f"SOR method converged in {k+1} iterations with omega = {omega}.")
            return x
    print(f"SOR method did not converge within {max_iterations} iterations with omega = {omega}.")
    return x

def conjugate_gradient(A, b, initial_guess, max_iterations=100, tolerance=1e-6):
    x = np.array(initial_guess, dtype=float)
    r = b - np.dot(A, x)
    p = np.copy(r)
    for k in range(max_iterations):
        Ap = np.dot(A, p)
        alpha = np.dot(r, r) / np.dot(p, Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        if np.linalg.norm(r_new) < tolerance:
            print(f"Conjugate Gradient method converged in {k+1} iterations.")
            return x
        beta = np.dot(r_new, r_new) / np.dot(r, r)
        p = r_new + beta * p
        r = r_new
    print(f"Conjugate Gradient method did not converge within {max_iterations} iterations.")
    return x


A = np.array([
    [ 4, -1,  0, -1,  0,  0],
    [-1,  4, -1,  0, -1,  0],
    [ 0, -1,  4,  0,  1, -1],  
    [-1,  0,  0,  4, -1, -1],
    [ 0, -1,  0, -1,  4, -1],
    [ 0,  0, -1, 0, -1,  4]
], dtype=float)

b = np.array([0, -1, 9, 4, 8, 6], dtype=float)
initial_guess = np.zeros_like(b)

# (a) Jacobi method
print("\n(a) Jacobi Method:")
solution_jacobi = jacobi(A, b, initial_guess)
print("Solution (Jacobi):", solution_jacobi)

# (b) Gauss-Seidel method
print("\n(b) Gauss-Seidel Method:")
solution_gauss_seidel = gauss_seidel(A, b, initial_guess)
print("Solution (Gauss-Seidel):", solution_gauss_seidel)

# (c) SOR method
print("\n(c) SOR Method:")
omega = 1.2
solution_sor = sor(A, b, initial_guess, omega)
print(f"Solution (SOR with omega={omega}):", solution_sor)

# (d) Conjugate Gradient method
print("\n(d) Conjugate Gradient Method:")
solution_cg = conjugate_gradient(A, b, initial_guess)
print("Solution (Conjugate Gradient):", solution_cg)
