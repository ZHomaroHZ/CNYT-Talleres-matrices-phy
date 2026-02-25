import numpy as np

def suma_vectores(v1, v2):
    return np.add(v1, v2)

def inverso_vector(v):
    return np.negative(v)

def escalar_vector(escalar, v):
    return escalar * v

def producto_interno(v1, v2):
    return np.vdot(v1, v2)

def norma_vector(v):
    return np.linalg.norm(v)

def distancia_vectores(v1, v2):
    return np.linalg.norm(v1 - v2)

def suma_matrices(m1, m2):
    return np.add(m1, m2)

def inversa_aditiva_matriz(m):
    return np.negative(m)

def escalar_matriz(escalar, m):
    return escalar * m

def transpuesta(m):
    return np.transpose(m)

def conjugada(m):
    return np.conjugate(m)

def adjunta(m):
    return np.conjugate(np.transpose(m))

def producto_matrices(m1, m2):
    return np.matmul(m1, m2)

def accion_matriz_vector(m, v):
    return np.matmul(m, v)

def valores_vectores_propios(m):
    return np.linalg.eig(m)

def es_unitaria(m):
    identidad = np.eye(m.shape[0])
    return np.allclose(adjunta(m) @ m, identidad)

def es_hermitiana(m):
    return np.allclose(m, adjunta(m))

def producto_tensor(a, b):
    return np.kron(a, b)
