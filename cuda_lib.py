import ctypes
import numpy as np

# Load the shared library
lib = ctypes.CDLL('./cuda_funcs.dll')

# Define the function signature
compute_determinants = lib.computeDeterminants
compute_determinants.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_int
]

linePlaneCrossPoints = lib.LinePlaneCrossPoints
linePlaneCrossPoints.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
                                     np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
                                     np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
                                     np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
                                     np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
                                     np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
                                     np.ctypeslib.ndpointer(dtype=ctypes.c_bool, ndim=1, flags='C_CONTIGUOUS'),
                                     ctypes.c_int]


def line_plane_crosspoints(origins,directions,p1s,p2s,p3s):
    n = len(origins)
    origins_flat = np.array(origins,dtype=np.float32).flatten()
    directions_flat = np.array(directions,dtype=np.float32).flatten()
    p1s_flat = np.array(p1s,dtype=np.float32).flatten()
    p2s_flat = np.array(p2s,dtype=np.float32).flatten()
    p3s_flat = np.array(p3s,dtype=np.float32).flatten()
    cross_points = np.zeros((n, 3), dtype=np.float32).flatten()
    has_intersections = np.zeros(n, dtype=bool)
    
    linePlaneCrossPoints(origins_flat, directions_flat, p1s_flat, p2s_flat, p3s_flat, cross_points,has_intersections, n)

    return cross_points

def calculate_determinants(matrices):
    n = len(matrices)
    matrices = np.array(matrices, dtype=np.float32).flatten()
    results = np.zeros(n, dtype=np.float32)

    compute_determinants(matrices, results, n)

    return results

