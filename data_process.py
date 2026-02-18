import numpy as np
from data_generate import *

RE = 1.0
SIGMA = 2.0 
EPS = 1e-8 

# nghiệm chính xác 
def exact_solution(x, t, sigma = SIGMA, Re = RE): 
    # Điều kiện biên
    if abs(x - X_MIN) < EPS or abs(x - X_MAX) < EPS:
        return 0 
    
    # Điều kiện ban đầu
    if(abs(t - T_MIN) < EPS): 
        up = 2 * np.pi * np.sin(np.pi * x)
        down = Re * (sigma + np.cos(np.pi * x))
        u = up/down 
        return u 
    
    up = 2 * np.pi * np.exp(-np.pi**2 * t / Re) * np.sin(np.pi * x)
    down = Re * (sigma + np.exp(-np.pi**2 * t / Re) * np.cos(np.pi * x))
    u = up/down 
    return u

def generate_data(sample_points):  
    data = []
    for point in sample_points:
        u = exact_solution(point[0], point[1])
        data.append([point[0], point[1], u]) # (x,t,u) 
    return np.array(data, dtype=np.float32)

def data_process():
    pde_points = generate_pde_points(N_PDE, (X_MIN, X_MAX), (T_MIN, T_MAX), method='lhs')
    ic_points = generate_ic_points(N_IC, X_MIN, X_MAX) 
    bc_points = generate_bc_points(N_BC, (T_MIN, T_MAX), [X_MIN, X_MAX])
    collocation_data = generate_data(pde_points) 
    initial_data = generate_data(ic_points)
    bound_data = generate_data(bc_points)
    # print("col", collocation_data)
    # print("ini", initial_data)
    # print("bou",bound_data) 
    return collocation_data, initial_data, bound_data 

if __name__ == "__main__":
    collocation_data, initial_data, bound_data = data_process()