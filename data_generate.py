import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
import torch

# CÁC HẰNG SỐ 
#------------------------------------
# Miền không gian-thời gian 
X_MIN, X_MAX = 0.0, 1.0   # Miền không gian
T_MIN, T_MAX = 0.0, 1.0   # Miền thời gian 

N_PDE = 10000  # điểm collocation
N_IC = 100     # điểm điều kiện ban đầu
N_BC = 100     # điểm điều kiện biên

# Tạo seed để có thể tái tạo kết quả
seed = 42
np.random.seed(seed)


# TẠO CÁC ĐIỂM DỮ LIỆU PDE NGẪU NHIÊN 
#------------------------------------
def generate_pde_points(n_points, x_range, t_range, method='lhs'):
    if method == 'lhs':
        # Sử dụng Latin Hypercube Sampling
        sampler = qmc.LatinHypercube(d=2, seed=seed)
        samples = sampler.random(n=n_points)
        
        # Scale về miền mong muốn
        x_min, x_max = x_range
        t_min, t_max = t_range
        
        points = np.empty_like(samples)
        points[:, 0] = x_min + (x_max - x_min) * samples[:, 0]
        points[:, 1] = t_min + (t_max - t_min) * samples[:, 1]
        
    elif method == 'random':
        x = np.random.uniform(x_range[0], x_range[1], n_points)
        t = np.random.uniform(t_range[0], t_range[1], n_points)
        points = np.column_stack([x, t])
    
    else:
        raise ValueError("Method must be 'lhs' or 'random'")
    
    return points


# TẠO CÁC ĐIỂM DỮ LIỆU BAN ĐẦU NGẪU NHIÊN 
#------------------------------------
def generate_ic_points(n_points, x_min, x_max, t_initial=0.0):
    # Phân bố đều trên miền không gian
    x_ic = np.linspace(x_min, x_max, n_points)
    t_ic = np.full_like(x_ic, t_initial)
    
    return np.column_stack([x_ic, t_ic])

# TẠO CÁC ĐIỂM DỮ LIỆU BIÊN NGẪU NHIÊN 
#------------------------------------
def generate_bc_points(n_points, t_range, boundaries):
    bc_points = []
    t_min, t_max = t_range
    n_points_per_boundary = n_points // 2
    for x_boundary in boundaries:
        # Phân bố đều trên miền thời gian
        t_bc = np.linspace(t_min, t_max, n_points_per_boundary)
        x_bc = np.full_like(t_bc, x_boundary)
        bc_points.append(np.column_stack([x_bc, t_bc]))
    
    return np.vstack(bc_points)