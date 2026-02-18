import torch
import numpy as np

def calculate_absolute_error_L1(true_solution, predicted_solution):
    if isinstance(true_solution, torch.Tensor):
        absolute_errors = torch.abs(true_solution - predicted_solution)
        L1_error = torch.mean(absolute_errors)
        return L1_error.item() if L1_error.dim() == 0 else L1_error
    else:
        # Numpy arrays
        absolute_errors = np.abs(true_solution - predicted_solution)
        L1_error = np.mean(absolute_errors)
        return L1_error

def calculate_max_absolute_error_Linf(true_solution, predicted_solution, axis=None):
    if isinstance(true_solution, torch.Tensor):
        absolute_errors = torch.abs(true_solution - predicted_solution)
        Linf_error = torch.max(absolute_errors) if axis is None else torch.max(absolute_errors, dim=axis)[0]
        return Linf_error.item() if Linf_error.dim() == 0 else Linf_error
    else:
        # Numpy arrays
        absolute_errors = np.abs(true_solution - predicted_solution)
        Linf_error = np.max(absolute_errors, axis=axis)
        return Linf_error


def calculate_l2_norm_error(true_solution, predicted_solution):
    if isinstance(true_solution, torch.Tensor):
        squared_errors = (true_solution - predicted_solution) ** 2
        l2_error = torch.sqrt(torch.mean(squared_errors))
        return l2_error.item() if l2_error.dim() == 0 else l2_error
    else:
        # Numpy arrays
        squared_errors = (true_solution - predicted_solution) ** 2
        l2_error = np.sqrt(np.mean(squared_errors))
        return l2_error


def calculate_all_errors(true_solution, predicted_solution, verbose=True):
   # Chuyển đổi về numpy arrays nếu là tensors
    if isinstance(true_solution, torch.Tensor):
        true_np = true_solution.detach().cpu().numpy()
        pred_np = predicted_solution.detach().cpu().numpy()
    else:
        true_np = np.array(true_solution)
        pred_np = np.array(predicted_solution)
    
    # Flatten nếu cần
    if true_np.ndim > 1:
        true_np = true_np.flatten()
        pred_np = pred_np.flatten()
    
    # Tính các sai số
    errors = {
        'L1': calculate_absolute_error_L1(true_np, pred_np),
        'Linf': calculate_max_absolute_error_Linf(true_np, pred_np),
        'L2': calculate_l2_norm_error(true_np, pred_np)
    }
    
    if verbose:
        print(f"Absolute Error: {errors['L1']:.6e}")
        print(f"Maximum Absolute Error: {errors['Linf']:.6e}")
        print(f"L2 Error: {errors['L2']:.6e}")
        
    return errors


# Hàm cho phân tích thống kê thêm (theo bài báo)
def calculate_statistical_metrics(true_solution, predicted_solution):
    if isinstance(true_solution, torch.Tensor):
        true_np = true_solution.detach().cpu().numpy()
        pred_np = predicted_solution.detach().cpu().numpy()
    else:
        true_np = np.array(true_solution)
        pred_np = np.array(predicted_solution)
    
    errors = pred_np - true_np
    
    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(np.mean(errors**2))
    
    # Standard Deviation of Error (SDE)
    sde = np.std(errors)
    
    # Coefficient of Variation (CV) - theo phần trăm
    cv = (sde / np.mean(np.abs(true_np))) * 100 if np.mean(np.abs(true_np)) != 0 else 0
    
    metrics = {
        'RMSE': rmse,
        'SDE': sde,
        'CV_percent': cv
    }
    
    return metrics


