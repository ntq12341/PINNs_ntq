import torch
import numpy as np
from network import PhysicsInformedNN
from data_process import *
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from perform_evaluate import * 
from visual import * 
from scipy.interpolate import RectBivariateSpline
import time

def train():
    data_c, data_i, data_b = data_process()
    layers = [2, 20, 20, 20, 20, 1]
    X_train = data_c[:,0:2]
    Xi = data_i[:, 0:2]
    Xb = data_b[:, 0:2]
    ui = data_i[:,2]
    ub = data_b[:,2]

    # training
    nu = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhysicsInformedNN(X_train, Xi, Xb, ui, ub, layers, device, nu)
      
    print("\nTraining with Adam optimizer...")
    start_time = time.time()                
    model.train(epochs=10000)
    elapsed = time.time() - start_time                
    print('\nTraining time: %.4f seconds' % (elapsed))
    
    print("\nFine-tuning with L-BFGS optimizer...")
    start_time = time.time()
    model.train_lbfgs(max_iter=10000)
    elapsed = time.time() - start_time
    print('\nFine-tuning time: %.4f seconds' % (elapsed))

    u_pred, f_pred = model.predict(data_c[:,0:2])

    # u_test = torch.tensor(data_c[:, 2], dtype=torch.float32, requires_grad=False).to(device) 
    u_test = data_c[:,2] 
    
    # Tính thêm metrics thống kê
    stats = calculate_statistical_metrics(u_test, u_pred)
    print(f"\nStatistical Metrics:")
    print(f"RMSE: {stats['RMSE']:.6e}")
    print(f"Standard Deviation of Error: {stats['SDE']:.6e}")
    print(f"Coefficient of Variation: {stats['CV_percent']:.2f}%")

    # Plot loss history
    plot_loss(model.loss_history, ylabel='Loss')
    
    return model


def predict_at_point(model, x, t, device):
    """
    Dự đoán nghiệm tại một điểm (x, t) cụ thể
    
    Parameters:
    -----------
    model : PhysicsInformedNN
        Mô hình đã được train
    x : float
        Tọa độ không gian
    t : float
        Tọa độ thời gian
    device : torch.device
        CPU hoặc CUDA device
        
    Returns:
    --------
    float : Giá trị dự đoán u(x, t)
    """
    # Tạo tensor đầu vào
    point = np.array([[x, t]], dtype=np.float32)
    
    # Dự đoán
    u_pred, f_pred = model.predict(point)
    
    # Convert về float
    u_pred_value = float(u_pred[0, 0])
    
    return u_pred_value


def predict_at_multiple_points(model, points_list, device):
    """
    Dự đoán nghiệm tại nhiều điểm cùng lúc
    
    Parameters:
    -----------
    model : PhysicsInformedNN
        Mô hình đã được train
    points_list : list of tuples
        Danh sách các điểm [(x1, t1), (x2, t2), ...]
    device : torch.device
        CPU hoặc CUDA device
        
    Returns:
    --------
    list of tuples : [(u_pred, u_exact, error), ...]
    """
    results = []
    
    for x, t in points_list:
        u_pred = predict_at_point(model, x, t, device)
        u_exact = exact_solution(x, t)
        error = abs(u_exact - u_pred)
        results.append((u_pred, u_exact, error))
    
    return results

def create_table1(model, device, save_csv=True):
    """
    Tạo bảng dự đoán tại nhiều điểm và so sánh với nghiệm chính xác
    (Giống Table 1 trong bài báo)
    """
    import pandas as pd
    
    # Định nghĩa các điểm test (giống Table 1 trong paper)
    test_configs = [
        {'t': 0.02, 'x_values': [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]},
        {'t': 0.04, 'x_values': [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]},
    ]
    
    results = []
    
    print("\n" + "="*80)
    print("PREDICTION TABLE (Similar to Table 1 in paper)")
    print("="*80)
    
    for config in test_configs:
        t = config['t']
        print(f"\nAt t = {t}:")
        print(f"{'x':<8} {'u_pred':<15} {'u_exact':<15} {'Absolute Error':<15}")
        print("-"*60)
        
        for x in config['x_values']:
            u_pred = predict_at_point(model, x, t, device)
            u_exact = exact_solution(x, t)
            error = abs(u_exact - u_pred)
            
            results.append({
                't': t,
                'x': x,
                'u_predicted': u_pred,
                'u_exact': u_exact,
                'absolute_error': error
            })
            
            print(f"{x:<8.1f} {u_pred:<15.8f} {u_exact:<15.8f} {error:<15.6e}")
    
    # Save to CSV
    if save_csv:
        df = pd.DataFrame(results)
        df.to_csv('prediction_table.csv', index=False)
        print("\n Prediction table saved to 'prediction_table.csv'")
    
    return results

def create_table2(model, device, N_points=5000, save_csv=True):
    """
    Tạo bảng sai số L2 và L-infinity tại các thời điểm t = 0, 0.2, 0.4, 0.6, 0.8, 1.0
    
    Parameters:
    -----------
    model : PhysicsInformedNN
        Mô hình đã được train
    device : torch.device
        CPU hoặc CUDA device
    N_points : int
        Số điểm trên mỗi mặt cắt thời gian (mặc định: 5000)
    save_csv : bool
        Có lưu kết quả ra file CSV không
    
    Returns:
    --------
    list : Danh sách kết quả cho từng thời điểm
    """
    import pandas as pd
    
    # Các thời điểm cần đánh giá
    t_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    print("\n" + "="*80)
    print("TABLE 2: L2 and L-infinity Errors at Different Time Instants")
    print("="*80)
    print(f"Number of points per time slice: {N_points}")
    print("-"*80)
    print(f"{'t':<10} {'L2 Error':<25} {'L-infinity Error':<25}")
    print("-"*80)
    
    results = []
    
    for t in t_values:
        # Tạo lưới điểm x đều trên [0, 1]
        x_vals = np.linspace(0, 1, N_points)
        
        # Tạo mảng điểm (x, t) cho thời điểm hiện tại
        X_star = np.column_stack([x_vals, np.full_like(x_vals, t)])
        
        # Dự đoán
        u_pred, f_pred = model.predict(X_star)
        u_pred = u_pred.flatten()  # Flatten để tính toán
        
        # Nghiệm chính xác
        u_exact = np.array([exact_solution(x, t) for x in x_vals])
        
        # Tính sai số L2 và L-infinity
        l2_error = calculate_l2_norm_error(u_exact, u_pred)
        linf_error = calculate_max_absolute_error_Linf(u_exact, u_pred)
        
        # Lưu kết quả
        results.append({
            't': t,
            'L2_error': l2_error,
            'Linf_error': linf_error,
            'N_points': N_points
        })
        
        # In kết quả
        print(f"t = {t:<5.1f}   {l2_error:<25.6e} {linf_error:<25.6e}")
    
    print("="*80)
    
    # Lưu ra file CSV
    if save_csv:
        df = pd.DataFrame(results)
        filename = f'table2_errors_with_ratio_N{N_points}.csv'
        df.to_csv(filename, index=False, float_format='%.6e')
        print(f"\n Table 2 saved to '{filename}'")
    
    return results

def plot_loss(losses, ylabel):
    epochs = len(losses)
    x_epochs = np.arange(1, epochs + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(x_epochs, losses, color='blue')
    plt.xlabel('Iteration')
    plt.ylabel(ylabel)
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    plt.tight_layout()
    plt.savefig('loss_history.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    model = train()
    
    # Sau khi train xong, bạn có thể dự đoán thêm tại bất kỳ điểm nào
    print("\n" + "="*80)
    print("ADDITIONAL PREDICTIONS")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Tạo bảng prediction (giống Table 1 trong paper)
    create_table1(model, device)
    create_table2(model, device, N_points=3000)
    create_table2(model, device, N_points=5000)
    # Vẽ đồ thị so sánh 
    result_3d(model, t_max=0.4, n_points=100, exact=True, cmap='viridis')
    result_3d(model, t_max=0.4, n_points=100, exact=False, cmap='cividis',save_path='pinn_origin.png')
    result_3d(model, t_max=0.4, n_points=100, exact=False, cmap='plasma')

    
    result_compare(model, t_values=[0.01, 0.2, 0.3], n_points=200) 
    # result_heatmap(model, t_max=0.4, n_points=200) 