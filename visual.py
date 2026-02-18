import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from data_process import * 

def result_3d(model, t_max=0.4, n_points=100):
    # ===== Tạo lưới =====
    x = np.linspace(0, 1, n_points)
    t = np.linspace(0, t_max, n_points)
    X, T = np.meshgrid(x, t)

    # ===== PINN prediction =====
    X_flat = X.flatten()[:, None]
    T_flat = T.flatten()[:, None]
    X_star = np.hstack([X_flat, T_flat])

    U_pinn_flat, _ = model.predict(X_star)
    U_pinn = U_pinn_flat.reshape(X.shape)

    # ===== Exact solution =====
    U_exact = np.zeros_like(X)
    for i in range(len(t)):
        for j in range(len(x)):
            U_exact[i, j] = exact_solution(x[j], t[i])

    # ===== Figure =====
    fig = plt.figure(figsize=(14, 6))

    # ===================== (a) EXACT =====================
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(
        X, T, U_exact,
        cmap='viridis',
        edgecolor='none',
        linewidth=0,
        antialiased=True
    )

    ax1.set_xlabel('x', fontsize=12, fontweight='bold')
    ax1.set_ylabel('t', fontsize=12, fontweight='bold')
    ax1.set_zlabel('u(x,t)', fontsize=12, fontweight='bold')
    ax1.set_title('(a)', fontsize=14, fontweight='bold')

    # ax1.set_zlim(0, 3.0)     
    ax1.view_init(elev=25, azim=-60)

    ax1.grid(False)         
    cbar1 = fig.colorbar(surf1, ax=ax1, shrink=0.6, pad=0.08)
    cbar1.set_label('u(x,t)', fontsize=10)

    # ===================== (b) PINN =====================
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(
        X, T, U_pinn,
        cmap='plasma',
        edgecolor='none',
        linewidth=0,
        antialiased=True
    )

    ax2.set_xlabel('x', fontsize=12, fontweight='bold')
    ax2.set_ylabel('t', fontsize=12, fontweight='bold')
    ax2.set_zlabel('u(x,t)', fontsize=12, fontweight='bold')
    ax2.set_title('(b)', fontsize=14, fontweight='bold')

    # ax2.set_zlim(0, 3.0)     
    ax2.view_init(elev=25, azim=-60)

    ax2.grid(False)          

    cbar2 = fig.colorbar(surf2, ax=ax2, shrink=0.6, pad=0.08)
    cbar2.set_label('u(x,t)', fontsize=10)

    plt.tight_layout()
    plt.savefig('3d_comparison_clean.png', dpi=300, bbox_inches='tight')
    plt.show()

def result_compare(model, t_values=[0.01, 0.2, 0.3], n_points=200):
    x_vals = np.linspace(0, 1, n_points)

    fig, axes = plt.subplots(1, len(t_values), figsize=(5*len(t_values), 4))

    # Nếu chỉ có 1 subplot
    if len(t_values) == 1:
        axes = [axes]

    for i, t in enumerate(t_values):

        # ===== Prediction =====
        X_star = np.column_stack([x_vals, np.full_like(x_vals, t)])
        u_pred, _ = model.predict(X_star)
        u_pred = u_pred.flatten()

        # ===== Exact =====
        u_exact = np.array([exact_solution(x, t) for x in x_vals])

        # ===== Plot =====
        axes[i].plot(x_vals, u_exact,
                     color='blue',
                     linewidth=2.5,
                     label='Exact')

        axes[i].plot(x_vals, u_pred,
                     color='red',
                     linestyle='--',
                     linewidth=2.5,
                     label='Prediction')

        # Title in đậm
        axes[i].set_title(f't = {t}',
                          fontsize=12,
                          fontweight='bold')

        # Label in đậm
        axes[i].set_xlabel('x',
                           fontsize=12,
                           fontweight='bold')
        axes[i].set_ylabel('u(x,t)',
                           fontsize=12,
                           fontweight='bold')

        axes[i].set_xlim([0, 1])
        axes[i].grid(False)  

    # ===== Legend giữa dưới =====
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='lower center',
               ncol=2,
               frameon=False,
               fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # chừa chỗ cho legend

    plt.savefig('comparison_clean.png',
                dpi=300,
                bbox_inches='tight')
    plt.show()

    
def result_heatmap(model, t_max=0.4, n_points=200):
    """
    Vẽ heatmap đúng orientation: 
    trục ngang = t
    trục dọc   = x
    """
    # Tạo lưới
    x = np.linspace(0, 1, n_points)
    t = np.linspace(0, t_max, n_points)
    X, T = np.meshgrid(x, t)
    
    # Chuẩn bị input cho model
    X_flat = X.flatten()[:, None]
    T_flat = T.flatten()[:, None]
    X_star = np.hstack([X_flat, T_flat])
    
    # Predict
    U_flat, _ = model.predict(X_star)
    U = U_flat.reshape(X.shape)
    
    # Vẽ đúng orientation (t ngang, x dọc)
    plt.figure(figsize=(20, 6))
    cp = plt.contourf(T, X, U, 50, cmap='viridis',vmin = 0, vmax = 3.2) 
    plt.colorbar(cp, label='u(x,t)')
    
    plt.xlabel('t', fontsize=12, fontweight='bold')
    plt.ylabel('x', fontsize=12, fontweight='bold')
    plt.title(f'u(x,t)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('heatmap_correct.png', dpi=300)
    plt.show()
