import sys
sys.path.insert(0, '../../Utilities/')

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
import time
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd  # Cần pandas

# Try to import plotting utilities, fallback to matplotlib if not available
try:
    from plotting import newfig, savefig
except ImportError:
    def newfig(width, height):
        fig = plt.figure(figsize=(width*6, height*6))
        return fig, fig.gca()
    def savefig(path):
        plt.savefig(path, bbox_inches='tight', dpi=300)

np.random.seed(1234)
torch.manual_seed(1234)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class PhysicsInformedNN:
    def __init__(self, X_u, u, X_f, layers, lb, ub, nu):
        
        self.lb = torch.tensor(lb, dtype=torch.float32).to(device)
        self.ub = torch.tensor(ub, dtype=torch.float32).to(device)
    
        self.x_u = torch.tensor(X_u[:,0:1], dtype=torch.float32, requires_grad=True).to(device)
        self.t_u = torch.tensor(X_u[:,1:2], dtype=torch.float32, requires_grad=True).to(device)
        
        self.x_f = torch.tensor(X_f[:,0:1], dtype=torch.float32, requires_grad=True).to(device)
        self.t_f = torch.tensor(X_f[:,1:2], dtype=torch.float32, requires_grad=True).to(device)
        
        self.u = torch.tensor(u, dtype=torch.float32).to(device)
        
        self.layers = layers
        self.nu = nu
        
        # Initialize Neural Network
        self.model = self.build_model(layers).to(device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # For L-BFGS optimizer (optional, uncomment if needed)
        # self.optimizer = torch.optim.LBFGS(
        #     self.model.parameters(), 
        #     max_iter=50000,
        #     tolerance_grad=1e-7,
        #     tolerance_change=1e-9,
        #     history_size=50,
        #     line_search_fn='strong_wolfe'
        # )
        
        self.loss_history = []

    def build_model(self, layers):
        """Build neural network"""
        modules = []
        for i in range(len(layers)-1):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers)-2:
                modules.append(nn.Tanh())
        
        model = nn.Sequential(*modules)
        
        # Xavier initialization
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        
        model.apply(init_weights)
        
        return model
    
    def neural_net(self, x, t):
        """Forward pass through the network"""
        # Normalize inputs
        X = torch.cat([x, t], dim=1)
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        return self.model(H)
    
    def net_u(self, x, t):
        """Predict u"""
        return self.neural_net(x, t)
    
    def net_f(self, x, t):
        """Physics-informed loss (PDE residual)"""
        u = self.net_u(x, t)
        
        # Compute gradients
        u_t = torch.autograd.grad(
            u, t, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        
        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]
        
        # Burgers' equation: u_t + u*u_x - nu*u_xx = 0
        f = u_t + u * u_x - self.nu * u_xx
        
        return f
    
    def loss_fn(self):
        """Calculate total loss"""
        # Data loss
        u_pred = self.net_u(self.x_u, self.t_u)
        loss_u = torch.mean((self.u - u_pred)**2)
        
        # Physics loss
        f_pred = self.net_f(self.x_f, self.t_f)
        loss_f = torch.mean(f_pred**2)
        
        # Total loss
        loss = loss_u + loss_f
        
        return loss, loss_u, loss_f
    
    def train(self, epochs=10000):
        """Training loop"""
        self.model.train()
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            loss, loss_u, loss_f = self.loss_fn()
            
            loss.backward()
            self.optimizer.step()
            
            self.loss_history.append(loss.item())
            
            if epoch % 1000 == 0:
                print(f'Epoch {epoch:5d}: Loss = {loss.item():.6e}, Loss_u = {loss_u.item():.6e}, Loss_f = {loss_f.item():.6e}')
    
    def train_lbfgs(self, max_iter=50000):
        """Training with L-BFGS optimizer (more accurate but slower)"""
        self.model.train()
        
        def closure():
            self.optimizer.zero_grad()
            loss, loss_u, loss_f = self.loss_fn()
            loss.backward()
            
            self.loss_history.append(loss.item())
            
            if len(self.loss_history) % 100 == 0:
                print(f'Iter {len(self.loss_history):5d}: Loss = {loss.item():.6e}')
            
            return loss
        
        self.optimizer = torch.optim.LBFGS(
            self.model.parameters(), 
            max_iter=max_iter,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            history_size=50,
            line_search_fn='strong_wolfe'
        )
        
        self.optimizer.step(closure)
    
    def predict(self, X_star):
        """Make predictions"""
        self.model.eval()
        
        x = torch.tensor(X_star[:,0:1], dtype=torch.float32, requires_grad=True).to(device)
        t = torch.tensor(X_star[:,1:2], dtype=torch.float32, requires_grad=True).to(device)
        
        with torch.no_grad():
            u_pred = self.net_u(x, t)
        
        # For f_pred we need gradients
        x.requires_grad = True
        t.requires_grad = True
        f_pred = self.net_f(x, t)
        
        return u_pred.cpu().numpy(), f_pred.detach().cpu().numpy()

if __name__ == "__main__": 
     
    nu = 1
    noise = 0.0        

    N_u = 100
    N_f = 10000
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    
    # # Load data
    # try:
    #     data = scipy.io.loadmat('burgers_shock.mat')
    # except FileNotFoundError:
    #     print("Error: Could not find '../Data/burgers_shock.mat'")
    #     print("Please ensure the data file exists or update the path.")
    #     sys.exit(1)
    
    # t = data['t'].flatten()[:,None]
    # x = data['x'].flatten()[:,None]
    # Exact = np.real(data['usol']).T
    # X, T = np.meshgrid(x,t)
    
    # Đường dẫn đến file CSV
    csv_file = 'burgers_data.csv'  # Thay đổi đường dẫn này
    
    # Tên các cột trong CSV (thay đổi nếu cần)
    x_column = 'x'  # hoặc 'x_i', 'X', etc.
    t_column = 't'  # hoặc 't_i', 'T', 'time', etc.
    u_column = 'u'  # hoặc 'u_i', 'U', 'solution', etc.
    
    try:
        # Load dữ liệu từ CSV
        x, t, X, T, Exact = load_data_from_csv(
            csv_file, 
            x_col=x_column, 
            t_col=t_column, 
            u_col=u_column
        )
        
    except FileNotFoundError:
        print(f"\n❌ Không tìm thấy file: {csv_file}")
        print("\n💡 Vui lòng:")
        print("   1. Đặt file CSV vào cùng thư mục với script")
        print("   2. Hoặc thay đổi đường dẫn trong biến 'csv_file'")
        print("   3. Đảm bảo file CSV có 3 cột: x, t, u")
        print("\n📝 Format CSV mẫu:")
        print("   x,t,u")
        print("   -1.0,0.0,0.5")
        print("   -0.99,0.0,0.48")
        print("   ...")
        import sys
        sys.exit(1)
    
    except ValueError as e:
        print(f"\n❌ Lỗi: {e}")
        print("\n💡 Kiểm tra lại:")
        print(f"   - File CSV có các cột: {x_column}, {t_column}, {u_column}")
        print("   - Hoặc thay đổi tên cột trong biến x_column, t_column, u_column")
        import sys
        sys.exit(1)
    
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact.flatten()[:,None]              

    # Domain bounds
    lb = X_star.min(0)
    ub = X_star.max(0)    
        
    xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T))
    uu1 = Exact[0:1,:].T
    xx2 = np.hstack((X[:,0:1], T[:,0:1]))
    uu2 = Exact[:,0:1]
    xx3 = np.hstack((X[:,-1:], T[:,-1:]))
    uu3 = Exact[:,-1:]
    
    X_u_train = np.vstack([xx1, xx2, xx3])
    X_f_train = lb + (ub-lb)*lhs(2, N_f)
    X_f_train = np.vstack((X_f_train, X_u_train))
    u_train = np.vstack([uu1, uu2, uu3])
    
    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    X_u_train = X_u_train[idx, :]
    u_train = u_train[idx,:]
        
    model = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub, nu)
    
    print("\nTraining with Adam optimizer...")
    start_time = time.time()                
    model.train(epochs=10000)
    elapsed = time.time() - start_time                
    print('\nTraining time: %.4f seconds' % (elapsed))
    
    # Optional: Fine-tune with L-BFGS (uncomment if needed)
    # print("\nFine-tuning with L-BFGS optimizer...")
    # start_time = time.time()
    # model.train_lbfgs(max_iter=5000)
    # elapsed = time.time() - start_time
    # print('\nFine-tuning time: %.4f seconds' % (elapsed))
    
    u_pred, f_pred = model.predict(X_star)
            
    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    print('\nRelative L2 error: %e' % (error_u))                     

    
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
    Error = np.abs(Exact - U_pred)
    
    
    ######################################################################
    ############################# Plotting ###############################
    ######################################################################    
    
    fig, ax = newfig(1.0, 1.1)
    ax.axis('off')
    
    ####### Row 0: u(t,x) ##################    
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])
    
    h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow', 
                  extent=[t.min(), t.max(), x.min(), x.max()], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (u_train.shape[0]), markersize = 4, clip_on = False)
    
    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)    
    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.legend(frameon=False, loc = 'best')
    ax.set_title('$u(t,x)$', fontsize = 10)
    
    ####### Row 1: u(t,x) slices ##################    
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)
    
    ax = plt.subplot(gs1[0, 0])
    ax.plot(x,Exact[25,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,U_pred[25,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')    
    ax.set_title('$t = 0.25$', fontsize = 10)
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])
    
    ax = plt.subplot(gs1[0, 1])
    ax.plot(x,Exact[50,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,U_pred[50,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])
    ax.set_title('$t = 0.50$', fontsize = 10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)
    
    ax = plt.subplot(gs1[0, 2])
    ax.plot(x,Exact[75,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,U_pred[75,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])    
    ax.set_title('$t = 0.75$', fontsize = 10)
    
    plt.tight_layout()
    plt.savefig('./burgers_pinn_pytorch.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved to 'burgers_pinn_pytorch.png'")
    plt.show()
    
    print("\nTraining completed successfully!")