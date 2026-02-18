import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import time
from torch.autograd import grad
from data_process import * 

class PhysicsInformedNN():
    def __init__(self, pde_points, i_points, b_points, i_data, b_data, layers, device, nu):
        
        data_points = np.vstack([i_points, b_points])
        data_values = np.concatenate([i_data, b_data])

        self.device = device

        # data
        self.x_f = torch.tensor(pde_points[:, 0:1], requires_grad=True).float().to(device)
        self.t_f = torch.tensor(pde_points[:, 1:2], requires_grad=True).float().to(device)
        
        # self.x_i = torch.tensor(i_points[:, 0:1], requires_grad=True).float().to(device)
        # self.t_i = torch.tensor(i_points[:, 1:2], requires_grad=True).float().to(device)
        
        # self.x_b = torch.tensor(b_points[:, 0:1], requires_grad=True).float().to(device)
        # self.t_b = torch.tensor(b_points[:, 1:2], requires_grad=True).float().to(device)

        # self.u_i = torch.tensor(i_data).float().to(device)
        # self.u_b = torch.tensor(b_data).float().to(device)
        
        self.x_data = torch.tensor(data_points[:, 0:1], requires_grad=True).float().to(device)
        self.t_data = torch.tensor(data_points[:, 1:2], requires_grad=True).float().to(device)
        self.u_data = torch.tensor(data_values).float().to(device).view(-1, 1)
        
        self.lb = torch.tensor([X_MIN, T_MIN]).float().to(device)  
        self.ub = torch.tensor([X_MAX, T_MAX]).float().to(device) 
        
        self.layers = layers
        self.nu = nu

        # Initialize Neural Network
        self.model = self.build_model(layers).to(device)

        self.optimizer_Adam = torch.optim.Adam(self.model.parameters(), lr=0.001)
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
        # ub_pred = self.net_u(self.x_b, self.t_b)
        # loss_ub = torch.mean((self.u_b - ub_pred)**2)
        
        # ui_pred = self.net_u(self.x_i, self.t_i)
        # loss_ui = torch.mean((self.u_i - ui_pred)**2)
        
        ud_pred = self.net_u(self.x_data, self.t_data)
        loss_d = torch.mean((self.u_data - ud_pred)**2)
        
        # Physics loss
        f_pred = self.net_f(self.x_f, self.t_f)
        loss_f = torch.mean(f_pred**2)
        
        # Total loss
        loss =  loss_d + loss_f
        
        return loss, loss_d, loss_f

    def train(self, epochs=10000):
        """Training loop"""
        self.model.train()
        
        for epoch in range(epochs):
            self.optimizer_Adam.zero_grad()
            
            loss, loss_d, loss_f = self.loss_fn()
            
            loss.backward()
            self.optimizer_Adam.step()
            
            self.loss_history.append(loss.item())
            
            if epoch % 1000 == 0:
                print(f'Epoch {epoch:5d}: Loss = {loss.item():.6e}, Loss_d = {loss_d.item():.6e}, Loss_f = {loss_f.item():.6e}')
    
    def train_lbfgs(self, max_iter=50000):
        """Training with L-BFGS optimizer (more accurate but slower)"""
        self.model.train()
        
        def closure():
            self.optimizer.zero_grad()
            loss, loss_d, loss_f = self.loss_fn() 
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
        
        x = torch.tensor(X_star[:,0:1], dtype=torch.float32, requires_grad=True).to(self.device)
        t = torch.tensor(X_star[:,1:2], dtype=torch.float32, requires_grad=True).to(self.device)
        
        with torch.no_grad():
            u_pred = self.net_u(x, t)
        
        # For f_pred we need gradients
        x.requires_grad = True
        t.requires_grad = True
        f_pred = self.net_f(x, t)
        
        return u_pred.cpu().numpy(), f_pred.detach().cpu().numpy()
