# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 13:16:52 2025

@author: taske
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducible results
torch.manual_seed(42)
np.random.seed(42)

print("PyTorch Direct Parameter Optimization - Linear Regression")
print("=" * 60)

# Generate synthetic linear data with noise
n_samples = 100
true_slope = 2.5
true_intercept = 1.0
noise_level = 0.3

# Create input data
x_data = torch.linspace(-2, 2, n_samples)
# Generate true linear relationship with noise
y_data = true_slope * x_data + true_intercept + noise_level * torch.randn(n_samples)

print(f"Generated {n_samples} data points")
print(f"True parameters: slope={true_slope}, intercept={true_intercept}")

# Initialize parameters to optimize (start with random values)
slope = torch.tensor([0.1], requires_grad=True, dtype=torch.float32)
intercept = torch.tensor([0.5], requires_grad=True, dtype=torch.float32)

print(f"Initial parameters: slope={slope.item():.3f}, intercept={intercept.item():.3f}")

# Training hyperparameters
learning_rate = 0.01
num_epochs = 200
print_every = 20

# Lists to store training history
loss_history = []
slope_history = []
intercept_history = []

print(f"\nStarting training for {num_epochs} epochs...")
print("New plots will be created every 10 epochs to show progress!")

# Training loop
for epoch in range(num_epochs):
    # Forward pass: make predictions using current parameters
    predictions = slope * x_data + intercept
    
    # Calculate loss (Mean Squared Error)
    loss = torch.mean((predictions - y_data)**2)
    
    # Backward pass: calculate gradients
    loss.backward()
    
    # Update parameters using gradients (manual gradient descent)
    with torch.no_grad():
        slope -= learning_rate * slope.grad
        intercept -= learning_rate * intercept.grad
        
        # Zero gradients for next iteration (important!)
        slope.grad.zero_()
        intercept.grad.zero_()
    
    # Store history for plotting
    loss_history.append(loss.item())
    slope_history.append(slope.item())
    intercept_history.append(intercept.item())
    
    # Print progress
    if epoch % print_every == 0 or epoch == num_epochs - 1:
        print(f"Epoch {epoch:3d}: Loss={loss.item():.4f}, "
              f"slope={slope.item():.3f}, intercept={intercept.item():.3f}")
    
    # Create new plots every 10 epochs
    if epoch % 10 == 0 or epoch == num_epochs - 1:
        # Close any existing plots to avoid memory issues
        plt.close('all')
        
        # Create entirely new figure and axes
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Data and current fitted line
        ax1.scatter(x_data.numpy(), y_data.numpy(), alpha=0.6, color='blue', label='Data')
        ax1.plot(x_data.numpy(), predictions.detach().numpy(), 'r-', linewidth=2, 
                label=f'Fitted: y={slope.item():.2f}x + {intercept.item():.2f}')
        ax1.plot(x_data.numpy(), true_slope * x_data.numpy() + true_intercept, 
                'g--', linewidth=2, label=f'True: y={true_slope}x + {true_intercept}')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title(f'Linear Regression - Epoch {epoch}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Training progress
        ax2.plot(loss_history, 'b-', linewidth=2, label='Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Mean Squared Error')
        ax2.set_title('Training Loss Over Time')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

# Final results
print("\nTraining completed!")
print(f"Final parameters: slope={slope.item():.3f}, intercept={intercept.item():.3f}")
print(f"True parameters:  slope={true_slope}, intercept={true_intercept}")
print(f"Final loss: {loss.item():.4f}")

# Calculate parameter errors
slope_error = abs(slope.item() - true_slope)
intercept_error = abs(intercept.item() - true_intercept)
print(f"Parameter errors: slope={slope_error:.3f}, intercept={intercept_error:.3f}")

# Show final comparison plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(x_data.numpy(), y_data.numpy(), alpha=0.6, color='blue', label='Data')
plt.plot(x_data.numpy(), predictions.detach().numpy(), 'r-', linewidth=2, 
         label=f'Fitted: y={slope.item():.2f}x + {intercept.item():.2f}')
plt.plot(x_data.numpy(), true_slope * x_data.numpy() + true_intercept, 
         'g--', linewidth=2, label=f'True: y={true_slope}x + {true_intercept}')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Final Linear Regression Result')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(loss_history, 'b-', linewidth=2, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Loss Convergence')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

print("\nKey Observations:")
print("1. The red line gradually moved toward the green dashed line (true relationship)")
print("2. The loss decreased over time, showing the optimization working")
print("3. PyTorch automatically calculated gradients for us")
print("4. We had full control over the training process")
print("5. New plots were created every 10 epochs to show training progress")