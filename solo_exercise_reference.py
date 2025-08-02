# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 13:25:27 2025

@author: taske
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducible results
torch.manual_seed(123)
np.random.seed(123)

print("Solo Exercise: Polynomial Curve Fitting with PyTorch")
print("=" * 55)

# Generate synthetic polynomial data
n_samples = 150
# True cubic polynomial: y = ax³ + bx² + cx + d
true_a = 0.5   # coefficient for x³
true_b = -1.2  # coefficient for x²  
true_c = 0.8   # coefficient for x
true_d = 2.0   # constant term
noise_level = 0.5

# Create input data
x_data = torch.linspace(-2, 2, n_samples)

# TODO 1: Generate the true polynomial relationship with noise
# Formula: y = true_a * x³ + true_b * x² + true_c * x + true_d + noise
# Hint: Use torch.randn(n_samples) for noise
y_data = true_a * x_data**3 + true_b * x_data**2 + true_c * x_data + true_d + noise_level * torch.randn(n_samples)

print(f"Generated {n_samples} data points for cubic polynomial")
print(f"True coefficients: a={true_a}, b={true_b}, c={true_c}, d={true_d}")

# TODO 2: Initialize parameters to optimize
# You need 4 parameters: a, b, c, d (coefficients for x³, x², x, constant)
# Start with small random values and set requires_grad=True
param_a = torch.tensor([0.1], requires_grad=True, dtype=torch.float32)  # x³ coefficient
param_b = torch.tensor([0.2], requires_grad=True, dtype=torch.float32)  # x² coefficient  
param_c = torch.tensor([0.1], requires_grad=True, dtype=torch.float32)  # x coefficient
param_d = torch.tensor([0.5], requires_grad=True, dtype=torch.float32)  # constant term

print(f"Initial parameters: a={param_a.item():.3f}, b={param_b.item():.3f}, "
      f"c={param_c.item():.3f}, d={param_d.item():.3f}")

# Training hyperparameters
learning_rate = 0.01  
num_epochs = 1000
print_every = 100

# Lists to store training history
loss_history = []

print(f"\nStarting training for {num_epochs} epochs...")

"""
PSEUDOCODE FOR TRAINING LOOP:

for each epoch from 0 to num_epochs:
    # FORWARD PASS
    1. Calculate predictions using current parameters
       predictions = param_a * x_data³ + param_b * x_data² + param_c * x_data + param_d
    
    # LOSS CALCULATION  
    2. Calculate Mean Squared Error loss
       loss = mean((predictions - y_data)²)
    
    # BACKWARD PASS
    3. Calculate gradients automatically
       loss.backward()
    
    # PARAMETER UPDATE
    4. Update all parameters using their gradients
       with torch.no_grad():
           param_a -= learning_rate * param_a.grad
           param_b -= learning_rate * param_b.grad  
           param_c -= learning_rate * param_c.grad
           param_d -= learning_rate * param_d.grad
    
    # GRADIENT CLEANUP
    5. Zero all gradients for next iteration
       param_a.grad.zero_()
       param_b.grad.zero_()
       param_c.grad.zero_()
       param_d.grad.zero_()
    
    # LOGGING
    6. Store loss and print progress periodically
"""

# TODO 3: Implement the training loop following the pseudocode above
for epoch in range(num_epochs):
    # Forward pass - calculate predictions using cubic polynomial
    predictions = param_a * x_data**3 + param_b * x_data**2 + param_c * x_data + param_d
    
    # Calculate loss (Mean Squared Error)
    loss = torch.mean((predictions - y_data)**2)
    
    # Backward pass - calculate gradients
    loss.backward()
    
    # Update parameters using gradients
    with torch.no_grad():
        param_a -= learning_rate * param_a.grad
        param_b -= learning_rate * param_b.grad  
        param_c -= learning_rate * param_c.grad
        param_d -= learning_rate * param_d.grad
        
        # Zero gradients for next iteration (important!)
        param_a.grad.zero_()
        param_b.grad.zero_()
        param_c.grad.zero_()
        param_d.grad.zero_()
    
    # Store history and print progress
    loss_history.append(loss.item())
    
    if epoch % print_every == 0 or epoch == num_epochs - 1:
        print(f"Epoch {epoch:4d}: Loss={loss.item():.4f}")

# Final results
print("\nTraining completed!")
print(f"Final parameters: a={param_a.item():.3f}, b={param_b.item():.3f}, "
      f"c={param_c.item():.3f}, d={param_d.item():.3f}")
print(f"True parameters:  a={true_a}, b={true_b}, c={true_c}, d={true_d}")
print(f"Final loss: {loss.item():.4f}")

# TODO 4: Create visualization comparing fitted curve to true curve and data
plt.figure(figsize=(15, 5))

# Plot 1: Data and fitted curves
plt.subplot(1, 3, 1)
plt.scatter(x_data.numpy(), y_data.numpy(), alpha=0.6, color='blue', label='Data', s=20)

# Plot the fitted polynomial curve using final parameters
fitted_curve = param_a * x_data**3 + param_b * x_data**2 + param_c * x_data + param_d
plt.plot(x_data.numpy(), fitted_curve.detach().numpy(), 'r-', linewidth=2, 
         label='Fitted Polynomial')

# Plot true curve for comparison  
true_curve = true_a * x_data**3 + true_b * x_data**2 + true_c * x_data + true_d
plt.plot(x_data.numpy(), true_curve.numpy(), 'g--', linewidth=2, 
         label='True Polynomial')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Polynomial Curve Fitting Results')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Training loss
plt.subplot(1, 3, 2)
plt.plot(loss_history, 'b-', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Training Loss Over Time')
plt.grid(True, alpha=0.3)

# Plot 3: Parameter convergence
plt.subplot(1, 3, 3)
# This would require storing parameter history - optional advanced feature
plt.text(0.1, 0.8, "Final Parameters:", fontsize=12, fontweight='bold')
plt.text(0.1, 0.7, f"a = {param_a.item():.3f} (true: {true_a})", fontsize=10)
plt.text(0.1, 0.6, f"b = {param_b.item():.3f} (true: {true_b})", fontsize=10)
plt.text(0.1, 0.5, f"c = {param_c.item():.3f} (true: {true_c})", fontsize=10)
plt.text(0.1, 0.4, f"d = {param_d.item():.3f} (true: {true_d})", fontsize=10)
plt.text(0.1, 0.2, f"Final Loss: {loss.item():.4f}", fontsize=10)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title('Parameter Summary')
plt.axis('off')

plt.tight_layout()
plt.show()

# Calculate parameter errors
a_error = abs(param_a.item() - true_a)
b_error = abs(param_b.item() - true_b)  
c_error = abs(param_c.item() - true_c)
d_error = abs(param_d.item() - true_d)

print("\nParameter errors:")
print(f"a: {a_error:.3f}, b: {b_error:.3f}, c: {c_error:.3f}, d: {d_error:.3f}")

if max(a_error, b_error, c_error, d_error) < 0.1:
    print("🎉 Excellent! All parameters converged very close to true values!")
elif max(a_error, b_error, c_error, d_error) < 0.2:
    print("✅ Good job! Parameters are reasonably close to true values.")
else:
    print("⚠️  Parameters could be closer. Try adjusting learning rate or more epochs.")

# BONUS CHALLENGE (Optional):
# Try implementing these advanced features:
# 1. Store parameter history and plot parameter convergence over time
# 2. Implement early stopping when loss improvement < threshold
# 3. Add L2 regularization to prevent overfitting
# 4. Experiment with different polynomial degrees (quadratic, quartic)
# 5. Try different optimizers (SGD with momentum, Adam-like updates)