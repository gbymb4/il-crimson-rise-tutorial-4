# Machine Learning Session 4: PyTorch Direct Parameter Optimization
## Linear & Polynomial Regression Through Gradient Descent

### Session Overview
**Duration**: 1 hour  
**Prerequisites**: Completed Session 3 (Neural networks with scikit-learn)  
**Goal**: Understand direct parameter optimization in PyTorch without neural network layers  
**Focus**: Manual gradient descent implementation for curve fitting

### Session Timeline
| Time      | Activity                                    |
| --------- | ------------------------------------------- |
| 0:00 - 0:05 | 1. Touching Base & Session Overview    |
| 0:05 - 0:15 | 2. From scikit-learn to PyTorch - What's Different?           |
| 0:15 - 0:25 | 3. Direct Parameter Optimization Concepts     |
| 0:25 - 0:35 | 4. Linear Regression Example with Live Visualization   |
| 0:35 - 0:55 | 5. Solo Exercise: Polynomial Curve Fitting                |
| 0:55 - 1:00 | 6. Wrap-up & Next Steps                    |

---

## 1. Touching Base & Session Overview (5 minutes)

### Quick Check-in
- Review Session 3's neural network classification with scikit-learn
- Ensure PyTorch is installed (`pip install torch matplotlib`)
- Preview today's shift from high-level to low-level control

### Today's Learning Objectives
By the end of this session, you will be able to:
- Understand the difference between high-level ML libraries and direct parameter optimization
- Implement gradient descent manually using PyTorch's automatic differentiation
- Optimize parameters for known mathematical functions (linear and polynomial)
- Visualize the optimization process in real-time
- Write training loops with explicit gradient handling

---

## 2. From scikit-learn to PyTorch - What's Different? (10 minutes)

### Key Conceptual Shifts

**scikit-learn Approach (Session 3)**
- High-level: `model.fit(X, y)` handles everything
- Pre-built architectures and optimizers
- Focus on choosing the right model and hyperparameters
- Black box: we don't see the optimization happening

**PyTorch Approach (Today)**
- Low-level: We explicitly define parameters and optimization steps
- Direct control over every aspect of training
- We see and control the gradient descent process
- Transparent: we implement the training loop ourselves

**Why Learn This Way?**
- Deeper understanding of how ML actually works "under the hood"
- Preparation for advanced techniques and custom models
- Better debugging skills when things go wrong
- Foundation for understanding research papers and advanced architectures

### The Core Difference
Instead of training neural networks, we're optimizing parameters directly for known mathematical functions:
- Linear: `y = mx + b` (optimize m and b)
- Polynomial: `y = ax³ + bx² + cx + d` (optimize a, b, c, d)

---

## 3. Direct Parameter Optimization Concepts (10 minutes)

### The Problem Setup
1. **Generate synthetic data** following a known pattern (with noise)
2. **Initialize parameters** randomly
3. **Define a loss function** (how wrong our predictions are)
4. **Use gradient descent** to adjust parameters
5. **Repeat until convergence**

### Key PyTorch Concepts

**Tensors with Gradients**
```python
# Create parameters that PyTorch will track gradients for
slope = torch.tensor([0.5], requires_grad=True)
intercept = torch.tensor([0.0], requires_grad=True)
```

**Forward Pass**
```python
# Make predictions using current parameters
predictions = slope * x_data + intercept
```

**Loss Calculation**
```python
# Calculate how wrong we are (Mean Squared Error)
loss = torch.mean((predictions - y_data)**2)
```

**Backward Pass**
```python
# Calculate gradients automatically
loss.backward()
```

**Parameter Update**
```python
# Update parameters using gradients
with torch.no_grad():
    slope -= learning_rate * slope.grad
    intercept -= learning_rate * intercept.grad
    slope.grad.zero_()  # Clear gradients for next iteration
    intercept.grad.zero_()
```

### Visualization Strategy
- Plot the data points (scatter)
- Plot the current fitted line/curve
- Update the plot every epoch to see optimization in action
- Show loss decreasing over time

---

## 4. Linear Regression Example with Live Visualization (10 minutes)

*This example script will be demonstrated live, showing real-time optimization*

### Complete Linear Regression Example

```python
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

# Setup interactive plotting
plt.ion()  # Turn on interactive mode
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Lists to store training history
loss_history = []
slope_history = []
intercept_history = []

print(f"\nStarting training for {num_epochs} epochs...")
print("Watch the live visualization to see the line fitting to the data!")

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
    
    # Live visualization every 5 epochs
    if epoch % 5 == 0 or epoch == num_epochs - 1:
        # Clear previous plots
        ax1.clear()
        ax2.clear()
        
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
        plt.pause(0.1)  # Brief pause to update display

plt.ioff()  # Turn off interactive mode
plt.show()

# Final results
print(f"\nTraining completed!")
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
```

### Key Points to Explain During Demo
- **Parameter initialization**: Why we start with random values
- **`requires_grad=True`**: Tells PyTorch to track gradients
- **Forward pass**: Making predictions with current parameters
- **Loss calculation**: Quantifying how wrong we are
- **`loss.backward()`**: Automatic differentiation in action
- **Manual parameter updates**: Implementing gradient descent ourselves
- **`zero_grad()`**: Why we must clear gradients each iteration
- **Live visualization**: Seeing optimization happen in real-time

---

## 5. Solo Exercise: Polynomial Curve Fitting (20 minutes)

### Exercise Instructions
Your task is to implement polynomial curve fitting using the same direct parameter optimization approach. You'll work with cubic polynomials and need to optimize 4 parameters instead of 2.

### Exercise Script with TODOs

```python
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
y_data = None  # TODO: Implement this

print(f"Generated {n_samples} data points for cubic polynomial")
print(f"True coefficients: a={true_a}, b={true_b}, c={true_c}, d={true_d}")

# TODO 2: Initialize parameters to optimize
# You need 4 parameters: a, b, c, d (coefficients for x³, x², x, constant)
# Start with small random values and set requires_grad=True
param_a = None  # TODO: Create parameter for x³ coefficient
param_b = None  # TODO: Create parameter for x² coefficient  
param_c = None  # TODO: Create parameter for x coefficient
param_d = None  # TODO: Create parameter for constant term

print(f"Initial parameters: a={param_a.item():.3f}, b={param_b.item():.3f}, "
      f"c={param_c.item():.3f}, d={param_d.item():.3f}")

# Training hyperparameters
learning_rate = 0.001  # Smaller learning rate for polynomial fitting
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
    # TODO: Forward pass - calculate predictions
    predictions = None  # TODO: Implement cubic polynomial prediction
    
    # TODO: Calculate loss (Mean Squared Error)
    loss = None  # TODO: Implement MSE loss
    
    # TODO: Backward pass - calculate gradients
    # TODO: Use loss.backward()
    
    # TODO: Update parameters using gradients
    with torch.no_grad():
        # TODO: Update param_a
        # TODO: Update param_b  
        # TODO: Update param_c
        # TODO: Update param_d
        
        # TODO: Zero all gradients
        pass
    
    # Store history and print progress
    loss_history.append(loss.item())
    
    if epoch % print_every == 0 or epoch == num_epochs - 1:
        print(f"Epoch {epoch:4d}: Loss={loss.item():.4f}")

# Final results
print(f"\nTraining completed!")
print(f"Final parameters: a={param_a.item():.3f}, b={param_b.item():.3f}, "
      f"c={param_c.item():.3f}, d={param_d.item():.3f}")
print(f"True parameters:  a={true_a}, b={true_b}, c={true_c}, d={true_d}")
print(f"Final loss: {loss.item():.4f}")

# TODO 4: Create visualization comparing fitted curve to true curve and data
plt.figure(figsize=(15, 5))

# Plot 1: Data and fitted curves
plt.subplot(1, 3, 1)
plt.scatter(x_data.numpy(), y_data.numpy(), alpha=0.6, color='blue', label='Data', s=20)

# TODO: Plot the fitted polynomial curve
fitted_curve = None  # TODO: Calculate fitted curve using final parameters
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
plt.text(0.1, 0.8, f"Final Parameters:", fontsize=12, fontweight='bold')
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

print(f"\nParameter errors:")
print(f"a: {a_error:.3f}, b: {b_error:.3f}, c: {c_error:.3f}, d: {d_error:.3f}")

if max(a_error, b_error, c_error, d_error) < 0.1:
    print("🎉 Excellent! All parameters converged very close to true values!")
elif max(a_error, b_error, c_error, d_error) < 0.2:
    print("✅ Good job! Parameters are reasonably close to true values.")
else:
    print("⚠️  Parameters could be closer. Try adjusting learning rate or more epochs.")
```

### Expected Outcomes
- Students should achieve parameter errors < 0.2 for all coefficients
- Loss should decrease steadily and plateau
- Fitted curve should closely match the true polynomial
- Students gain hands-on experience with gradient descent mechanics

### Hints for Common Issues
- **Gradients exploding**: Reduce learning rate
- **Slow convergence**: Increase learning rate or epochs
- **Parameters not updating**: Check `requires_grad=True` and `zero_grad()`
- **Visualization errors**: Ensure `detach()` when converting to numpy

---

## 6. Wrap-up & Next Steps (5 minutes)

### Key Takeaways
- Direct parameter optimization gives us complete control over the learning process
- PyTorch's automatic differentiation (`autograd`) handles gradient calculation
- The training loop pattern: forward pass → loss → backward pass → parameter update → zero gradients
- Real-time visualization helps us understand what the optimizer is doing
- This foundation prepares us for understanding neural networks at a deeper level

### What We've Accomplished
- Implemented gradient descent from scratch using PyTorch
- Optimized parameters for both linear and polynomial functions
- Visualized the optimization process in real-time
- Gained intuition for how machine learning actually learns

### Progression from Session 3
- **Session 3**: High-level neural networks with scikit-learn (`model.fit()`)
- **Session 4**: Low-level parameter optimization with PyTorch (manual training loops)
- **Next**: Combining these concepts for custom neural network architectures

### Next Session Preview
In Session 5, we'll bridge today's manual approach with neural networks:
- Building custom neural networks using PyTorch's `nn.Module`
- Automatic optimization with `torch.optim`
- Handling more complex architectures and datasets
- Introduction to convolutional layers for image data

### Homework/Practice (Optional)
1. Experiment with different polynomial degrees (quadratic, quartic)
2. Try different learning rates and observe convergence behavior
3. Add momentum to the parameter updates for faster convergence
4. Implement early stopping when loss plateaus

### Questions & Discussion
- How did manual gradient descent compare to scikit-learn's automatic training?
- Which was more challenging: linear or polynomial fitting, and why?
- What insights did you gain from watching the optimization in real-time?
- How might you apply this direct optimization approach to other mathematical problems?