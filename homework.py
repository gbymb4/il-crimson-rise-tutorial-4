# -*- coding: utf-8 -*-
"""
Homework Assignment: Exotic Function Fitting with PyTorch
Building on the polynomial fitting exercise, implement parameter fitting 
for three different types of mathematical functions.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducible results
torch.manual_seed(456)
np.random.seed(456)

print("Homework: Exotic Function Fitting with PyTorch")
print("=" * 50)

# =============================================================================
# PROBLEM 1: SINUSOIDAL FUNCTION FITTING
# Target function: y = A * sin(B * x + C) + D
# Where: A = amplitude, B = frequency, C = phase shift, D = vertical offset
# =============================================================================

def problem_1_sinusoidal():
    print("\nPROBLEM 1: Sinusoidal Function Fitting")
    print("-" * 40)
    
    # Generate synthetic sinusoidal data
    n_samples = 200
    # True parameters for y = A * sin(B * x + C) + D
    true_A = 2.5    # amplitude
    true_B = 1.8    # frequency  
    true_C = 0.7    # phase shift
    true_D = 1.0    # vertical offset
    noise_level = 0.3
    
    # Create input data over 2 full periods
    x_data = torch.linspace(0, 4*np.pi, n_samples)
    
    # TODO 1.1: Generate the true sinusoidal relationship with noise
    # Formula: y = A * sin(B * x + C) + D + noise
    y_data = # YOUR CODE HERE
    
    print(f"Generated {n_samples} data points for sinusoidal function")
    print(f"True parameters: A={true_A}, B={true_B}, C={true_C}, D={true_D}")
    
    # TODO 1.2: Initialize parameters to optimize
    # Start with reasonable guesses - sine functions can be tricky!
    # Hint: A should be positive, B around 1-3, C between 0-2π, D can be any value
    param_A = # YOUR CODE HERE - amplitude parameter
    param_B = # YOUR CODE HERE - frequency parameter  
    param_C = # YOUR CODE HERE - phase shift parameter
    param_D = # YOUR CODE HERE - offset parameter
    
    print(f"Initial parameters: A={param_A.item():.3f}, B={param_B.item():.3f}, "
          f"C={param_C.item():.3f}, D={param_D.item():.3f}")
    
    # Training hyperparameters
    learning_rate = 0.005  # Smaller learning rate for stability
    num_epochs = 2000      # More epochs may be needed
    print_every = 200
    
    loss_history = []
    
    # TODO 1.3: Implement the training loop
    # Remember: predictions = param_A * torch.sin(param_B * x_data + param_C) + param_D
    for epoch in range(num_epochs):
        # Forward pass
        predictions = # YOUR CODE HERE
        
        # Calculate loss
        loss = # YOUR CODE HERE
        
        # Backward pass
        # YOUR CODE HERE
        
        # Update parameters
        with torch.no_grad():
            # YOUR CODE HERE - update all 4 parameters
            pass
            
            # Zero gradients
            # YOUR CODE HERE
        
        loss_history.append(loss.item())
        
        if epoch % print_every == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch:4d}: Loss={loss.item():.4f}")
    
    # Results and visualization
    print(f"Final parameters: A={param_A.item():.3f}, B={param_B.item():.3f}, "
          f"C={param_C.item():.3f}, D={param_D.item():.3f}")
    print(f"True parameters:  A={true_A}, B={true_B}, C={true_C}, D={true_D}")
    
    return x_data, y_data, param_A, param_B, param_C, param_D, loss_history, true_A, true_B, true_C, true_D


# =============================================================================
# PROBLEM 2: LOGARITHMIC FUNCTION FITTING  
# Target function: y = A * log(B * x + C) + D
# Where: A = scaling factor, B = input scaling, C = horizontal shift, D = vertical offset
# Note: We need B*x + C > 0 for real logarithm
# =============================================================================

def problem_2_logarithmic():
    print("\nPROBLEM 2: Logarithmic Function Fitting")
    print("-" * 40)
    
    # Generate synthetic logarithmic data
    n_samples = 180
    # True parameters for y = A * log(B * x + C) + D
    true_A = 3.2    # scaling factor
    true_B = 0.8    # input scaling
    true_C = 1.5    # horizontal shift (ensures B*x + C > 0)
    true_D = -0.5   # vertical offset
    noise_level = 0.4
    
    # Create input data - ensure B*x + C > 0
    x_data = torch.linspace(0.1, 8.0, n_samples)
    
    # TODO 2.1: Generate the true logarithmic relationship with noise
    # Formula: y = A * log(B * x + C) + D + noise
    # Use torch.log() for natural logarithm
    y_data = # YOUR CODE HERE
    
    print(f"Generated {n_samples} data points for logarithmic function")
    print(f"True parameters: A={true_A}, B={true_B}, C={true_C}, D={true_D}")
    
    # TODO 2.2: Initialize parameters to optimize
    # Be careful: B and C must ensure B*x + C > 0 for all x values
    param_A = # YOUR CODE HERE
    param_B = # YOUR CODE HERE  
    param_C = # YOUR CODE HERE
    param_D = # YOUR CODE HERE
    
    print(f"Initial parameters: A={param_A.item():.3f}, B={param_B.item():.3f}, "
          f"C={param_C.item():.3f}, D={param_D.item():.3f}")
    
    # Training hyperparameters
    learning_rate = 0.008
    num_epochs = 1500
    print_every = 150
    
    loss_history = []
    
    # TODO 2.3: Implement the training loop
    # Remember: predictions = param_A * torch.log(param_B * x_data + param_C) + param_D
    # Add a small epsilon to prevent log(0): torch.log(param_B * x_data + param_C + 1e-8)
    for epoch in range(num_epochs):
        # YOUR CODE HERE - implement full training loop
        pass
    
    # Results
    print(f"Final parameters: A={param_A.item():.3f}, B={param_B.item():.3f}, "
          f"C={param_C.item():.3f}, D={param_D.item():.3f}")
    print(f"True parameters:  A={true_A}, B={true_B}, C={true_C}, D={true_D}")
    
    return x_data, y_data, param_A, param_B, param_C, param_D, loss_history, true_A, true_B, true_C, true_D


# =============================================================================
# PROBLEM 3: EXPONENTIAL DECAY FUNCTION FITTING
# Target function: y = A * exp(-B * x) + C
# Where: A = initial amplitude, B = decay rate, C = asymptotic value
# =============================================================================

def problem_3_exponential():
    print("\nPROBLEM 3: Exponential Decay Function Fitting")
    print("-" * 40)
    
    # Generate synthetic exponential decay data
    n_samples = 160
    # True parameters for y = A * exp(-B * x) + C
    true_A = 4.0    # initial amplitude
    true_B = 0.6    # decay rate
    true_C = 1.2    # asymptotic value
    noise_level = 0.25
    
    # Create input data
    x_data = torch.linspace(0, 6, n_samples)
    
    # TODO 3.1: Generate the true exponential decay relationship with noise
    # Formula: y = A * exp(-B * x) + C + noise
    # Use torch.exp() for exponential function
    y_data = # YOUR CODE HERE
    
    print(f"Generated {n_samples} data points for exponential decay function")
    print(f"True parameters: A={true_A}, B={true_B}, C={true_C}")
    
    # TODO 3.2: Initialize parameters to optimize
    # Hints: A should be positive, B should be positive for decay, C can be any value
    param_A = # YOUR CODE HERE
    param_B = # YOUR CODE HERE
    param_C = # YOUR CODE HERE
    
    print(f"Initial parameters: A={param_A.item():.3f}, B={param_B.item():.3f}, "
          f"C={param_C.item():.3f}")
    
    # Training hyperparameters  
    learning_rate = 0.01
    num_epochs = 1200
    print_every = 120
    
    loss_history = []
    
    # TODO 3.3: Implement the training loop
    # Remember: predictions = param_A * torch.exp(-param_B * x_data) + param_C
    for epoch in range(num_epochs):
        # YOUR CODE HERE - implement full training loop
        pass
    
    # Results
    print(f"Final parameters: A={param_A.item():.3f}, B={param_B.item():.3f}, "
          f"C={param_C.item():.3f}")
    print(f"True parameters:  A={true_A}, B={true_B}, C={true_C}")
    
    return x_data, y_data, param_A, param_B, param_C, loss_history, true_A, true_B, true_C


# =============================================================================
# MAIN EXECUTION AND VISUALIZATION
# =============================================================================

if __name__ == "__main__":
    # Execute all three problems
    print("Starting homework problems...")
    
    # Problem 1: Sinusoidal
    sin_results = problem_1_sinusoidal()
    
    # Problem 2: Logarithmic  
    log_results = problem_2_logarithmic()
    
    # Problem 3: Exponential
    exp_results = problem_3_exponential()
    
    # TODO: Create comprehensive visualization
    # Create a 3x3 subplot grid showing:
    # Row 1: Data + fitted curves for each function
    # Row 2: Training loss curves  
    # Row 3: Parameter comparison tables
    
    plt.figure(figsize=(18, 12))
    
    # Sinusoidal results
    plt.subplot(3, 3, 1)
    x_sin, y_sin, pA_sin, pB_sin, pC_sin, pD_sin, loss_sin, tA_sin, tB_sin, tC_sin, tD_sin = sin_results
    # TODO: Plot sinusoidal data, fitted curve, and true curve
    plt.title('Sinusoidal Function Fit')
    
    plt.subplot(3, 3, 4)
    # TODO: Plot sinusoidal training loss
    plt.title('Sinusoidal Training Loss')
    
    plt.subplot(3, 3, 7)
    # TODO: Display sinusoidal parameter comparison
    plt.title('Sinusoidal Parameters')
    plt.axis('off')
    
    # Logarithmic results
    plt.subplot(3, 3, 2)
    x_log, y_log, pA_log, pB_log, pC_log, pD_log, loss_log, tA_log, tB_log, tC_log, tD_log = log_results
    # TODO: Plot logarithmic data, fitted curve, and true curve
    plt.title('Logarithmic Function Fit')
    
    plt.subplot(3, 3, 5)
    # TODO: Plot logarithmic training loss
    plt.title('Logarithmic Training Loss')
    
    plt.subplot(3, 3, 8)
    # TODO: Display logarithmic parameter comparison
    plt.title('Logarithmic Parameters')
    plt.axis('off')
    
    # Exponential results
    plt.subplot(3, 3, 3)
    x_exp, y_exp, pA_exp, pB_exp, pC_exp, loss_exp, tA_exp, tB_exp, tC_exp = exp_results
    # TODO: Plot exponential data, fitted curve, and true curve
    plt.title('Exponential Decay Function Fit')
    
    plt.subplot(3, 3, 6)
    # TODO: Plot exponential training loss
    plt.title('Exponential Training Loss')
    
    plt.subplot(3, 3, 9)
    # TODO: Display exponential parameter comparison
    plt.title('Exponential Parameters')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # TODO: Calculate and report parameter errors for all three functions
    print("\n" + "="*60)
    print("HOMEWORK SUMMARY")
    print("="*60)
    
    # Calculate errors for each function and provide overall assessment
    # YOUR CODE HERE
    
    print("\nCongratulations on completing the exotic function fitting homework!")
    print("This exercise demonstrates the versatility of gradient descent")
    print("for fitting parameters of diverse mathematical functions.")


# =============================================================================
# REFLECTION QUESTIONS (Answer in comments or separate document)
# =============================================================================
"""
1. Which function was most challenging to fit and why?

2. How did the learning rates need to be adjusted for different functions?
   What happens if the learning rate is too high or too low?

3. What strategies could you use if a function wasn't converging well?

4. How would you extend this approach to fit functions with more parameters?

5. BONUS: Can you think of other exotic functions you'd like to try fitting?
   (e.g., Gaussian, power law, sigmoid, etc.)
   Extend the code to be able to take in any generic function with pre-defined
   parameters, generate some data for this function, then fit a learned curve
   to the data.
"""