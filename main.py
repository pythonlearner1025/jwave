import numpy as np
import matplotlib.pyplot as plt

# Parameters
beta = 0.1  # Shift parameter
k = 10.0    # Wavenumber
x = np.linspace(0, 10, 1000)  # Spatial domain

# Compute complex shift
shift = np.sqrt(1 + 1j * beta)
k_real = k * np.real(shift)
k_imag = k * np.imag(shift)

# Original oscillatory wave (real part)
u_original = np.cos(k * x)

# Shifted wave (real part with damping)
u_shifted = np.exp(-k_imag * x) * np.cos(k_real * x)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, u_original, label='Original: cos(k x)', color='blue')
plt.plot(x, u_shifted, label=f'Shifted (β={beta}): e^{-k_imag*x} cos(k_real x)', color='red')
plt.title('Effect of Complex Shift on Oscillatory Wave')
plt.xlabel('x')
plt.ylabel('Real Part of u(x)')
plt.legend()
plt.grid(True)
plt.show()