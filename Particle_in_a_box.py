import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

L = 10
sigma = 1
p0 = 18.7
hbar = 1e-34
s = sigma / L
p = L * p0 / hbar
m = 1
w0 = np.pi*2*hbar/2*m*L**2
N = 565

def cn_integral(n, L, sigma, p0, hbar):
    y = np.linspace(-0.5, 0.5, 1000)  # Discretize the interval [-1/2, 1/2]
    cos_term = np.cos(n * np.pi * y)
    exp_term = np.exp(-((y / (2 * sigma / L)) ** 2))
    integrand = cos_term * np.cos((L * p0 / hbar) * y) * exp_term
    return np.sqrt(np.sqrt(2) / np.sqrt(np.pi)) * np.sqrt(L / sigma) * np.trapz(integrand, y)

def dn_integral(n, L, sigma, p0, hbar):
    y = np.linspace(-0.5, 0.5, 1000)  # Discretize the interval [-1/2, 1/2]
    sin_term = np.sin(n * np.pi * y)
    exp_term = np.exp(-((y / (2 * sigma / L)) ** 2))
    integrand = 1j * sin_term * np.sin((L * p0 / hbar) * y) * exp_term
    return np.sqrt(np.sqrt(2) / np.sqrt(np.pi)) * np.sqrt(L / sigma) * np.trapz(integrand, y)

def check_sum():
    sum = 0
    for i in range(1,N):
        cn = cn_integral(2*i-1, L, sigma, p0, hbar)
        dn = dn_integral(2*i, L, sigma, p0, hbar)
        sum += np.abs(cn)**2 + np.abs(dn)**2 
    print(sum)

def u(n, x):
    cos_term = np.cos(n * np.pi * x)
    coeff = np.sqrt(2 / L)
    return coeff * cos_term
    
def v(n, x):
    sin_term = np.sin(n * np.pi * x)
    coeff = np.sqrt(2 / L)
    return coeff * sin_term
    
#initialise arrays
cn = np.zeros(N)
dn = np.empty(N, dtype=complex)
for  i in range (1, N):
    cn[i-1] = cn_integral(2*i-1, L, sigma, p0, hbar)
    dn[i-1] = dn_integral(2*i, L, sigma, p0, hbar)

def f(x, t):
    sum = 0
    for i in range (1, N):
        ci = cn[i-1]
        di = dn[i-1]
        ui = u(i, x)
        vi = v(i, x)
        sum += np.abs(ci)**2 * ui**2 + np.abs(di)**2 * vi**2 - 2*ci*abs(di)*ui*vi*np.sin(4*i-1)*w0*t
        for j in range (1, i):
            cj = cn[j-1]
            dj = dn[j-1]
            uj = u(j, x)
            vj = v(j, x)
            sum += 2 * ci * cj * ui * uj * np.cos( (2*i - 1)**2 - (2*j - 1)**2 )*w0*t
            sum += 2 * np.abs(di) * np.abs(dj) * vi * vj * np.cos( (2*i)**2 - (2*j)**2 ) * w0 * t
            sum -= 2 * ci * np.abs(di) * ui * vj * np.sin( (2*i - 1)**2 - (2*j)**2 ) * w0 * t
            sum -= 2 * ci * np.abs(dj) * ui * vj * np.sin( (2*j - 1)**2 - (2*i)**2 ) * w0 * t
    return sum

# Define the range of x values
x_values = np.linspace(-L, L, 50)

# Create a figure and axis object
fig, ax = plt.subplots()

# Initialize an empty plot
line, = ax.plot([], [])

# Function to initialize the plot
def init():
    ax.set_xlim(-L, L)
    ax.set_ylim(0, 1)
    return line,

# Function to update the plot for each frame
def update(frame):
    t = frame * 0.2  # Adjust the speed of the animation by changing the factor here
    y_values = f(x_values, t)
    line.set_data(x_values, y_values)
    return line,

# Create the animation
ani = FuncAnimation(fig, update, frames=range(100), init_func=init, blit=True)

# Display the animation
plt.show()
