import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the function f(x, t)
def f(x, t):
    return np.sin(x + t)

# Define the range of x values
x_values = np.linspace(0, 2*np.pi, 100)

# Create a figure and axis object
fig, ax = plt.subplots()

# Initialize an empty plot
line, = ax.plot([], [])

# Function to initialize the plot
def init():
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1.1, 1.1)
    return line,

# Function to update the plot for each frame
def update(frame):
    t = frame * 0.1  # Adjust the speed of the animation by changing the factor here
    y_values = f(x_values, t)
    line.set_data(x_values, y_values)
    return line,

# Create the animation
ani = FuncAnimation(fig, update, frames=range(100), init_func=init, blit=True)

# Display the animation
plt.show()
