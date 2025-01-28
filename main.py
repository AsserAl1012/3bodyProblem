# Import required libraries
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('dark_background')

# Style settings for modern design
def apply_style(widget):
    style = ttk.Style()
    style.theme_use("clam")  # Clean and modern look
    style.configure(
        "TScale",
        background="#1e1e2e",
        foreground="#f8f8f2",
        troughcolor="#3a3a44",
        sliderlength=20,
        sliderthickness=10,
        borderwidth=2,
        relief="flat",
    )
    widget.configure(style="TScale")

# Define the main simulation class
class ThreeBodyProblem:
    def __init__(self, m1, m2, m3, p1_start, v1_start, p2_start, v2_start, p3_start, v3_start, delta_t, steps):
        # Masses
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        
        # Ensure positions and velocities are NumPy arrays
        self.p1_start = np.array(p1_start, dtype=float)
        self.v1_start = np.array(v1_start, dtype=float)
        self.p2_start = np.array(p2_start, dtype=float)
        self.v2_start = np.array(v2_start, dtype=float)
        self.p3_start = np.array(p3_start, dtype=float)
        self.v3_start = np.array(v3_start, dtype=float)
        
        # Simulation parameters
        self.delta_t = delta_t
        self.steps = steps
        
        # Trajectory arrays
        self.p1 = np.zeros((steps, 3))
        self.v1 = np.zeros((steps, 3))
        self.p2 = np.zeros((steps, 3))
        self.v2 = np.zeros((steps, 3))
        self.p3 = np.zeros((steps, 3))
        self.v3 = np.zeros((steps, 3))
        
        # Set initial positions and velocities
        self.p1[0], self.v1[0] = self.p1_start, self.v1_start
        self.p2[0], self.v2[0] = self.p2_start, self.v2_start
        self.p3[0], self.v3[0] = self.p3_start, self.v3_start


    def accelerations(self, p1, p2, p3):
        dv1 = -9.8 * self.m2 * (p1 - p2) / (np.linalg.norm(p1 - p2)**3) - \
              9.8 * self.m3 * (p1 - p3) / (np.linalg.norm(p1 - p3)**3)
        
        dv2 = -9.8 * self.m3 * (p2 - p3) / (np.linalg.norm(p2 - p3)**3) - \
              9.8 * self.m1 * (p2 - p1) / (np.linalg.norm(p2 - p1)**3)
        
        dv3 = -9.8 * self.m1 * (p3 - p1) / (np.linalg.norm(p3 - p1)**3) - \
              9.8 * self.m2 * (p3 - p2) / (np.linalg.norm(p3 - p2)**3)
        
        return dv1, dv2, dv3

    def evolve_system(self):
        for i in range(self.steps - 1):
            dv1, dv2, dv3 = self.accelerations(self.p1[i], self.p2[i], self.p3[i])

            self.v1[i + 1] = self.v1[i] + dv1 * self.delta_t
            self.v2[i + 1] = self.v2[i] + dv2 * self.delta_t
            self.v3[i + 1] = self.v3[i] + dv3 * self.delta_t

            self.p1[i + 1] = self.p1[i] + self.v1[i] * self.delta_t
            self.p2[i + 1] = self.p2[i] + self.v2[i] * self.delta_t
            self.p3[i + 1] = self.p3[i] + self.v3[i] * self.delta_t

    def animate_trajectories(self):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('black')

        ax.set_xlim([-20, 20])
        ax.set_ylim([-20, 20])
        ax.set_zlim([-20, 20])

        line1, = ax.plot([], [], [], 'o-', color='red', label='Body 1')
        line2, = ax.plot([], [], [], 'o-', color='white', label='Body 2')
        line3, = ax.plot([], [], [], 'o-', color='blue', label='Body 3')
        point1, = ax.plot([], [], [], 'o', color='red')
        point2, = ax.plot([], [], [], 'o', color='white')
        point3, = ax.plot([], [], [], 'o', color='blue')
        ax.legend()

        def init():
            line1.set_data([], [])
            line1.set_3d_properties([])
            line2.set_data([], [])
            line2.set_3d_properties([])
            line3.set_data([], [])
            line3.set_3d_properties([])
            point1.set_data([], [])
            point1.set_3d_properties([])
            point2.set_data([], [])
            point2.set_3d_properties([])
            point3.set_data([], [])
            point3.set_3d_properties([])
            return line1, line2, line3, point1, point2, point3

        def update(frame):
            line1.set_data(self.p1[:frame, 0], self.p1[:frame, 1])
            line1.set_3d_properties(self.p1[:frame, 2])
            line2.set_data(self.p2[:frame, 0], self.p2[:frame, 1])
            line2.set_3d_properties(self.p2[:frame, 2])
            line3.set_data(self.p3[:frame, 0], self.p3[:frame, 1])
            line3.set_3d_properties(self.p3[:frame, 2])
            point1.set_data([self.p1[frame, 0]], [self.p1[frame, 1]])
            point1.set_3d_properties([self.p1[frame, 2]])
            point2.set_data([self.p2[frame, 0]], [self.p2[frame, 1]])
            point2.set_3d_properties([self.p2[frame, 2]])
            point3.set_data([self.p3[frame, 0]], [self.p3[frame, 1]])
            point3.set_3d_properties([self.p3[frame, 2]])
            return line1, line2, line3, point1, point2, point3

        ani = FuncAnimation(fig, update, frames=range(0, self.steps, 100), init_func=init, blit=True, interval=50)
        plt.show()

# GUI for Parameter Tuning
def create_gui():
    def start_simulation():
        # Retrieve all values from sliders and inputs
        m1 = float(mass1_value.get())
        m2 = float(mass2_value.get())
        m3 = float(mass3_value.get())
        delta_t = float(delta_t_value.get())
        steps = int(steps_value.get())

        # Ensure positions and velocities are floats
        p1_start = [float(p1x.get()), float(p1y.get()), float(p1z.get())]
        v1_start = [float(v1x.get()), float(v1y.get()), float(v1z.get())]
        p2_start = [float(p2x.get()), float(p2y.get()), float(p2z.get())]
        v2_start = [float(v2x.get()), float(v2y.get()), float(v2z.get())]
        p3_start = [float(p3x.get()), float(p3y.get()), float(p3z.get())]
        v3_start = [float(v3x.get()), float(v3y.get()), float(v3z.get())]

        # Run the simulation
        simulation = ThreeBodyProblem(
            m1, m2, m3, p1_start, v1_start, p2_start, v2_start, p3_start, v3_start, delta_t, steps
        )
        simulation.evolve_system()
        simulation.animate_trajectories()


    # Main window
    root = tk.Tk()
    root.title("Three-Body Problem Parameter Tuning")
    root.geometry("600x800")
    root.configure(bg="#1e1e2e")

    ttk.Label(root, text="Three-Body Problem Parameters", font=("Arial", 16), background="#1e1e2e", foreground="white").pack(pady=10)

    # Helper function to create sliders with real-time value display
    def create_slider(parent, label_text, from_, to, initial, resolution=1, is_float=False):
        frame = ttk.Frame(parent)
        frame.pack(pady=5, anchor="w")
        label = ttk.Label(frame, text=label_text, background="#1e1e2e", foreground="white")
        label.grid(row=0, column=0, sticky="w")

        value = tk.StringVar(value=str(initial))  # Variable to display the current value
        slider = ttk.Scale(frame, from_=from_, to=to, orient="horizontal", length=300)
        slider.set(initial)
        slider.grid(row=0, column=1, padx=5)

        value_label = ttk.Label(frame, textvariable=value, background="#1e1e2e", foreground="white")
        value_label.grid(row=0, column=2)

        # Update the value in real-time as the slider moves
        def update_value(event=None):
            current_value = slider.get()
            value.set(f"{current_value:.4f}" if is_float else f"{int(current_value)}")

        slider.bind("<Motion>", update_value)
        slider.bind("<ButtonRelease-1>", update_value)

        apply_style(slider)
        return slider, value

    # Mass sliders
    mass1_slider, mass1_value = create_slider(root, "Mass of Body 1", 1, 50, 10)
    mass2_slider, mass2_value = create_slider(root, "Mass of Body 2", 1, 50, 20)
    mass3_slider, mass3_value = create_slider(root, "Mass of Body 3", 1, 50, 30)

    # Time Step slider
    delta_t_factor = 10000  # Precision factor
    delta_t_slider, delta_t_value = create_slider(root, "Time Step (delta_t)", 1 / delta_t_factor, 0.01, 0.001, is_float=True)

    # Steps slider
    steps_slider, steps_value = create_slider(root, "Number of Steps", 1000, 50000, 20000)

    # Position and Velocity Inputs (as before)
    ttk.Label(root, text="Initial Positions (Body 1)", background="#1e1e2e", foreground="white").pack()
    p1x, p1y, p1z = tk.Entry(root), tk.Entry(root), tk.Entry(root)
    p1x.insert(0, "-10"), p1y.insert(0, "10"), p1z.insert(0, "-11")
    p1x.pack(), p1y.pack(), p1z.pack()

    ttk.Label(root, text="Initial Velocities (Body 1)", background="#1e1e2e", foreground="white").pack()
    v1x, v1y, v1z = tk.Entry(root), tk.Entry(root), tk.Entry(root)
    v1x.insert(0, "-3"), v1y.insert(0, "0"), v1z.insert(0, "0")
    v1x.pack(), v1y.pack(), v1z.pack()

    ttk.Label(root, text="Initial Positions (Body 2)", background="#1e1e2e", foreground="white").pack()
    p2x, p2y, p2z = tk.Entry(root), tk.Entry(root), tk.Entry(root)
    p2x.insert(0, "0"), p2y.insert(0, "0"), p2z.insert(0, "0")
    p2x.pack(), p2y.pack(), p2z.pack()

    ttk.Label(root, text="Initial Velocities (Body 2)", background="#1e1e2e", foreground="white").pack()
    v2x, v2y, v2z = tk.Entry(root), tk.Entry(root), tk.Entry(root)
    v2x.insert(0, "0"), v2y.insert(0, "0"), v2z.insert(0, "0")
    v2x.pack(), v2y.pack(), v2z.pack()

    ttk.Label(root, text="Initial Positions (Body 3)", background="#1e1e2e", foreground="white").pack()
    p3x, p3y, p3z = tk.Entry(root), tk.Entry(root), tk.Entry(root)
    p3x.insert(0, "10"), p3y.insert(0, "10"), p3z.insert(0, "12")
    p3x.pack(), p3y.pack(), p3z.pack()

    ttk.Label(root, text="Initial Velocities (Body 3)", background="#1e1e2e", foreground="white").pack()
    v3x, v3y, v3z = tk.Entry(root), tk.Entry(root), tk.Entry(root)
    v3x.insert(0, "3"), v3y.insert(0, "0"), v3z.insert(0, "0")
    v3x.pack(), v3y.pack(), v3z.pack()

    # Confirm Button
    confirm_button = ttk.Button(root, text="Confirm", command=start_simulation)
    confirm_button.pack(pady=20)

    root.mainloop()

# Run the GUI
create_gui()
