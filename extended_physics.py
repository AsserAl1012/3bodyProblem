import numpy as np

class NBodySimulation:
    def __init__(self, masses, positions, velocities, delta_t, steps, G=9.8):
        """
        Initialize the N-body simulation.

        :param masses: Array of masses [m1, m2, ..., mN]
        :param positions: Initial positions of bodies (N x 3 array)
        :param velocities: Initial velocities of bodies (N x 3 array)
        :param delta_t: Time step
        :param steps: Number of steps in the simulation
        :param G: Gravitational constant (default 9.8)
        """
        self.N = len(masses)
        self.masses = np.array(masses, dtype=float)
        self.positions = np.zeros((steps, self.N, 3), dtype=float)
        self.velocities = np.zeros((steps, self.N, 3), dtype=float)
        self.delta_t = delta_t
        self.steps = steps
        self.G = G

        # Set initial positions and velocities
        self.positions[0] = np.array(positions, dtype=float)
        self.velocities[0] = np.array(velocities, dtype=float)

    def accelerations(self, positions):
        """
        Calculate the accelerations of all bodies at a given time step using Newton's law of gravitation.

        :param positions: Current positions of all bodies (N x 3 array)
        :return: Accelerations of all bodies (N x 3 array)
        """
        accelerations = np.zeros((self.N, 3), dtype=float)
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    r_ij = positions[j] - positions[i]
                    distance = np.linalg.norm(r_ij)
                    if distance > 0:  # Avoid division by zero
                        accelerations[i] += self.G * self.masses[j] * r_ij / (distance**3)
        return accelerations

    def evolve_system(self, relativistic=False, collision_radius=None):
        """
        Simulate the system over time.

        :param relativistic: Whether to include relativistic corrections
        :param collision_radius: Threshold distance for collisions (default None)
        """
        for t in range(self.steps - 1):
            # Calculate accelerations
            acc = self.accelerations(self.positions[t])

            # Relativistic corrections (if enabled)
            if relativistic:
                c = 299792.458  # Speed of light in km/s
                for i in range(self.N):
                    v = np.linalg.norm(self.velocities[t][i])
                    acc[i] *= (1 - (v / c)**2)

            # Update velocities and positions
            self.velocities[t + 1] = self.velocities[t] + acc * self.delta_t
            self.positions[t + 1] = self.positions[t] + self.velocities[t] * self.delta_t

            # Collision modeling (if enabled)
            if collision_radius:
                self.handle_collisions(t + 1, collision_radius)

    def handle_collisions(self, t, collision_radius):
        """
        Detect and handle collisions between bodies.

        :param t: Current time step
        :param collision_radius: Threshold distance for collisions
        """
        for i in range(self.N):
            for j in range(i + 1, self.N):  # Avoid double-checking pairs
                distance = np.linalg.norm(self.positions[t][j] - self.positions[t][i])
                if distance < collision_radius:
                    # Merge the two bodies: Combine masses, conserve momentum
                    total_mass = self.masses[i] + self.masses[j]
                    combined_velocity = (
                        self.masses[i] * self.velocities[t][i] +
                        self.masses[j] * self.velocities[t][j]
                    ) / total_mass

                    # Update positions, velocities, and masses
                    self.positions[t][i] = (self.positions[t][i] + self.positions[t][j]) / 2
                    self.velocities[t][i] = combined_velocity
                    self.masses[i] = total_mass

                    # Remove the second body by marking it as inactive
                    self.masses[j] = 0
                    self.positions[t][j] = self.positions[t][i]
                    self.velocities[t][j] = self.velocities[t][i]

    def get_trajectories(self):
        """
        Retrieve the trajectories of all active bodies.

        :return: Trajectories of bodies (steps x N x 3 array)
        """
        active_bodies = self.masses > 0  # Filter out inactive bodies
        return self.positions[:, active_bodies, :]

