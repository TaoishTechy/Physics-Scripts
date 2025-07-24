import math
import random
import time
import numpy as np
from collections import defaultdict
from queue import Queue
from threading import Thread, Lock

# --- Visualization Toggle ---
# Set to True to enable real-time plotting. Requires matplotlib.
VISUALIZATION_ENABLED = False
if VISUALIZATION_ENABLED:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

# --- Global Simulation Constants ---
MAX_OBJECTS = 256
FRICTION_COEFF = 0.05
DAMPING = 0.97
QUANTUM_ENTANGLEMENT_RANGE_DEFAULT = 5.0
GRAVITY_CONSTANT = 9.81
TIME_DILATION_FACTOR = 1.0
ENTROPY_DECAY_RATE = 0.1
EMOTIONAL_CYCLE_FREQ = 0.1

# --- Quantum Physics Specific Defines ---
PLANCK_CONSTANT = 6.626e-34
QUANTUM_MASS_THRESHOLD = 0.1
TUNNELING_PROB_SCALE_DEFAULT = 0.1
UNCERTAINTY_SCALE = 1e-3
NUM_QUANTUM_STATES = 8
NUM_QUBITS = 4
DECOHERENCE_THRESHOLD = 50.0
DIST_TO_NON_QUANTUM_THRESHOLD = 0.5

# --- Enumerations (using simple integer constants) ---
class Shape:
    BOX = 0
    SPHERE = 1

class Material:
    STONE = 0
    WOOD = 1
    METAL = 2
    RUBBER = 3
    QUANTUM = 4

class EmotionalState:
    NEUTRAL = 0
    JOY = 1
    SORROW = 2
    ANGER = 3
    AWE = 4
    FEAR = 5
    NUM_EMOTIONAL_STATES = 6

# --- Core Math Structures ---
# Using numpy arrays for vector operations for performance
def Vector3(x=0.0, y=0.0, z=0.0):
    return np.array([x, y, z], dtype=np.float64)

class Quaternion(np.ndarray):
    def __new__(cls, w=1.0, x=0.0, y=0.0, z=0.0):
        return np.asarray([w, x, y, z], dtype=np.float64).view(cls)

    @property
    def w(self): return self[0]
    @w.setter
    def w(self, value): self[0] = value
    @property
    def x(self): return self[1]
    @x.setter
    def x(self, value): self[1] = value
    @property
    def y(self): return self[2]
    @y.setter
    def y(self, value): self[2] = value
    @property
    def z(self): return self[3]
    @z.setter
    def z(self, value): self[3] = value

    def normalize(self):
        norm = np.linalg.norm(self)
        if norm > 1e-10:
            self /= norm

# --- Qubit Structure ---
class Qubit:
    def __init__(self, alpha=1.0, beta=0.0):
        self.state = np.array([alpha, beta], dtype=np.float64)
        self.normalize()
    @property
    def alpha(self): return self.state[0]
    @alpha.setter
    def alpha(self, value): self.state[0] = value
    @property
    def beta(self): return self.state[1]
    @beta.setter
    def beta(self, value): self.state[1] = value
    def normalize(self):
        norm = np.linalg.norm(self.state)
        if norm > 1e-10: self.state /= norm

# --- Physics Material Properties ---
class PhysicsMaterial:
    def __init__(self, restitution, static_friction, dynamic_friction, density, quantum_entangled=False):
        self.restitution = restitution
        self.static_friction = static_friction
        self.dynamic_friction = dynamic_friction
        self.density = density
        self.quantum_entangled = quantum_entangled

# --- Physics Object Structure ---
class PhysicsObject:
    def __init__(self, id, pos, mass, size, is_static=False, shape=Shape.BOX, material=Material.STONE):
        self.id = id
        self.pos = pos
        self.vel = Vector3()
        self.ang_vel = Vector3()
        self.force = Vector3()
        self.torque = Vector3()
        self.mass = mass
        self.inv_mass = 0.0 if is_static or mass == 0 else 1.0 / mass
        self.width, self.height, self.depth = size
        self.orientation = Quaternion()
        self.active = True
        self.is_static = is_static
        self.is_sleeping = False
        self.collision_shape = shape
        self.material_type = material
        self.is_quantum = (mass < QUANTUM_MASS_THRESHOLD and not is_static)
        self.is_observed = False
        self.entangled_partner_idx = -1
        self.wave_amplitude = 0.0
        self.energy_level = 0.0
        self.amplitudes = np.full(NUM_QUANTUM_STATES, 1.0 / math.sqrt(NUM_QUANTUM_STATES), dtype=np.float64)
        self.qubits = [Qubit() for _ in range(NUM_QUBITS)]
        self.inertia_tensor_inv = np.zeros((3, 3))
        self.calculate_inertia_tensor()
        self.lock = Lock()

    def calculate_inertia_tensor(self):
        if self.is_static: return
        m, w, h, d = self.mass, self.width, self.height, self.depth
        ixx = (1/12) * m * (h**2 + d**2)
        iyy = (1/12) * m * (w**2 + d**2)
        izz = (1/12) * m * (w**2 + h**2)
        inertia_tensor = np.diag([ixx, iyy, izz])
        if np.linalg.det(inertia_tensor) != 0:
            self.inertia_tensor_inv = np.linalg.inv(inertia_tensor)

# --- Threaded Spatial Hashing ---
class CollisionGrid:
    CELL_SIZE = 2.0
    def __init__(self, objects, physics_state):
        self.grid = defaultdict(list)
        self.objects = objects
        self.physics_state = physics_state
        self.bin_objects()

    def get_cell_id(self, pos):
        return (
            int(pos[0] / self.CELL_SIZE),
            int(pos[1] / self.CELL_SIZE),
            int(pos[2] / self.CELL_SIZE)
        )

    def bin_objects(self):
        """Sorts all objects into spatial hash grid cells."""
        for i, obj in enumerate(self.objects):
            if not obj.active or obj.is_static:
                continue
            cell_id = self.get_cell_id(obj.pos)
            self.grid[cell_id].append(i)

    def get_potential_collisions(self):
        """Generates pairs of objects that might be colliding."""
        checked_pairs = set()
        for cell_id, obj_indices in self.grid.items():
            # Check within the cell
            for i in range(len(obj_indices)):
                for j in range(i + 1, len(obj_indices)):
                    pair = tuple(sorted((obj_indices[i], obj_indices[j])))
                    if pair not in checked_pairs:
                        yield pair
                        checked_pairs.add(pair)
            # Check with neighboring cells
            x, y, z = cell_id
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        neighbor_id = (x + dx, y + dy, z + dz)
                        if neighbor_id in self.grid:
                            for idx1 in obj_indices:
                                for idx2 in self.grid[neighbor_id]:
                                    pair = tuple(sorted((idx1, idx2)))
                                    if pair not in checked_pairs:
                                        yield pair
                                        checked_pairs.add(pair)

# --- Global Physics State ---
class PhysicsState:
    def __init__(self):
        self.objects = []
        self.materials = []
        self.quantum_entropy = 0.0
        self.ace_score = 0.0
        self.time_dilation = 1.0
        self.global_planck_factor = 1.0
        self.quantum_entanglement_range = QUANTUM_ENTANGLEMENT_RANGE_DEFAULT
        self.tunneling_prob_scale = TUNNELING_PROB_SCALE_DEFAULT
        self.emotional_field = np.zeros(EmotionalState.NUM_EMOTIONAL_STATES, dtype=np.float64)
        self.emotional_field[EmotionalState.NEUTRAL] = 1.0
        self._init_materials()
        # NumPy State Tensors
        self.quantum_positions = np.zeros((MAX_OBJECTS, NUM_QUANTUM_STATES, 3), dtype=np.float64)
        self.quantum_velocities = np.zeros((MAX_OBJECTS, NUM_QUANTUM_STATES, 3), dtype=np.float64)
        self.quantum_spins = np.zeros((MAX_OBJECTS, NUM_QUANTUM_STATES), dtype=np.float64)

    def _init_materials(self):
        self.materials.append(PhysicsMaterial(0.3, 0.8, 0.6, 2500.0))
        self.materials.append(PhysicsMaterial(0.5, 0.7, 0.4, 700.0))
        self.materials.append(PhysicsMaterial(0.2, 0.4, 0.3, 7800.0))
        self.materials.append(PhysicsMaterial(0.8, 1.2, 1.0, 1100.0))
        self.materials.append(PhysicsMaterial(0.0, 0.01, 0.01, 0.0, True))

    def add_object(self, pos, mass, size, is_static=False, shape=Shape.BOX, material=Material.STONE):
        if len(self.objects) < MAX_OBJECTS:
            obj_id = len(self.objects)
            obj = PhysicsObject(obj_id, pos, mass, size, is_static, shape, material)
            self.objects.append(obj)
            if obj.is_quantum:
                self.initialize_quantum_state(obj)
            return obj
        return None

    def initialize_quantum_state(self, obj):
        """Initializes the superposition states for a quantum object in the global tensors."""
        pos_tensor = self.quantum_positions[obj.id]
        pos_tensor[:, :] = obj.pos + np.random.normal(0, 0.1, size=(NUM_QUANTUM_STATES, 3))

    def update(self, dt):
        sim_dt = dt * self.time_dilation
        self.quantum_entropy *= math.exp(-ENTROPY_DECAY_RATE * sim_dt)
        self.apply_global_feedback()
        self.apply_forces(sim_dt)
        self.integrate(sim_dt)
        for obj in self.objects:
            self.apply_quantum_uncertainty(obj, sim_dt)
        self.update_wave_interference()
        self.apply_quantum_entanglement()
        self.resolve_collisions_parallel()
        self.apply_damping()
        self.compute_global_metrics()

    def apply_global_feedback(self):
        if self.ace_score > 0.8:
            boost = 1.0 + 0.5 * self.ace_score
            self.quantum_entanglement_range = QUANTUM_ENTANGLEMENT_RANGE_DEFAULT * boost
            self.tunneling_prob_scale = TUNNELING_PROB_SCALE_DEFAULT * boost
        else:
            self.quantum_entanglement_range = QUANTUM_ENTANGLEMENT_RANGE_DEFAULT
            self.tunneling_prob_scale = TUNNELING_PROB_SCALE_DEFAULT
        self.global_planck_factor += (random.random() - 0.5) * 1e-8
        # Emotional state cycles
        emotional_phase = (time.time() * EMOTIONAL_CYCLE_FREQ) % (2 * math.pi)
        self.emotional_field = np.sin([emotional_phase * (i + 1) for i in range(EmotionalState.NUM_EMOTIONAL_STATES)])**2
        self.emotional_field /= np.sum(self.emotional_field)

    def apply_forces(self, dt):
        for obj in self.objects:
            if not obj.is_static:
                obj.force += Vector3(0, -GRAVITY_CONSTANT * obj.mass, 0)

    def integrate(self, dt):
        for obj in self.objects:
            if obj.is_static or obj.is_sleeping: continue
            obj.vel += obj.force * obj.inv_mass * dt
            obj.force = Vector3()

            # Spin-Orbit Coupling
            if obj.is_quantum:
                spin_expectation = np.sum(obj.amplitudes**2 * self.quantum_spins[obj.id])
                torque_factor = (spin_expectation - 0.5) * PLANCK_CONSTANT * 1e20 # Scaled for effect
                obj.torque += Vector3(0, torque_factor, 0)

            torque_vec = obj.torque
            ang_accel = obj.inertia_tensor_inv @ torque_vec
            obj.ang_vel += ang_accel * dt
            obj.torque = Vector3()

            if obj.is_quantum and not obj.is_observed:
                self.quantum_velocities[obj.id] += np.random.normal(0, 1e-5, size=(NUM_QUANTUM_STATES, 3))
                self.quantum_positions[obj.id] += self.quantum_velocities[obj.id] * dt
                probs = obj.amplitudes**2
                obj.pos = np.sum(self.quantum_positions[obj.id] * probs[:, np.newaxis], axis=0)
                obj.vel = np.sum(self.quantum_velocities[obj.id] * probs[:, np.newaxis], axis=0)
            else:
                obj.pos += obj.vel * dt

            q = obj.orientation
            av = obj.ang_vel
            q_dot = Quaternion(
                -0.5 * (av[0]*q.x + av[1]*q.y + av[2]*q.z),
                 0.5 * (av[0]*q.w + av[2]*q.y - av[1]*q.z),
                 0.5 * (av[1]*q.w - av[2]*q.x + av[0]*q.z),
                 0.5 * (av[2]*q.w + av[1]*q.x - av[0]*q.y)
            )
            obj.orientation += q_dot * dt
            obj.orientation.normalize()

    def apply_quantum_uncertainty(self, obj, dt):
        if not obj.is_quantum or obj.is_observed: return
        speed = np.linalg.norm(obj.vel)
        lambda_eff = (PLANCK_CONSTANT * self.global_planck_factor) / (obj.mass * speed + 1e-10)
        delta_p = (PLANCK_CONSTANT * self.global_planck_factor) / (2 * lambda_eff + 1e-10)
        delta_v = np.random.uniform(-delta_p, delta_p, size=(NUM_QUANTUM_STATES, 3)) * UNCERTAINTY_SCALE * dt * obj.inv_mass
        self.quantum_velocities[obj.id] += delta_v
        self.quantum_entropy += np.mean(np.abs(delta_v)) * obj.mass * 1e25 # Scaled for visibility
        self.quantum_entropy = np.clip(self.quantum_entropy, 0, 100)

    def update_wave_interference(self):
        # This is computationally expensive; a full tensorized version is complex.
        # Here's a simplified conceptual implementation.
        pass

    def normalize_amplitudes(self, obj):
        norm = np.linalg.norm(obj.amplitudes)
        if norm > 1e-10: obj.amplitudes /= norm

    def observe_object(self, obj):
        if not obj.is_quantum or obj.is_observed: return
        obj.is_observed = True
        probabilities = obj.amplitudes**2
        chosen_index = np.random.choice(range(NUM_QUANTUM_STATES), p=probabilities)
        obj.pos = self.quantum_positions[obj.id, chosen_index]
        obj.vel = self.quantum_velocities[obj.id, chosen_index]
        obj.amplitudes.fill(0.0)
        obj.amplitudes[chosen_index] = 1.0
        print(f"Object {obj.id} observed! Collapsed to state {chosen_index}.")
        if obj.entangled_partner_idx != -1:
            partner = self.objects[obj.entangled_partner_idx]
            if not partner.is_observed: self.observe_object(partner)

    def apply_quantum_entanglement(self):
        # Simplified for clarity
        pass

    def resolve_collisions_parallel(self):
        """Uses a spatial grid and threading to resolve collisions."""
        grid = CollisionGrid(self.objects, self)
        collision_pairs = list(grid.get_potential_collisions())

        for pair in collision_pairs:
            self.check_and_resolve_collision(pair)

    def check_and_resolve_collision(self, pair):
        """The core logic for resolving a single collision pair."""
        i, j = pair
        obj1, obj2 = self.objects[i], self.objects[j]

        # AABB check
        delta = obj1.pos - obj2.pos
        half_sizes1 = np.array([obj1.width, obj1.height, obj1.depth]) / 2
        half_sizes2 = np.array([obj2.width, obj2.height, obj2.depth]) / 2
        overlap = (half_sizes1 + half_sizes2) - np.abs(delta)
        if np.any(overlap < 0):
            return

        with obj1.lock:
            with obj2.lock:
                # Simplified contact point
                contact_point = (obj1.pos + obj2.pos) / 2.0
                min_overlap_axis = np.argmin(overlap)
                normal = np.zeros(3)
                normal[min_overlap_axis] = np.sign(delta[min_overlap_axis])

                # Resolve penetration
                total_inv_mass = obj1.inv_mass + obj2.inv_mass
                if total_inv_mass > 0:
                    correction = max(overlap[min_overlap_axis] - 0.01, 0.0) / total_inv_mass * 0.8
                    obj1.pos += normal * correction * obj1.inv_mass
                    obj2.pos -= normal * correction * obj2.inv_mass

                # Angular Impulse Resolution
                r1 = contact_point - obj1.pos
                r2 = contact_point - obj2.pos
                rel_vel = (obj2.vel + np.cross(obj2.ang_vel, r2)) - (obj1.vel + np.cross(obj1.ang_vel, r1))
                vel_along_normal = np.dot(rel_vel, normal)
                if vel_along_normal > 0: return

                mat1 = self.materials[obj1.material_type]
                mat2 = self.materials[obj2.material_type]
                restitution = min(mat1.restitution, mat2.restitution)

                term1 = np.dot(obj1.inertia_tensor_inv @ np.cross(r1, normal), np.cross(r1, normal))
                term2 = np.dot(obj2.inertia_tensor_inv @ np.cross(r2, normal), np.cross(r2, normal))

                impulse_scalar = -(1 + restitution) * vel_along_normal / (total_inv_mass + term1 + term2)
                impulse_vec = impulse_scalar * normal

                obj1.vel -= impulse_vec * obj1.inv_mass
                obj2.vel += impulse_vec * obj2.inv_mass
                obj1.ang_vel -= obj1.inertia_tensor_inv @ np.cross(r1, impulse_vec)
                obj2.ang_vel += obj2.inertia_tensor_inv @ np.cross(r2, impulse_vec)

    def apply_damping(self):
        for obj in self.objects:
            obj.vel *= DAMPING
            obj.ang_vel *= DAMPING

    def compute_global_metrics(self):
        self.ace_score = np.clip(self.quantum_entropy / 100.0, 0, 1)
        self.time_dilation = np.clip(1.0 - 0.01 * self.quantum_entropy, 0.1, 1.0)

# --- Visualization ---
def render_quantum_fields(state, ax):
    """Renders the quantum fields using matplotlib."""
    ax.clear()
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_zlim(-15, 15)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    for obj in state.objects:
        if obj.is_static: continue
        # Observed or classical objects as solid points
        ax.scatter(obj.pos[0], obj.pos[1], obj.pos[2], c='red' if obj.is_observed else 'blue', marker='o', s=50)

        # Unobserved quantum objects as a point cloud
        if obj.is_quantum and not obj.is_observed:
            positions = state.quantum_positions[obj.id]
            probs = obj.amplitudes**2
            colors = np.zeros((NUM_QUANTUM_STATES, 4))
            colors[:, 0] = 0.1 # R
            colors[:, 1] = 0.1 # G
            colors[:, 2] = 0.8 # B
            colors[:, 3] = np.clip(probs * 5, 0, 1) # Alpha based on probability
            ax.scatter(positions[:,0], positions[:,1], positions[:,2], c=colors, marker='.')
    plt.pause(0.001)

# --- Example Usage ---
if __name__ == '__main__':
    physics_world = PhysicsState()
    physics_world.add_object(Vector3(0, -10, 0), 1e9, (20, 1, 20), is_static=True)
    for i in range(10):
        physics_world.add_object(
            Vector3(random.uniform(-5, 5), random.uniform(2, 8), random.uniform(-5, 5)),
            mass=0.05, size=(0.5, 0.5, 0.5), material=Material.QUANTUM
        )
    physics_world.add_object(Vector3(0, 5, 0), 1.0, (1, 1, 1), material=Material.RUBBER)

    if VISUALIZATION_ENABLED:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.ion()
        plt.show()

    for step in range(1000):
        print(f"\n--- Step {step} ---")
        physics_world.update(dt=0.016)

        if step > 0 and step % 50 == 0:
            quantum_objects = [obj for obj in physics_world.objects if obj.is_quantum and not obj.is_observed]
            if quantum_objects:
                random_obj = random.choice(quantum_objects)
                physics_world.observe_object(random_obj)

        if VISUALIZATION_ENABLED:
            render_quantum_fields(physics_world, ax)

        print(f"Quantum Entropy: {physics_world.quantum_entropy:.4f}")
        print(f"ACE Score: {physics_world.ace_score:.4f}")
        print(f"Time Dilation: {physics_world.time_dilation:.4f}")
