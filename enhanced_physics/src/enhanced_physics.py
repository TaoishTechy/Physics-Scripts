import math
import random
import numpy as np

# --- Global Simulation Constants ---
MAX_OBJECTS = 256
FRICTION_COEFF = 0.05
DAMPING = 0.97
QUANTUM_ENTANGLEMENT_RANGE_DEFAULT = 5.0
GRAVITY_CONSTANT = 9.81
TIME_DILATION_FACTOR = 1.0

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
class Vector3:
    """A simple 3D vector class."""
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Vector3({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"

class Quaternion:
    """A simple quaternion class for orientation."""
    def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

# --- Qubit Structure for Quantum Computing Elements ---
class Qubit:
    """Represents a quantum bit with real amplitudes for simplicity."""
    def __init__(self, alpha=1.0, beta=0.0):
        self.alpha = alpha  # Amplitude for |0>
        self.beta = beta    # Amplitude for |1>
        self.normalize()

    def normalize(self):
        """Ensures the state is normalized (alpha^2 + beta^2 = 1)."""
        norm_sq = self.alpha**2 + self.beta**2
        if norm_sq > 1e-10:
            norm = math.sqrt(norm_sq)
            self.alpha /= norm
            self.beta /= norm
        else:
            self.alpha = 1.0
            self.beta = 0.0

# --- Physics Material Properties ---
class PhysicsMaterial:
    """Stores properties for different material types."""
    def __init__(self, restitution, static_friction, dynamic_friction, density, quantum_entangled=False):
        self.restitution = restitution
        self.static_friction = static_friction
        self.dynamic_friction = dynamic_friction
        self.density = density
        self.quantum_entangled = quantum_entangled

# --- Physics Object Structure ---
class PhysicsObject:
    """Represents a single object in the physics simulation."""
    def __init__(self, pos, mass, size, is_static=False, shape=Shape.BOX, material=Material.STONE):
        self.pos = pos
        self.prev_pos = Vector3(pos.x, pos.y, pos.z)
        self.vel = Vector3()
        self.ang_vel = Vector3()
        self.force = Vector3()
        self.mass = mass
        self.inv_mass = 0.0 if is_static or mass == 0 else 1.0 / mass
        self.width, self.height, self.depth = size
        self.orientation = Quaternion()

        self.active = True
        self.is_static = is_static
        self.is_sleeping = False
        self.collision_shape = shape
        self.material_type = material

        # Quantum fields
        self.is_quantum = (mass < QUANTUM_MASS_THRESHOLD and not is_static)
        self.is_observed = False
        self.entangled_partner_idx = -1
        self.wave_amplitude = 0.0
        self.energy_level = 0.0

        # Wavefunction superposition states
        self.possible_positions = [Vector3(pos.x, pos.y, pos.z) for _ in range(NUM_QUANTUM_STATES)]
        self.possible_velocities = [Vector3() for _ in range(NUM_QUANTUM_STATES)]
        self.possible_spins = [0.0 for _ in range(NUM_QUANTUM_STATES)]
        self.amplitudes = [1.0 / math.sqrt(NUM_QUANTUM_STATES) for _ in range(NUM_QUANTUM_STATES)]

        # Quantum computing elements
        self.qubits = [Qubit() for _ in range(NUM_QUBITS)]

        if self.is_quantum:
            self.initialize_quantum_state()

    def initialize_quantum_state(self):
        """Initializes the superposition states for a quantum object."""
        for i in range(NUM_QUANTUM_STATES):
            self.possible_positions[i] = Vector3(
                self.pos.x + random.uniform(-0.1, 0.1),
                self.pos.y + random.uniform(-0.1, 0.1),
                self.pos.z + random.uniform(-0.1, 0.1)
            )
            self.amplitudes[i] = 1.0 / math.sqrt(NUM_QUANTUM_STATES)

# --- Global Physics State ---
class PhysicsState:
    """Manages the entire physics simulation."""
    def __init__(self):
        self.objects = []
        self.materials = []
        self.quantum_entropy = 0.0
        self.ace_score = 0.0
        self.time_dilation = 1.0
        self.global_planck_factor = 1.0
        self.quantum_entanglement_range = QUANTUM_ENTANGLEMENT_RANGE_DEFAULT
        self.tunneling_prob_scale = TUNNELING_PROB_SCALE_DEFAULT
        self.emotional_field = [0.0] * EmotionalState.NUM_EMOTIONAL_STATES
        self.emotional_field[EmotionalState.NEUTRAL] = 1.0
        self._init_materials()

    def _init_materials(self):
        """Initializes the library of physics materials."""
        self.materials.append(PhysicsMaterial(0.3, 0.8, 0.6, 2500.0)) # Stone
        self.materials.append(PhysicsMaterial(0.5, 0.7, 0.4, 700.0))  # Wood
        self.materials.append(PhysicsMaterial(0.2, 0.4, 0.3, 7800.0)) # Metal
        self.materials.append(PhysicsMaterial(0.8, 1.2, 1.0, 1100.0)) # Rubber
        self.materials.append(PhysicsMaterial(0.0, 0.01, 0.01, 0.0, True)) # Quantum

    def add_object(self, pos, mass, size, is_static=False, shape=Shape.BOX, material=Material.STONE):
        """Adds a new object to the simulation."""
        if len(self.objects) < MAX_OBJECTS:
            obj = PhysicsObject(pos, mass, size, is_static, shape, material)

            # Eq 1: Initial de-Broglie wavelength
            speed_sq = obj.vel.x**2 + obj.vel.y**2 + obj.vel.z**2
            obj.wave_amplitude = (PLANCK_CONSTANT * self.global_planck_factor) / (obj.mass * math.sqrt(speed_sq) + 1e-10) if obj.is_quantum else 0.0

            self.objects.append(obj)
            return obj
        return None

    def update(self, dt):
        """The main physics update loop."""
        sim_dt = dt * self.time_dilation

        self.apply_global_feedback()
        self.apply_forces(sim_dt)
        self.integrate(sim_dt)
        self.update_wave_interference()
        self.apply_quantum_entanglement()
        self.resolve_collisions()
        self.apply_damping()
        self.compute_global_metrics()

    def apply_global_feedback(self):
        """Applies feedback from ACE score and emotional fields."""
        # Eq 7: ACE feedback amplification
        if self.ace_score > 0.8:
            boost = 1.0 + 0.5 * self.ace_score
            self.quantum_entanglement_range = QUANTUM_ENTANGLEMENT_RANGE_DEFAULT * boost
            self.tunneling_prob_scale = TUNNELING_PROB_SCALE_DEFAULT * boost
        else:
            self.quantum_entanglement_range = QUANTUM_ENTANGLEMENT_RANGE_DEFAULT
            self.tunneling_prob_scale = TUNNELING_PROB_SCALE_DEFAULT

        # Drift cosmic constants (simplified)
        self.global_planck_factor += (random.random() - 0.5) * 1e-8

    def apply_forces(self, dt):
        """Applies forces like gravity to all objects."""
        for obj in self.objects:
            if not obj.is_static:
                obj.force.y -= GRAVITY_CONSTANT * obj.mass

    def integrate(self, dt):
        """Integrates motion over a time step."""
        for obj in self.objects:
            if obj.is_static or obj.is_sleeping:
                continue

            # Update velocity from force
            obj.vel.x += obj.force.x * obj.inv_mass * dt
            obj.vel.y += obj.force.y * obj.inv_mass * dt
            obj.vel.z += obj.force.z * obj.inv_mass * dt
            obj.force = Vector3() # Clear forces

            if obj.is_quantum and not obj.is_observed:
                # Evolve all possible states
                for i in range(NUM_QUANTUM_STATES):
                    obj.possible_positions[i].x += obj.possible_velocities[i].x * dt
                    obj.possible_positions[i].y += obj.possible_velocities[i].y * dt
                    obj.possible_positions[i].z += obj.possible_velocities[i].z * dt

                # Update expectation values for position and velocity
                # Eq 4 (partially): Calculating expectation values for integration
                exp_pos = Vector3()
                exp_vel = Vector3()
                for i in range(NUM_QUANTUM_STATES):
                    amp = obj.amplitudes[i]
                    exp_pos.x += obj.possible_positions[i].x * amp**2
                    exp_pos.y += obj.possible_positions[i].y * amp**2
                    exp_pos.z += obj.possible_positions[i].z * amp**2
                    exp_vel.x += obj.possible_velocities[i].x * amp**2
                    exp_vel.y += obj.possible_velocities[i].y * amp**2
                    exp_vel.z += obj.possible_velocities[i].z * amp**2
                obj.pos = exp_pos
                obj.vel = exp_vel
            else:
                # Classical integration
                obj.pos.x += obj.vel.x * dt
                obj.pos.y += obj.vel.y * dt
                obj.pos.z += obj.vel.z * dt

    def apply_quantum_uncertainty(self, obj, dt):
        """Applies Heisenberg uncertainty kicks."""
        if not obj.is_quantum or obj.is_observed:
            return

        # Eq 2: Effective wavelength
        speed = math.sqrt(obj.vel.x**2 + obj.vel.y**2 + obj.vel.z**2)
        lambda_eff = (PLANCK_CONSTANT * self.global_planck_factor) / (obj.mass * speed + 1e-10)

        # Eq 3: Momentum uncertainty kick
        delta_p = (PLANCK_CONSTANT * self.global_planck_factor) / (2 * lambda_eff + 1e-10)

        for i in range(NUM_QUANTUM_STATES):
            delta_v_x = random.uniform(-delta_p, delta_p) * UNCERTAINTY_SCALE * dt * obj.inv_mass
            delta_v_y = random.uniform(-delta_p, delta_p) * UNCERTAINTY_SCALE * dt * obj.inv_mass
            delta_v_z = random.uniform(-delta_p, delta_p) * UNCERTAINTY_SCALE * dt * obj.inv_mass

            obj.possible_velocities[i].x += delta_v_x
            obj.possible_velocities[i].y += delta_v_y
            obj.possible_velocities[i].z += delta_v_z

        self.quantum_entropy += abs(delta_p) * UNCERTAINTY_SCALE
        self.quantum_entropy = np.clip(self.quantum_entropy, 0, 100)

    def update_wave_interference(self):
        """Models interference between nearby quantum objects."""
        for i, obj1 in enumerate(self.objects):
            if not obj1.is_quantum or obj1.is_observed: continue
            for j in range(i + 1, len(self.objects)):
                obj2 = self.objects[j]
                if not obj2.is_quantum or obj2.is_observed: continue

                dist_sq = (obj1.pos.x - obj2.pos.x)**2 + (obj1.pos.y - obj2.pos.y)**2 + (obj1.pos.z - obj2.pos.z)**2
                if dist_sq < 9.0: # Interference range
                    for k in range(NUM_QUANTUM_STATES):
                        for m in range(NUM_QUANTUM_STATES):
                            local_dist_sq = (obj1.possible_positions[k].x - obj2.possible_positions[m].x)**2 + \
                                            (obj1.possible_positions[k].y - obj2.possible_positions[m].y)**2 + \
                                            (obj1.possible_positions[k].z - obj2.possible_positions[m].z)**2
                            local_dist = math.sqrt(local_dist_sq)

                            phase_diff1 = 2 * math.pi * local_dist / (obj1.wave_amplitude + 1e-10)
                            phase_diff2 = 2 * math.pi * local_dist / (obj2.wave_amplitude + 1e-10)

                            obj1.amplitudes[k] += obj2.amplitudes[m] * math.cos(phase_diff1)
                            obj2.amplitudes[m] += obj1.amplitudes[k] * math.cos(phase_diff2)

                    # Eq 5: Wave-function normalization
                    self.normalize_amplitudes(obj1)
                    self.normalize_amplitudes(obj2)

    def normalize_amplitudes(self, obj):
        """Normalizes the probability amplitudes for an object."""
        sum_sq = sum(a**2 for a in obj.amplitudes)
        if sum_sq > 1e-10:
            norm = math.sqrt(sum_sq)
            obj.amplitudes = [a / norm for a in obj.amplitudes]

    def observe_object(self, obj):
        """Collapses the wavefunction of an object upon observation."""
        if not obj.is_quantum or obj.is_observed:
            return

        obj.is_observed = True

        # Eq 4: Expectation-value collapse (roulette wheel selection)
        probabilities = [amp**2 for amp in obj.amplitudes]
        chosen_index = random.choices(range(NUM_QUANTUM_STATES), weights=probabilities, k=1)[0]

        obj.pos = obj.possible_positions[chosen_index]
        obj.vel = obj.possible_velocities[chosen_index]

        # Reset wavefunction to a single state
        obj.amplitudes = [0.0] * NUM_QUANTUM_STATES
        obj.amplitudes[chosen_index] = 1.0

        print(f"Object {self.objects.index(obj)} observed! Collapsed to state {chosen_index}.")

        # Cascade observation to entangled partner
        if obj.entangled_partner_idx != -1:
            partner = self.objects[obj.entangled_partner_idx]
            if not partner.is_observed:
                self.observe_object(partner)

    def apply_quantum_entanglement(self):
        """Handles entanglement and decoherence."""
        for i, obj in enumerate(self.objects):
            if not obj.is_quantum: continue

            # Environmental Decoherence
            dist_to_classical = min([math.sqrt((obj.pos.x - other.pos.x)**2 + (obj.pos.y - other.pos.y)**2 + (obj.pos.z - other.pos.z)**2)
                                     for other in self.objects if not other.is_quantum] or [float('inf')])
            if not obj.is_observed and (self.quantum_entropy > DECOHERENCE_THRESHOLD or dist_to_classical < DIST_TO_NON_QUANTUM_THRESHOLD):
                self.observe_object(obj)

            # Entanglement Logic
            if obj.entangled_partner_idx == -1 and obj.material_type == Material.QUANTUM:
                for j in range(i + 1, len(self.objects)):
                    other = self.objects[j]
                    if other.material_type == Material.QUANTUM and other.entangled_partner_idx == -1:
                        dist_sq = (obj.pos.x - other.pos.x)**2 + (obj.pos.y - other.pos.y)**2 + (obj.pos.z - other.pos.z)**2
                        if dist_sq < self.quantum_entanglement_range**2:
                            obj.entangled_partner_idx = j
                            other.entangled_partner_idx = i
                            # Simple anti-correlation for spins
                            for k in range(NUM_QUANTUM_STATES):
                                obj.possible_spins[k] = 1.0 - other.possible_spins[k]
                            print(f"Objects {i} and {j} are now entangled!")
                            break

    def resolve_collisions(self):
        """Detects and resolves collisions between objects."""
        for i, obj1 in enumerate(self.objects):
            for j in range(i + 1, len(self.objects)):
                obj2 = self.objects[j]
                if obj1.is_static and obj2.is_static:
                    continue

                # Simple AABB collision check
                if (abs(obj1.pos.x - obj2.pos.x) * 2 < (obj1.width + obj2.width) and
                    abs(obj1.pos.y - obj2.pos.y) * 2 < (obj1.height + obj2.height) and
                    abs(obj1.pos.z - obj2.pos.z) * 2 < (obj1.depth + obj2.depth)):

                    # Quantum Tunneling
                    if self.compute_tunneling_probability(obj1, obj2) > random.random():
                        print(f"Quantum tunnel between {i} and {j}!")
                        continue

                    # On collision, unobserved quantum objects are observed
                    if obj1.is_quantum and not obj1.is_observed: self.observe_object(obj1)
                    if obj2.is_quantum and not obj2.is_observed: self.observe_object(obj2)

                    self.resolve_penetration(obj1, obj2)
                    self.resolve_impulse(obj1, obj2)

    def compute_tunneling_probability(self, obj1, obj2):
        """Placeholder for a quantum tunneling calculation."""
        # A real implementation would be very complex. This is a stub.
        if obj1.is_quantum or obj2.is_quantum:
            return 0.01 * self.tunneling_prob_scale
        return 0.0

    def resolve_penetration(self, obj1, obj2):
        """Corrects object positions to resolve overlap."""
        # Simplified penetration resolution
        overlap_x = (obj1.width + obj2.width) / 2.0 - abs(obj1.pos.x - obj2.pos.x)
        overlap_y = (obj1.height + obj2.height) / 2.0 - abs(obj1.pos.y - obj2.pos.y)
        overlap_z = (obj1.depth + obj2.depth) / 2.0 - abs(obj1.pos.z - obj2.pos.z)

        min_overlap = min(overlap_x, overlap_y, overlap_z)

        total_inv_mass = obj1.inv_mass + obj2.inv_mass
        if total_inv_mass == 0: return

        correction_percent = 0.8 # How much to correct
        correction_slop = 0.01
        correction = max(min_overlap - correction_slop, 0.0) / total_inv_mass * correction_percent

        if min_overlap == overlap_x:
            normal = Vector3(np.sign(obj1.pos.x - obj2.pos.x), 0, 0)
        elif min_overlap == overlap_y:
            normal = Vector3(0, np.sign(obj1.pos.y - obj2.pos.y), 0)
        else:
            normal = Vector3(0, 0, np.sign(obj1.pos.z - obj2.pos.z))

        obj1.pos.x += normal.x * correction * obj1.inv_mass
        obj1.pos.y += normal.y * correction * obj1.inv_mass
        obj1.pos.z += normal.z * correction * obj1.inv_mass
        obj2.pos.x -= normal.x * correction * obj2.inv_mass
        obj2.pos.y -= normal.y * correction * obj2.inv_mass
        obj2.pos.z -= normal.z * correction * obj2.inv_mass

    def resolve_impulse(self, obj1, obj2):
        """Applies collision impulse to object velocities."""
        # Simplified impulse resolution along the y-axis for now
        rel_vel_y = obj2.vel.y - obj1.vel.y
        if rel_vel_y > 0: return # Objects are separating

        mat1 = self.materials[obj1.material_type]
        mat2 = self.materials[obj2.material_type]
        restitution = min(mat1.restitution, mat2.restitution)

        impulse = -(1 + restitution) * rel_vel_y / (obj1.inv_mass + obj2.inv_mass)

        obj1.vel.y -= impulse * obj1.inv_mass
        obj2.vel.y += impulse * obj2.inv_mass

    def apply_damping(self):
        """Applies velocity damping to all objects."""
        for obj in self.objects:
            obj.vel.x *= DAMPING
            obj.vel.y *= DAMPING
            obj.vel.z *= DAMPING

    def compute_global_metrics(self):
        """Computes global metrics like total energy and ACE score."""
        total_ke = 0
        for obj in self.objects:
            if obj.is_static: continue
            ke = 0.5 * obj.mass * (obj.vel.x**2 + obj.vel.y**2 + obj.vel.z**2)

            # Eq 6: 1-D particle-in-a-box energy quantization
            if obj.is_quantum and not obj.is_observed:
                L = max(obj.width, obj.height, obj.depth)
                if L > 1e-10:
                    base_energy = (PLANCK_CONSTANT**2 * self.global_planck_factor**2) / (8 * obj.mass * L**2)
                    n = round(math.sqrt(ke / base_energy + 0.5)) if base_energy > 0 else 1
                    n = max(1, n)
                    ke = n**2 * base_energy
                    obj.energy_level = n
            total_ke += ke

        # Simplified ACE score calculation
        self.ace_score = np.clip(self.quantum_entropy / 100.0, 0, 1)

        # Quantum-driven time dilation
        self.time_dilation = np.clip(1.0 - 0.01 * self.quantum_entropy, 0.1, 1.0)

    # --- Quantum Computing Layer ---
    def apply_hadamard(self, qubit):
        """Applies a Hadamard gate to a qubit."""
        alpha_new = (qubit.alpha + qubit.beta) / math.sqrt(2)
        beta_new = (qubit.alpha - qubit.beta) / math.sqrt(2)
        qubit.alpha, qubit.beta = alpha_new, beta_new
        qubit.normalize()

    def apply_cnot(self, control_qubit, target_qubit):
        """Applies a CNOT gate."""
        if abs(control_qubit.beta)**2 > 0.5: # If control is |1>
            target_qubit.alpha, target_qubit.beta = target_qubit.beta, target_qubit.alpha
            target_qubit.normalize()

# --- Example Usage ---
if __name__ == '__main__':
    # Initialize the simulation
    physics_world = PhysicsState()

    # Add a floor
    physics_world.add_object(Vector3(0, -10, 0), 1e9, (20, 1, 20), is_static=True)

    # Add some quantum objects
    for i in range(5):
        physics_world.add_object(
            Vector3(random.uniform(-5, 5), random.uniform(2, 5), random.uniform(-5, 5)),
            mass=0.05, # Below the quantum threshold
            size=(0.5, 0.5, 0.5),
            material=Material.QUANTUM
        )

    # Add a classical object
    physics_world.add_object(Vector3(0, 5, 0), 1.0, (1, 1, 1), material=Material.RUBBER)

    # Simulation loop
    for step in range(500):
        print(f"\n--- Step {step} ---")
        physics_world.update(dt=0.016)

        # Periodically observe a random quantum object
        if step % 50 == 0:
            quantum_objects = [obj for obj in physics_world.objects if obj.is_quantum]
            if quantum_objects:
                random_obj = random.choice(quantum_objects)
                physics_world.observe_object(random_obj)

        print(f"Quantum Entropy: {physics_world.quantum_entropy:.4f}")
        print(f"ACE Score: {physics_world.ace_score:.4f}")
        print(f"Time Dilation: {physics_world.time_dilation:.4f}")
        for i, obj in enumerate(physics_world.objects):
            if not obj.is_static:
                print(f"Object {i}: Pos={obj.pos}, Vel={obj.vel}, Quantum={obj.is_quantum}, Observed={obj.is_observed}")
