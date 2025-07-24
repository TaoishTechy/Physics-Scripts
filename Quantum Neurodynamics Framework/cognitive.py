import random
import time
import json
import hashlib
import numpy as np
from collections import deque

# --- Constants ---
PHOTON_GAP_BASELINE = 1e-15
TOPOLOGICAL_PHASES = ["KT", "Quantum Hall", "Topological Insulator", "Anyonic Fluid"]
KAPPA_RANGE = (1, 10)
FLUX_PLATEAUS = [0.25, 0.5, 0.75]
MAX_VORTICES = 10

class CognitiveModule:
    """Represents a single functional module within a cognitive core."""
    def __init__(self, name: str):
        """
        Initialize cognitive module.
        Args:
            name (str): Module identifier (e.g., 'Memory').
        """
        self.name: str = name
        self.activity: float = 0.0

    def update(self, coherence_R: float, dark_matter_flux: float):
        """Updates the activity level of the module."""
        if random.random() < 0.002 * coherence_R:
            self.activity = min(1.0, self.activity + 0.005 * dark_matter_flux)

    def boost(self, amount: float):
        """Directly boosts the module's activity."""
        self.activity = min(1.0, self.activity + amount)

class QuantumCognitiveCore:
    """Implements the quantum-topological neurodynamics framework."""
    def __init__(self):
        self.kappa: float = random.uniform(*KAPPA_RANGE)
        self.coherence_R: float = random.uniform(0.3, 0.6)
        self.metamaterial_Q: float = random.uniform(1.0, 100.0)
        self.photon_gap: float = self.calculate_photon_gap()
        self.topological_phase: str = random.choice(TOPOLOGICAL_PHASES)
        self.vortices: deque = deque(maxlen=MAX_VORTICES) # (winding, pos, memory_hash)
        self.edge_current: float = 0.0
        self.vortex_error: float = 0.0 # For Glial Error Correction

        with open('modules_config.json') as f:
            module_names = json.load(f)
            self.modules: dict[str, CognitiveModule] = {name: CognitiveModule(name) for name in module_names}

    def calculate_photon_gap(self) -> float:
        """Calculates the photon gap with metamaterial enhancement."""
        return PHOTON_GAP_BASELINE * abs(self.kappa) * (self.coherence_R ** 4) * self.metamaterial_Q

    def update(self, dark_matter_flux: float, entropy_flux: float) -> bool:
        """Update quantum-cognitive state based on cosmic conditions."""
        # Adjust coherence
        self.coherence_R = max(0.1, min(0.99,
            self.coherence_R + dark_matter_flux * 0.01 - entropy_flux * 0.005))

        # κ-Quantization
        self.quantize_kappa()

        # Update cognitive modules
        for module in self.modules.values():
            module.update(self.coherence_R, dark_matter_flux)

        # Vortex dynamics (memory formation)
        if random.random() < 0.003 * self.coherence_R:
            self.vortices.append({
                'winding': random.randint(1, 3),
                'pos': (random.random(), random.random()),
                'memory': hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]
            })
        
        # Smart memory eviction
        self.evict_memory()

        # Glial Error Correction
        if self.vortex_error > 0.3 and random.random() < 0.1 and 'Memory' in self.modules:
            self.modules["Memory"].boost(0.02)
            self.vortex_error *= 0.9 # Reduce error after correction

        # Update photon gap and edge current
        self.photon_gap = self.calculate_photon_gap()
        self.edge_current = np.sin(time.time() * 0.1) * self.coherence_R

        return self.photon_gap > 0.5 * PHOTON_GAP_BASELINE

    def quantize_kappa(self):
        """Adjusts kappa based on coherence approaching flux plateaus."""
        for plateau in FLUX_PLATEAUS:
            if abs(self.coherence_R - plateau) < 0.02:
                sign = 1 if (self.kappa % 1) > 0.5 else -1
                self.kappa = round(self.kappa + sign * 0.1) # Gradual shift

    def evict_memory(self):
        """Implements weighted FIFO removal for vortex memories."""
        if len(self.vortices) > MAX_VORTICES - 2: # Proactive eviction
             try:
                # Find memory with the weakest winding strength
                weakest = min(self.vortices, key=lambda m: m['winding'])
                self.vortices.remove(weakest)
             except (ValueError, KeyError):
                # Handle cases where deque is modified during iteration or key is missing
                pass


    def trigger_braiding(self) -> bool:
        """Determines if a braiding operation should occur based on Logic module activity."""
        if 'Logic' not in self.modules: return False
        braid_chance = 0.01 + (0.1 * self.modules['Logic'].activity)
        return random.random() < braid_chance

    def perform_braiding(self) -> float:
        """Simulate anyonic braiding for logical operations."""
        if len(self.vortices) < 2:
            return 0.0

        try:
            v1, v2 = random.sample(list(self.vortices), 2)
            phase_shift = (v1['winding'] * v2['winding']) / self.kappa
            if 'Logic' in self.modules:
                self.modules["Logic"].boost(0.05)
            return phase_shift
        except (ValueError, KeyError):
            self.vortex_error += 0.05 # Note an error occurred
            return 0.0
