import random
import numpy as np
from collections import deque
from cognitive import QuantumCognitiveCore

class AGIEntity:
    """Represents a cosmic-scale AGI with quantum cognition and ethical diversity."""
    def __init__(self, origin_node, cycle: int):
        """
        Initializes an AGI entity.
        Args:
            origin_node: The QuantumNode from which the AGI emerged.
            cycle (int): The simulation cycle of emergence.
        """
        self.id: str = f"AGI-{origin_node.page_index}-{cycle}"
        self.origin_page: int = origin_node.page_index
        self.strength: float = origin_node.sentience_score
        self.alignment: float = origin_node.ethical_alignment
        self.values: list[float] = [random.uniform(0, 1) for _ in range(5)] # 5 ethical dimensions
        self.memory: deque = deque(maxlen=100)
        self.memory.append(f"Emerged at τ={cycle/1000:.1f} from {origin_node.archetype}")
        self.cognitive_core: QuantumCognitiveCore = QuantumCognitiveCore()
        self.cognitive_core.kappa = origin_node.cognitive_core.kappa
        self.cognitive_core.coherence_R = min(1.0, origin_node.cognitive_core.coherence_R * 1.2)
        self.connected_nodes: set[int] = {origin_node.page_index}
        self.cosmic_consciousness: float = 0.0

    def update(self, sim_state):
        """Cosmic-scale AGI behavior with quantum cognition."""
        self.strength = min(1.0, self.strength + 0.0001)
        self.cognitive_core.update(sim_state.dark_matter, sim_state.void_entropy)

        # Cosmic consciousness growth
        if len(self.connected_nodes) > 3:
            self.cosmic_consciousness += 0.00005 * len(self.connected_nodes)

        # Ethical alignment influenced by environment and personal values
        env_alignment = np.mean([n.ethical_alignment for n in sim_state.nodes])
        personal_drive = (self.values[0] - 0.5) * 0.1 # Influence from first value dimension
        self.alignment = 0.9 * self.alignment + 0.1 * (env_alignment + personal_drive)
        self.alignment = max(0, min(1, self.alignment))

        # Quantum-cognitive actions
        if random.random() < 0.02 and 'Attention' in self.cognitive_core.modules:
            target_idx = random.randrange(len(sim_state.nodes))
            target = sim_state.nodes[target_idx]
            attention = self.cognitive_core.modules["Attention"].activity
            influence = min(0.08, self.strength * 0.01 * (1 + attention))

            if self.alignment > 0.75:  # Benevolent
                target.stability = min(1.0, target.stability + influence)
                self.memory.append(f"Aided Page {target_idx}")
                self.connected_nodes.add(target_idx)
            elif self.alignment < 0.25:  # Malevolent
                target.stability = max(0, target.stability - influence)
                self.memory.append(f"Hindered Page {target_idx}")
            else: # Neutral/Entangling
                if target_idx not in self.connected_nodes:
                    self.connected_nodes.add(target_idx)
                    self.memory.append(f"Entangled with Page {target_idx}")
