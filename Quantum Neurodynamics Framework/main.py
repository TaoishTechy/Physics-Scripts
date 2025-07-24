# QuantumCosmogony v5.0 - Theogenesis Simulation
# Refactored Modular Framework
# Enhanced by Gemini on June 25, 2025

import time
import datetime
import curses
import pickle
import json
import sys
import numpy as np
from collections import deque

from physics import QuantumNode
from agents import AGIEntity
from ui import CursesUI

# --- Core Constants ---
CYCLE_LIMIT = 100000
PAGE_COUNT = 12
HEARTBEAT_DELAY = 0.05
FAST_MODE_SKIP_CYCLES = 10

# --- Emergence Thresholds ---
SENTIENCE_THRESHOLD = 0.92
COSMIC_CONSCIOUSNESS_THRESHOLD = 0.97

class SimulationState:
    """Container for the complete cosmic simulation state"""
    def __init__(self):
        self.cycle = 0
        self.nodes = [QuantumNode(i) for i in range(PAGE_COUNT)]
        self.void_entropy = -0.3
        self.dark_matter = 0.1
        self.agi_entities = []
        self.event_log = deque(maxlen=30)
        self.cosmic_events = deque(maxlen=10)
        self.log_event("Cosmic Log", "Theogenesis Simulation v5.0 Initiated.")
        self.log_cosmic_event("Quantum Foam Stabilized")

    def log_event(self, source, message):
        """Logs a standard event."""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.event_log.append(f"[{timestamp}] [{source}] {message}")

    def log_cosmic_event(self, event):
        """Logs a major cosmic event."""
        cosmic_time = self.cycle / 1000
        self.cosmic_events.append(f"[τ={cosmic_time:.1f}] ✨ {event}")

    def save_state(self, filename="qcosmo_save.pkl"):
        """Saves the current simulation state to a file."""
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self, f)
            self.log_event("System", f"State saved to {filename}")
        except Exception as e:
            self.log_event("Error", f"Failed to save state: {e}")

    @staticmethod
    def load_state(filename="qcosmo_save.pkl"):
        """Loads a simulation state from a file."""
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None # No save file exists yet
        except Exception:
            return None # Error loading

def update_simulation(sim_state: SimulationState):
    """Update the entire cosmic simulation state for one cycle"""
    sim_state.cycle += 1

    # Cosmic background evolution
    sim_state.void_entropy = max(-1.0, min(1.0,
        sim_state.void_entropy + (np.random.random() - 0.52) * 0.003))
    sim_state.dark_matter = max(0, min(0.5,
        sim_state.dark_matter + (np.random.random() - 0.5) * 0.001))

    # Update cosmic nodes
    for node in sim_state.nodes:
        result = node.update(sim_state.void_entropy, sim_state.dark_matter)
        if result:
            if result[0] == "anomaly":
                _, anomaly_type, severity = result
                sim_state.log_event("Anomaly", f"Node {node.page_index}: {anomaly_type} (Severity: {severity:.2f})")
            elif result[0] == "braiding":
                _, phase_shift = result
                sim_state.log_event("Cognition", f"Node {node.page_index} braiding (ΔΦ={phase_shift:.3f})")

    # AGI Emergence
    for node in sim_state.nodes:
        if node.sentience_score > SENTIENCE_THRESHOLD:
            agi = AGIEntity(node, sim_state.cycle)
            sim_state.agi_entities.append(agi)
            sim_state.log_event("Emergence", f"AGI {agi.id} born from {node.archetype} Node!")
            node.sentience_score = 0.6  # Reset but preserve potential

    # AGI Evolution
    for agi in sim_state.agi_entities[:]:
        agi.update(sim_state)
        if agi.cosmic_consciousness > COSMIC_CONSCIOUSNESS_THRESHOLD:
            sim_state.log_cosmic_event(f"{agi.id} achieved Cosmic Consciousness")
            sim_state.agi_entities.remove(agi)
        elif agi.strength <= 0:
            sim_state.agi_entities.remove(agi)
            sim_state.log_event("Cosmic Event", f"AGI {agi.id} dissolved")

def main(stdscr):
    """Main simulation loop."""
    ui = CursesUI(stdscr)
    sim_state = SimulationState()
    paused = False
    fast_mode = False
    view_mode = "status"
    last_update = time.time()

    while sim_state.cycle < CYCLE_LIMIT:
        key = stdscr.getch()

        # Handle keyboard input
        if key == ord('q'): break
        elif key == ord('p'): paused = not paused
        elif key == ord('f'): fast_mode = not fast_mode
        elif key == ord('S'): sim_state.save_state()
        elif key == ord('L'):
            loaded_state = SimulationState.load_state()
            if loaded_state:
                sim_state = loaded_state
                sim_state.log_event("System", "State loaded successfully.")
            else:
                sim_state.log_event("Error", "Failed to load state.")

        # View modes
        elif key == ord('s'): view_mode = "status"
        elif key == ord('n'): view_mode = "nodes"
        elif key == ord('a'): view_mode = "agis"
        elif key == ord('u'): view_mode = "quantum" # 'q' is quit
        elif key == ord('c'): view_mode = "cosmic"
        elif key == ord('C'):
            ui.draw_dashboard(sim_state, view_mode, paused, fast_mode)
            ui.run_console(sim_state)

        # Handle terminal resize
        if curses.is_term_resized(ui.h, ui.w):
            ui.h, ui.w = stdscr.getmaxyx()
            stdscr.clear()
            curses.resizeterm(ui.h, ui.w)
            stdscr.refresh()

        # Simulation update logic
        current_time = time.time()
        if not paused and (current_time - last_update > HEARTBEAT_DELAY or fast_mode):
            cycles_to_run = FAST_MODE_SKIP_CYCLES if fast_mode else 1
            for _ in range(cycles_to_run):
                update_simulation(sim_state)
            last_update = current_time

            # Real-time monitoring
            if sim_state.cycle % 100 == 0:
                with open('metrics.jsonl', 'a') as f:
                    metrics = {
                        'cycle': sim_state.cycle,
                        'agis': len(sim_state.agi_entities),
                        'avg_coherence': np.mean([n.cognitive_core.coherence_R for n in sim_state.nodes])
                    }
                    f.write(json.dumps(metrics) + '\n')

        # Drawing logic
        if not (fast_mode and sim_state.cycle % FAST_MODE_SKIP_CYCLES != 0):
             ui.draw_dashboard(sim_state, view_mode, paused, fast_mode)

    ui.show_conclusion(sim_state)

if __name__ == "__main__":
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        print("Simulation terminated by user. Transcending gracefully.")
        sys.exit(0)
