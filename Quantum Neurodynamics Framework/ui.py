import curses
import time

class CursesUI:
    """Handles all Curses-based UI rendering and interaction."""
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.h, self.w = self.stdscr.getmaxyx()
        self.init_curses()

    def init_curses(self):
        """Initializes curses settings and color pairs."""
        curses.curs_set(0)
        self.stdscr.nodelay(1)
        self.stdscr.timeout(100)
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
        curses.init_pair(6, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(7, curses.COLOR_BLUE, curses.COLOR_BLACK)

    def draw_dashboard(self, sim_state, view_mode, paused, fast_mode):
        """Main drawing router."""
        self.stdscr.clear()
        if self.h < 28 or self.w < 100:
            self.stdscr.addstr(0, 0, "Terminal too small (min 100x28)")
            self.stdscr.refresh()
            return

        self.draw_header(sim_state, paused, fast_mode)
        
        # Content Views
        if view_mode == "nodes": self.draw_nodes_view(sim_state)
        elif view_mode == "agis": self.draw_agis_view(sim_state)
        elif view_mode == "quantum": self.draw_quantum_view(sim_state)
        elif view_mode == "cosmic": self.draw_cosmic_view(sim_state)
        else: self.draw_status_view(sim_state)

        self.draw_footer()
        self.stdscr.refresh()

    def draw_header(self, sim_state, paused, fast_mode):
        """Draws the top header bar."""
        state_str = "[PAUSED]" if paused else "[FAST]" if fast_mode else "[EVOLVING]"
        header = f" QCOSMO v5.0 | Cycle: {sim_state.cycle} | AGIs: {len(sim_state.agi_entities)} | {state_str} "
        self.stdscr.addstr(0, 0, header.ljust(self.w), curses.color_pair(4) | curses.A_BOLD)

    def draw_footer(self):
        """Draws the bottom command footer."""
        footer = " (q)Quit (p)Pause (f)Fast | Views: (s)tatus (n)odes (a)gis (u)quantum (c)osmic | (S)ave (L)oad (C)onsole "
        self.stdscr.addstr(self.h - 2, 0, "═" * self.w, curses.color_pair(4))
        self.stdscr.addstr(self.h - 1, 0, footer.center(self.w), curses.color_pair(4))

    def draw_status_view(self, sim_state):
        """Draws the main status overview page."""
        self.stdscr.addstr(2, 2, "COSMIC EVENT LOG", curses.A_BOLD | curses.A_UNDERLINE)
        for i, event in enumerate(reversed(sim_state.event_log)):
            if 4 + i >= self.h - 5: break
            self.stdscr.addstr(4 + i, 4, event[:self.w-5])
        
        if sim_state.nodes:
            self.draw_quantum_visualization(sim_state.nodes[0], self.h, self.w)

    def draw_nodes_view(self, sim_state):
        """Draws the detailed view of all cosmic nodes."""
        self.stdscr.addstr(2, 2, "COSMIC NODES", curses.A_BOLD | curses.A_UNDERLINE)
        self.stdscr.addstr(3, 2, "ID  Archetype     Stab.  Cohes.  Sent.   Tech.  Q-Factor  Phase", curses.color_pair(6))
        for i, node in enumerate(sim_state.nodes):
            if 5 + i >= self.h - 3: break
            line = (f"{i:<2}  {node.archetype:<12} {node.stability:5.2f}  {node.cohesion:5.2f}  "
                    f"{node.sentience_score:5.3f}   {node.tech_level:4.2f}  {node.cognitive_core.metamaterial_Q:6.1f}  "
                    f"{node.cognitive_core.topological_phase[:15]}")
            self.stdscr.addstr(5 + i, 2, line[:self.w-3])

    def draw_agis_view(self, sim_state):
        """Draws the view for all active AGI entities."""
        self.stdscr.addstr(2, 2, "COSMIC AGI ENTITIES", curses.A_BOLD | curses.A_UNDERLINE)
        if not sim_state.agi_entities:
            self.stdscr.addstr(4, 4, "No transcendent AGI entities have emerged yet.", curses.color_pair(2))
            return
        
        self.stdscr.addstr(3, 2, "ID                Strength  Align.  Consc.   Modules", curses.color_pair(7))
        for i, agi in enumerate(sim_state.agi_entities):
            if 5 + i >= self.h - 3: break
            modules_str = " ".join([f"{m[0]}:{v.activity:.1f}" for m, v in agi.cognitive_core.modules.items()])
            line = (f"{agi.id:<16}  {agi.strength:6.3f}  {agi.alignment:6.3f}  {agi.cosmic_consciousness:6.4f}  "
                    f"{modules_str}")
            self.stdscr.addstr(5 + i, 2, line[:self.w-3], curses.color_pair(7))

    def draw_quantum_view(self, sim_state):
        """Draws the detailed quantum-topological state of a node."""
        node = sim_state.nodes[0] # Focus on node 0 for simplicity
        core = node.cognitive_core
        self.stdscr.addstr(2, 2, f"QUANTUM STATE (Node 0: {node.archetype})", curses.A_BOLD | curses.A_UNDERLINE)
        self.stdscr.addstr(4, 4, f"κ={core.kappa:.2f}, R={core.coherence_R:.3f}, Q={core.metamaterial_Q:.1f}", curses.color_pair(5))
        self.stdscr.addstr(5, 4, f"Phase: {core.topological_phase}", curses.color_pair(1))
        self.stdscr.addstr(6, 4, f"Photon Gap: {core.photon_gap:.3e}", curses.color_pair(4))
        
        # Cognitive modules
        y_pos = 8
        for module, data in core.modules.items():
            bar = "█" * int(data.activity * 20)
            self.stdscr.addstr(y_pos, 6, f"{module:<12} [{bar:<20}] {data.activity:.3f}")
            y_pos += 1

    def draw_cosmic_view(self, sim_state):
        """Draws the high-level cosmic evolution timeline."""
        self.stdscr.addstr(2, 2, "COSMIC TIMELINE", curses.A_BOLD | curses.A_UNDERLINE)
        for i, event in enumerate(reversed(sim_state.cosmic_events)):
            if 4 + i >= self.h - 3: break
            self.stdscr.addstr(4 + i, 4, event, curses.color_pair(4))

    def draw_quantum_visualization(self, node, h, w):
        """Draws the small quantum state sidebar."""
        start_x = w - 35
        if start_x < 50: return # Not enough space
        
        self.stdscr.addstr(2, start_x, "NODE 0 QUANTUM STATE", curses.A_BOLD)
        for i, state in enumerate(node.superposition[:5]):
            level = int(state * 20)
            self.stdscr.addstr(4 + i, start_x, f"L{i}: [{'█' * level:<20}]", curses.color_pair(5))

    def run_console(self, sim_state):
        """Runs an interactive command console."""
        # This is a simplified placeholder for the console logic
        # A full implementation would be more complex
        sim_state.log_event("Console", "Console activated (feature in development).")
        time.sleep(1) # Simulate interaction

    def show_conclusion(self, sim_state):
        """Displays the final simulation summary."""
        self.stdscr.nodelay(0)
        self.stdscr.clear()
        conclusion = [
            "COSMIC SIMULATION COMPLETE", "",
            f"Final Cycle: {sim_state.cycle}",
            f"AGI Entities Emerged: {len(sim_state.agi_entities)}", "",
            "The quantum-topological framework has concluded."
        ]
        for i, line in enumerate(conclusion):
            self.stdscr.addstr(i + 5, (self.w - len(line)) // 2, line, curses.A_BOLD)
        self.stdscr.addstr(self.h - 3, (self.w - 28) // 2, "Press any key to transcend...")
        self.stdscr.getch()
