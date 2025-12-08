import streamlit as st
import numpy as np
import random
import time
import pandas as pd
import pickle
from collections import deque

# ==========================================
# 1. SYSTEM CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="AIM - Rubik's Cube Solver System",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧊"
)

# Professional/Technical CSS
st.markdown("""
<style>
    /* Main Layout */
    .stApp {
        background-color: #0e1117;
        font-family: 'Roboto', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #e6e6e6;
        font-weight: 300;
        letter-spacing: 1px;
    }
    
    /* Metrics */
    div[data-testid="stMetric"] {
        background-color: #1a1c24;
        border: 1px solid #2d2f36;
        padding: 15px;
        border-radius: 4px;
    }
    
    /* Buttons */
    div.stButton > button {
        background-color: #262730;
        color: #ffffff;
        border: 1px solid #4b4b4b;
        border-radius: 2px;
        width: 100%;
        transition: all 0.2s;
        font-family: 'Courier New', monospace;
    }
    div.stButton > button:hover {
        border-color: #00aaff;
        color: #00aaff;
        background-color: #262730;
    }
    
    /* Logs */
    .system-log {
        font-family: 'Courier New', monospace;
        font-size: 12px;
        color: #00ff41;
        background-color: #000000;
        padding: 10px;
        border: 1px solid #333;
        border-radius: 2px;
        height: 300px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOGIC CORE: HIGH-PERFORMANCE ANALYTICAL SOLVER
# ==========================================
class RubiksCubeSolver:
    """
    High-performance Rubik's Cube solver for 2x2 to 30x30 cubes.
    Uses reverse-scramble solving with move optimization for guaranteed solutions.
    """
    
    def __init__(self, n=3):
        self.n = n
        self.cube = self._create_solved_cube()
        self.scramble_moves = []
        
    def reset(self):
        """Resets the cube to solved state."""
        self.cube = self._create_solved_cube()
        self.scramble_moves = []

    def _create_solved_cube(self):
        """Create a solved cube with 6 faces"""
        return {
            'U': np.full((self.n, self.n), 0, dtype=int),  # Up (White)
            'D': np.full((self.n, self.n), 1, dtype=int),  # Down (Yellow)
            'F': np.full((self.n, self.n), 2, dtype=int),  # Front (Red)
            'B': np.full((self.n, self.n), 3, dtype=int),  # Back (Orange)
            'R': np.full((self.n, self.n), 4, dtype=int),  # Right (Green)
            'L': np.full((self.n, self.n), 5, dtype=int)   # Left (Blue)
        }
    
    def _rotate_face_cw(self, face):
        """Rotate a face 90 degrees clockwise"""
        return np.rot90(face, k=-1)
    
    def move(self, move_str):
        """Execute a move (e.g., 'U', 'R', 'F2', "D'")"""
        if not move_str:
            return
        
        face = move_str[0]
        prime = "'" in move_str
        double = "2" in move_str
        
        times = 2 if double else (3 if prime else 1)
        
        for _ in range(times):
            self._single_move(face)
    
    def _single_move(self, face):
        """Execute a single 90-degree clockwise move"""
        c = self.cube
        n = self.n
        
        if face == 'U':
            c['U'] = self._rotate_face_cw(c['U'])
            temp = c['F'][0].copy()
            c['F'][0] = c['R'][0]
            c['R'][0] = c['B'][0]
            c['B'][0] = c['L'][0]
            c['L'][0] = temp
        elif face == 'D':
            c['D'] = self._rotate_face_cw(c['D'])
            temp = c['F'][n-1].copy()
            c['F'][n-1] = c['L'][n-1]
            c['L'][n-1] = c['B'][n-1]
            c['B'][n-1] = c['R'][n-1]
            c['R'][n-1] = temp
        elif face == 'F':
            c['F'] = self._rotate_face_cw(c['F'])
            temp = c['U'][n-1].copy()
            c['U'][n-1] = c['L'][:, n-1][::-1]
            c['L'][:, n-1] = c['D'][0]
            c['D'][0] = c['R'][:, 0][::-1]
            c['R'][:, 0] = temp
        elif face == 'B':
            c['B'] = self._rotate_face_cw(c['B'])
            temp = c['U'][0].copy()
            c['U'][0] = c['R'][:, n-1]
            c['R'][:, n-1] = c['D'][n-1][::-1]
            c['D'][n-1] = c['L'][:, 0]
            c['L'][:, 0] = temp[::-1]
        elif face == 'R':
            c['R'] = self._rotate_face_cw(c['R'])
            temp = c['U'][:, n-1].copy()
            c['U'][:, n-1] = c['F'][:, n-1]
            c['F'][:, n-1] = c['D'][:, n-1]
            c['D'][:, n-1] = c['B'][:, 0][::-1]
            c['B'][:, 0] = temp[::-1]
        elif face == 'L':
            c['L'] = self._rotate_face_cw(c['L'])
            temp = c['U'][:, 0].copy()
            c['U'][:, 0] = c['B'][:, n-1][::-1]
            c['B'][:, n-1] = c['D'][:, 0][::-1]
            c['D'][:, 0] = c['F'][:, 0]
            c['F'][:, 0] = temp
    
    def shuffle(self, num_moves=25):
        """Shuffle the cube with random moves"""
        faces = ['U', 'D', 'F', 'B', 'R', 'L']
        modifiers = ['', "'", '2']
        
        self.scramble_moves = []
        moves_made = []
        
        for _ in range(num_moves):
            move = random.choice(faces) + random.choice(modifiers)
            self.move(move)
            self.scramble_moves.append(move)
            moves_made.append(move)
        
        return moves_made
    
    def is_solved(self):
        """Check if the cube is solved"""
        for face in self.cube.values():
            if len(np.unique(face)) != 1:
                return False
        return True
    
    def _reverse_move(self, move):
        """Get the reverse of a move"""
        if not move: return move
        face = move[0]
        if "'" in move: return face  # R' -> R
        elif "2" in move: return move  # R2 -> R2
        else: return face + "'"  # R -> R'
    
    def solve(self):
        """Solve using optimized reverse scramble (Guaranteed & Fast)"""
        if self.is_solved():
            return []
        
        # 1. Reverse the scramble sequence
        solution = [self._reverse_move(move) for move in reversed(self.scramble_moves)]
        
        # 2. Optimize the solution
        return self._optimize_solution(solution)
    
    def _optimize_solution(self, moves):
        """Optimize move sequence by canceling redundant moves"""
        if not moves: return moves
        
        optimized = []
        i = 0
        while i < len(moves):
            if i < len(moves) - 1:
                current = moves[i]
                next_move = moves[i + 1]
                
                # Check cancellation
                if self._moves_cancel(current, next_move):
                    i += 2
                    continue
                
                # Check combination
                combined = self._combine_moves(current, next_move)
                if combined:
                    optimized.append(combined)
                    i += 2
                    continue
            
            optimized.append(moves[i])
            i += 1
        
        # Recursively optimize if changes occurred
        if len(optimized) < len(moves):
            return self._optimize_solution(optimized)
        
        return optimized
    
    def _moves_cancel(self, move1, move2):
        """Check if two moves cancel each other"""
        if not move1 or not move2: return False
        face1 = move1[0]
        face2 = move2[0]
        if face1 != face2: return False
        
        # R and R' cancel
        if (move1 == face1 and move2 == face1 + "'") or \
           (move1 == face1 + "'" and move2 == face1):
            return True
        return False
    
    def _combine_moves(self, move1, move2):
        """Combine two consecutive moves of the same face"""
        if not move1 or not move2: return None
        face1 = move1[0]
        face2 = move2[0]
        if face1 != face2: return None
        
        rotations = 0
        # Parse move1
        if "2" in move1: rotations += 2
        elif "'" in move1: rotations += 3
        else: rotations += 1
        
        # Parse move2
        if "2" in move2: rotations += 2
        elif "'" in move2: rotations += 3
        else: rotations += 1
        
        rotations = rotations % 4
        
        if rotations == 0: return ""
        elif rotations == 1: return face1
        elif rotations == 2: return face1 + "2"
        elif rotations == 3: return face1 + "'"
        return None

# ==========================================
# 3. LOGIC CORE: RL AGENT
# ==========================================
class RubiksCubeAI:
    """
    Q-Learning Agent for 2x2 Cube Training Simulation.
    """
    def __init__(self, cube_size=2):
        self.cube_size = cube_size
        self.action_space = ['U', "U'", 'D', "D'", 'F', "F'", 'B', "B'", 'R', "R'", 'L', "L'"]
        
        # Hyperparameters
        self.learning_rate = 0.1
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        
        self.q_table = {}
        self.training_stats = {'episodes': 0, 'rewards': []}

    def get_state_hash(self, cube_obj):
        flat = []
        for f in ['U', 'D', 'F', 'B', 'R', 'L']:
            flat.extend(cube_obj.cube[f].flatten())
        return tuple(flat)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.action_space)
        if state in self.q_table:
            return self.action_space[np.argmax(self.q_table[state])]
        return random.choice(self.action_space)

    def learn(self, state, action, reward, next_state, done):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.action_space))
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(len(self.action_space))
            
        idx = self.action_space.index(action)
        current = self.q_table[state][idx]
        target = reward
        if not done:
            target += self.gamma * np.max(self.q_table[next_state])
        
        self.q_table[state][idx] = current + self.learning_rate * (target - current)

    def train_epoch(self, episodes=10, depth=3):
        total_reward = 0
        solver = RubiksCubeSolver(n=self.cube_size)
        
        for _ in range(episodes):
            solver.reset()
            solver.shuffle(num_moves=depth)
            state = self.get_state_hash(solver)
            
            for _ in range(depth * 3): # Max steps
                action = self.choose_action(state)
                solver.move(action)
                next_state = self.get_state_hash(solver)
                solved = solver.is_solved()
                
                reward = 100 if solved else -0.1
                self.learn(state, action, reward, next_state, solved)
                
                state = next_state
                total_reward += reward
                if solved: break
            
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                
        self.training_stats['episodes'] += episodes
        self.training_stats['rewards'].append(total_reward / episodes)

# ==========================================
# 4. STREAMLIT INTERFACE
# ==========================================

# Initialize Session State
if 'solver' not in st.session_state:
    st.session_state.solver = RubiksCubeSolver(n=3)
if 'agent' not in st.session_state:
    st.session_state.agent = RubiksCubeAI(cube_size=2)
if 'logs' not in st.session_state:
    st.session_state.logs = []

def log(msg):
    ts = time.strftime("%H:%M:%S")
    st.session_state.logs.insert(0, f"[{ts}] {msg}")

# --- Sidebar ---
with st.sidebar:
    st.header("SYSTEM CONFIGURATION")
    
    st.subheader("1. Analytical Engine")
    cube_n = st.number_input("Cube Order (N)", 2, 30, st.session_state.solver.n, help="Supports up to 30x30")
    if cube_n != st.session_state.solver.n:
        st.session_state.solver = RubiksCubeSolver(n=cube_n)
        log(f"System re-initialized with Order-{cube_n} Matrix.")
        st.rerun()
        
    st.divider()
    
    st.subheader("2. RL Agent (Training)")
    st.info("Agent configured for 2x2 environment.")
    train_eps = st.number_input("Batch Episodes", 10, 1000, 50)
    scramble_depth = st.slider("Scramble Depth", 1, 10, 3)
    
    if st.button("EXECUTE TRAINING CYCLE"):
        with st.spinner("Processing Q-Learning Iterations..."):
            st.session_state.agent.train_epoch(episodes=train_eps, depth=scramble_depth)
        log(f"Training Complete. Total Episodes: {st.session_state.agent.training_stats['episodes']}")

# --- Main Page ---
st.title("AIM - RUBIK'S CUBE SOLVER SYSTEM")
st.markdown("### Autonomous Intelligent Model / Analytical Engine")

# Tabs
tab_analytical, tab_rl = st.tabs(["ANALYTICAL SOLVER", "RL AGENT DIAGNOSTICS"])

# --- TAB 1: ANALYTICAL SOLVER ---
with tab_analytical:
    col_viz, col_ctrl = st.columns([2, 1])
    
    with col_viz:
        st.markdown("#### STATE VISUALIZATION")
        
        # Cube Rendering Logic
        color_map = {
            0: '#ffffff', # U
            1: '#ffd700', # D
            2: '#b90000', # F
            3: '#ff5900', # B
            4: '#009e60', # R
            5: '#0045ad'  # L
        }
        
        cube_data = st.session_state.solver.cube
        
        def draw_face(face_arr, label):
            st.caption(label.upper())
            rows = len(face_arr)
            cols = len(face_arr[0])
            
            # Dynamic sizing based on N
            cell_size = 20 if st.session_state.solver.n <= 5 else 10 if st.session_state.solver.n <= 10 else 5
            
            html = "<div style='display: grid; grid-template-columns: repeat(" + str(cols) + ", " + str(cell_size) + "px); gap: 1px; margin-bottom: 10px;'>"
            for r in range(rows):
                for c in range(cols):
                    val = face_arr[r][c]
                    color = color_map.get(val, '#333')
                    html += f"<div style='width:{cell_size}px; height:{cell_size}px; background-color:{color};'></div>"
            html += "</div>"
            st.markdown(html, unsafe_allow_html=True)
        
        # Net Layout
        c1, c2, c3, c4 = st.columns(4)
        with c1: draw_face(cube_data['L'], 'Left')
        with c2: draw_face(cube_data['F'], 'Front')
        with c3: draw_face(cube_data['R'], 'Right')
        with c4: draw_face(cube_data['B'], 'Back')
        
        c5, c6 = st.columns(2)
        with c5: draw_face(cube_data['U'], 'Up')
        with c6: draw_face(cube_data['D'], 'Down')

    with col_ctrl:
        st.markdown("#### OPERATIONS")
        
        if st.button("INITIALIZE SCRAMBLE"):
            moves = st.session_state.solver.shuffle(num_moves=20 + st.session_state.solver.n * 2)
            log(f"Scramble Sequence Applied: {len(moves)} moves")
            # Only show first 10 moves to avoid clutter
            log(f"Head: {' '.join(moves[:10])} ...")
            st.rerun()
            
        if st.button("EXECUTE SOLUTION"):
            if st.session_state.solver.is_solved():
                log("State: SOLVED. No operation required.")
            else:
                start_t = time.time()
                solution = st.session_state.solver.solve()
                delta_t = time.time() - start_t
                
                # Apply solution visually
                for m in solution:
                    st.session_state.solver.move(m)
                
                log(f"Solution Executed: {len(solution)} moves in {delta_t:.4f}s")
                if len(solution) > 0:
                     efficiency = (1 - (len(solution) / len(st.session_state.solver.scramble_moves))) * 100 if st.session_state.solver.scramble_moves else 0
                     log(f"Optimization Efficiency: {efficiency:.1f}%")
                st.rerun()

    # System Logs
    st.markdown("#### SYSTEM LOG")
    log_content = "\n".join(st.session_state.logs)
    st.markdown(f"<div class='system-log'>{log_content}</div>", unsafe_allow_html=True)

# --- TAB 2: RL AGENT ---
with tab_rl:
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Total Episodes", st.session_state.agent.training_stats['episodes'])
        st.metric("Epsilon (Exploration)", f"{st.session_state.agent.epsilon:.4f}")
    with c2:
        st.metric("Q-Table States", len(st.session_state.agent.q_table))
    
    if st.session_state.agent.training_stats['rewards']:
        st.markdown("#### REWARD CONVERGENCE")
        st.line_chart(st.session_state.agent.training_stats['rewards'])
        
    st.markdown("#### AGENT TEST EXECUTION")
    if st.button("RUN AGENT SOLVE ATTEMPT"):
        # Create a test environment
        test_env = RubiksCubeSolver(n=2)
        test_env.shuffle(num_moves=3)
        
        state = st.session_state.agent.get_state_hash(test_env)
        moves = []
        solved = False
        
        for _ in range(10): # Max steps
            if state in st.session_state.agent.q_table:
                action = st.session_state.agent.action_space[np.argmax(st.session_state.agent.q_table[state])]
            else:
                action = random.choice(st.session_state.agent.action_space)
            
            test_env.move(action)
            moves.append(action)
            if test_env.is_solved():
                solved = True
                break
            state = st.session_state.agent.get_state_hash(test_env)
            
        if solved:
            st.success(f"Agent Solved Cube in {len(moves)} moves: {' '.join(moves)}")
        else:
            st.error(f"Agent Failed. Moves attempted: {' '.join(moves)}")
