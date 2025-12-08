import streamlit as st
import numpy as np
import random
import time
import pandas as pd
import json
import pickle
from collections import deque

# ==========================================
# 1. SYSTEM CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="AIM - RL Rubik's Solver",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧊"
)

# Formal, High-Tech CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        font-family: 'Roboto Mono', monospace;
    }
    div.stButton > button {
        background-color: #262730;
        color: #ffffff;
        border: 1px solid #4b4b4b;
        border-radius: 4px;
        transition: all 0.3s;
    }
    div.stButton > button:hover {
        border-color: #00d2ff;
        color: #00d2ff;
    }
    .metric-card {
        background-color: #1e1e1e;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    h1, h2, h3 {
        color: #e0e0e0;
    }
    .console-log {
        font-family: 'Courier New', monospace;
        background-color: #000;
        color: #0f0;
        padding: 10px;
        border-radius: 5px;
        height: 200px;
        overflow-y: auto;
        font-size: 0.85em;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. RUBIK'S CUBE ENVIRONMENT (LOGIC CORE)
# ==========================================
class RubiksCubeEnvironment:
    """
    Simulates an NxN Rubik's Cube.
    State representation: Dictionary of 6 faces, each an NxN numpy array.
    """
    def __init__(self, n=3):
        self.n = n
        self.cube = self._create_solved_cube()
        self.scramble_history = []
        
    def _create_solved_cube(self):
        # 0:U(White), 1:D(Yellow), 2:F(Red), 3:B(Orange), 4:R(Green), 5:L(Blue)
        return {
            'U': np.full((self.n, self.n), 0, dtype=int),
            'D': np.full((self.n, self.n), 1, dtype=int),
            'F': np.full((self.n, self.n), 2, dtype=int),
            'B': np.full((self.n, self.n), 3, dtype=int),
            'R': np.full((self.n, self.n), 4, dtype=int),
            'L': np.full((self.n, self.n), 5, dtype=int)
        }
    
    def reset(self):
        self.cube = self._create_solved_cube()
        self.scramble_history = []

    def _rotate_face_cw(self, face):
        return np.rot90(face, k=-1)
    
    def move(self, move_str):
        if not move_str: return
        face = move_str[0]
        prime = "'" in move_str
        double = "2" in move_str
        times = 2 if double else (3 if prime else 1)
        
        for _ in range(times):
            self._single_move(face)
            
    def _single_move(self, face):
        c = self.cube
        n = self.n
        if face == 'U':
            c['U'] = self._rotate_face_cw(c['U'])
            temp = c['F'][0].copy()
            c['F'][0], c['R'][0], c['B'][0], c['L'][0] = c['R'][0], c['B'][0], c['L'][0], temp
        elif face == 'D':
            c['D'] = self._rotate_face_cw(c['D'])
            temp = c['F'][n-1].copy()
            c['F'][n-1], c['L'][n-1], c['B'][n-1], c['R'][n-1] = c['L'][n-1], c['B'][n-1], c['R'][n-1], temp
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

    def shuffle(self, num_moves=20):
        faces = ['U', 'D', 'F', 'B', 'R', 'L']
        modifiers = ['', "'", '2']
        moves = []
        for _ in range(num_moves):
            m = random.choice(faces) + random.choice(modifiers)
            self.move(m)
            moves.append(m)
        self.scramble_history = moves
        return moves

    def is_solved(self):
        for face in self.cube.values():
            if len(np.unique(face)) != 1: return False
        return True

    # --- ANALYTICAL SOLVER (Expert System) ---
    def _reverse_move(self, move):
        if "'" in move: return move[0]
        elif "2" in move: return move
        else: return move + "'"

    def solve_analytical(self):
        """Solves by reversing the scramble history and optimizing."""
        if not self.scramble_history: return []
        raw_solution = [self._reverse_move(m) for m in reversed(self.scramble_history)]
        return self._optimize_solution(raw_solution)

    def _optimize_solution(self, moves):
        if not moves: return []
        optimized = []
        i = 0
        while i < len(moves):
            if i < len(moves) - 1:
                curr, nxt = moves[i], moves[i+1]
                if curr[0] == nxt[0]:
                    # Check cancellation (R R')
                    if (curr == nxt[0] and nxt == curr[0]+"'") or (curr == nxt[0]+"'" and nxt == curr[0]):
                        i += 2; continue
                optimized.append(curr)
                i += 1
            else:
                optimized.append(moves[i])
                i += 1
        return optimized if len(optimized) == len(moves) else self._optimize_solution(optimized)

# ==========================================
# 3. REINFORCEMENT LEARNING AGENT (Q-LEARNING)
# ==========================================
class RLAgent:
    def __init__(self, cube_size=2):
        self.cube_size = cube_size
        self.action_space = ['U', "U'", 'D', "D'", 'F', "F'", 'B', "B'", 'R', "R'", 'L', "L'"]
        
        # Hyperparameters
        self.learning_rate = 0.1
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        
        # Q-Table: Map state_hash -> np.array(q_values)
        self.q_table = {}
        
        # Metrics
        self.training_log = []

    def get_state_hash(self, cube_env):
        # Flatten all faces to create a unique signature
        flat = []
        for f in ['U', 'D', 'F', 'B', 'R', 'L']:
            flat.extend(cube_env.cube[f].flatten())
        return tuple(flat)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)
        
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.action_space))
            
        action_idx = np.argmax(self.q_table[state])
        return self.action_space[action_idx]

    def learn(self, state, action_str, reward, next_state, done):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.action_space))
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(len(self.action_space))
            
        action_idx = self.action_space.index(action_str)
        
        target = reward
        if not done:
            target = reward + self.gamma * np.max(self.q_table[next_state])
            
        current = self.q_table[state][action_idx]
        self.q_table[state][action_idx] = current + self.learning_rate * (target - current)

    def calculate_reward(self, env):
        if env.is_solved(): return 100
        # Simple heuristic: negative reward per step to encourage speed
        return -0.1

    def train_step(self, env):
        state = self.get_state_hash(env)
        action = self.choose_action(state)
        
        # Execute
        env.move(action)
        next_state = self.get_state_hash(env)
        done = env.is_solved()
        reward = self.calculate_reward(env)
        
        # Learn
        self.learn(state, action, reward, next_state, done)
        
        # Epsilon decay
        if done and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return done, reward

# ==========================================
# 4. STREAMLIT UI IMPLEMENTATION
# ==========================================

# --- Initialization ---
if 'cube_env' not in st.session_state:
    st.session_state.cube_env = RubiksCubeEnvironment(n=3)
if 'rl_agent' not in st.session_state:
    st.session_state.rl_agent = RLAgent(cube_size=2) # RL works best on 2x2 for demo
if 'console_log' not in st.session_state:
    st.session_state.console_log = []

def log(message):
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.console_log.insert(0, f"[{timestamp}] {message}")
    if len(st.session_state.console_log) > 50:
        st.session_state.console_log.pop()

# --- Sidebar Controls ---
with st.sidebar:
    st.title("⚙️ System Config")
    
    st.subheader("Environment Parameters")
    c_size = st.number_input("Cube Order (N)", min_value=2, max_value=10, value=3)
    if c_size != st.session_state.cube_env.n:
        st.session_state.cube_env = RubiksCubeEnvironment(n=c_size)
        log(f"Environment re-initialized. Order: {c_size}")

    st.divider()
    
    st.subheader("RL Hyperparameters")
    lr = st.slider("Learning Rate (α)", 0.01, 1.0, 0.1)
    gamma = st.slider("Discount Factor (γ)", 0.5, 0.99, 0.95)
    eps = st.slider("Exploration Rate (ε)", 0.0, 1.0, st.session_state.rl_agent.epsilon)
    
    # Update Agent Params
    st.session_state.rl_agent.learning_rate = lr
    st.session_state.rl_agent.gamma = gamma
    st.session_state.rl_agent.epsilon = eps

# --- Main Interface ---
st.title("AIM - Autonomous Intelligent Model")
st.caption("Reinforcement Learning Rubik's Cube Solver Agent")

tab1, tab2 = st.tabs(["🖥️ Operations Console", "🧠 Neural Training"])

# --- TAB 1: OPERATIONS ---
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("State Visualization")
        
        # Cube Visualization (2D Map)
        c = st.session_state.cube_env.cube
        
        # Color mapping for visualization
        color_map = {
            0: '#ffffff', # White (U)
            1: '#ffd700', # Yellow (D)
            2: '#ff0000', # Red (F)
            3: '#ff8c00', # Orange (B)
            4: '#00ff00', # Green (R)
            5: '#0000ff'  # Blue (L)
        }
        
        def render_face(face_data, label):
            st.write(f"**{label}**")
            cols = st.columns(len(face_data))
            for i, row in enumerate(face_data):
                row_html = ""
                for val in row:
                    color = color_map.get(val, '#333')
                    row_html += f'<div style="width:20px;height:20px;background-color:{color};border:1px solid #000;display:inline-block;margin:1px;"></div>'
                st.markdown(f"<div>{row_html}</div>", unsafe_allow_html=True)

        # Layout for cube net
        r1, r2, r3, r4 = st.columns(4)
        with r1: render_face(c['L'], 'Left')
        with r2: render_face(c['F'], 'Front')
        with r3: render_face(c['R'], 'Right')
        with r4: render_face(c['B'], 'Back')
        
        r5, r6 = st.columns(2)
        with r5: render_face(c['U'], 'Up')
        with r6: render_face(c['D'], 'Down')

    with col2:
        st.subheader("Control Unit")
        
        if st.button("🎲 Scramble Environment"):
            moves = st.session_state.cube_env.shuffle(num_moves=20)
            log(f"Scrambled: {' '.join(moves)}")
            st.rerun()
            
        if st.button("🧩 Analytical Solve (Expert)"):
            if st.session_state.cube_env.is_solved():
                log("System State: Already Solved.")
            else:
                solution = st.session_state.cube_env.solve_analytical()
                # Execute visual solve
                for m in solution:
                    st.session_state.cube_env.move(m)
                log(f"Analytical Solution Executed: {len(solution)} moves")
                st.rerun()
        
        if st.button("🤖 RL Agent Attempt"):
            # Simple greedy attempt using Q-table
            log("Agent initiating solve attempt...")
            moves_made = []
            limit = 20
            solved = False
            for _ in range(limit):
                if st.session_state.cube_env.is_solved():
                    solved = True
                    break
                state = st.session_state.rl_agent.get_state_hash(st.session_state.cube_env)
                # Pure exploitation for testing
                if state in st.session_state.rl_agent.q_table:
                    action_idx = np.argmax(st.session_state.rl_agent.q_table[state])
                    action = st.session_state.rl_agent.action_space[action_idx]
                else:
                    action = random.choice(st.session_state.rl_agent.action_space)
                
                st.session_state.cube_env.move(action)
                moves_made.append(action)
            
            if solved:
                log(f"Agent Success! Path: {' '.join(moves_made)}")
            else:
                log(f"Agent Failed (Limit {limit}). Path: {' '.join(moves_made)}")
            st.rerun()

        st.subheader("System Log")
        log_text = "\n".join(st.session_state.console_log)
        st.text_area("Console Output", log_text, height=150, disabled=True)

# --- TAB 2: TRAINING ---
with tab2:
    st.subheader("Reinforcement Learning Pipeline")
    
    col_t1, col_t2 = st.columns([1, 2])
    
    with col_t1:
        st.info("Note: Training is optimized for 2x2 Cubes due to state-space complexity in browser environment.")
        episodes = st.number_input("Training Episodes", 10, 1000, 100)
        
        if st.button("🚀 Start Training Session"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Temporary training env
            train_env = RubiksCubeEnvironment(n=st.session_state.rl_agent.cube_size)
            
            rewards_history = []
            
            for ep in range(episodes):
                # Reset
                train_env.reset()
                train_env.shuffle(num_moves=3) # Curriculum learning: Start with shallow scrambles
                
                total_reward = 0
                for step in range(10): # Max steps per episode
                    done, r = st.session_state.rl_agent.train_step(train_env)
                    total_reward += r
                    if done: break
                
                rewards_history.append(total_reward)
                progress_bar.progress((ep + 1) / episodes)
                status_text.text(f"Episode {ep+1}/{episodes} | Epsilon: {st.session_state.rl_agent.epsilon:.4f}")
            
            st.session_state.training_history = rewards_history
            log(f"Training Complete. {episodes} episodes processed.")
            log(f"Knowledge Base: {len(st.session_state.rl_agent.q_table)} states mapped.")
            st.rerun()

    with col_t2:
        if 'training_history' in st.session_state and st.session_state.training_history:
            st.subheader("Convergence Metrics")
            df = pd.DataFrame(st.session_state.training_history, columns=['Reward'])
            st.line_chart(df)
            
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("Total States Learned", len(st.session_state.rl_agent.q_table))
            col_m2.metric("Final Epsilon", f"{st.session_state.rl_agent.epsilon:.4f}")
        else:
            st.warning("No training data available. Initialize training session.")

# Footer
st.markdown("---")
st.markdown("AIM v1.0 | Reinforced Cube Solver Agent | Environment: Discrete N-Dimensional Grid")
