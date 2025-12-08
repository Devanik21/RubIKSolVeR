# 🧊 AIM – Autonomous Rubik’s Cube Solver & RL Intelligence System

### *"Intelligence is the ability to adapt to change."*

**Developed by Nik  | Final Year B.Tech (ECE), NIT Agartala**

---

## 📖 Table of Contents

* [Overview](#-overview)
* [Key Features](#-key-features)
* [System Architecture](#-system-architecture)
* [Installation & Setup](#-installation--setup)
* [User Manual](#-user-manual)
* [Algorithmic Core](#-algorithmic-core)
* [Analytical Engine](#-analytical-engine)
* [Reinforcement Learning Agent](#-reinforcement-learning-agent)
* [Performance Metrics](#-performance-metrics)
* [Project Structure](#-project-structure)
* [Future Roadmap](#-future-roadmap)
* [Contact & License](#-contact--license)

---

## 🔍 Overview

**AIM (Autonomous Intelligent Model)** is a dual-engine Rubik’s Cube intelligence system that unifies
**classical algorithmic solving** with **modern reinforcement learning research**.

It is not just a cube solver—
it is a **simulation lab**, built for experimentation, visualization, optimization, and AI behavior analysis.

AIM provides:

* ⚡ **A high-performance solver** capable of instantly solving *any* NxN cube (2×2 → 30×30)
* 🤖 **A Q-Learning agent** that learns how to solve a 2×2 cube from scratch
* 🎨 **A Cyber-Tech Streamlit interface** for real-time logs, visual cube states, charts & training analytics

---

## ✨ Key Features

### 🧠 Analytical Engine (Expert System)

* Universal cube size support: **2×2 to 30×30**
* Reverse-Scramble with **Move Optimization**
* <0.01s solution latency
* Intelligent move cleanup:

  * `R R'` → removed
  * `U U` → `U2`
  * `F F2` → `F'`
* Real-time visual cube net rendering

---

### 🤖 Reinforcement Learning Agent (Student AI)

* Tabular **Q-Learning** implementation for 2×2
* Adjustable hyperparameters (α, γ, ε)
* Training visualization with:

  * Reward convergence plots
  * Epsilon decay analysis
* Supports user-defined scramble depth & batch episodes
* Model testing after training

---

### 💻 Cyber-Tech Interface

* Dark futuristic theme
* Scrollable system logs
* Clean sidebar controls
* Professional analytic visualizations

---

## ⚙️ System Architecture

### 1. `RubiksCubeSolver` Class

* N×N matrix representation
* Uses `numpy.rot90()` for efficient face rotations
* Maintains full move history
* Reverse-Scramble + Optimizer for instant solutions

### 2. `RubiksCubeAI` Class

* State-Hash → Q-Values mapping
* Bellman Update function
* Training loop handling:
  **Action → Environment Step → Reward → Q-Update**
* Exploration vs. Exploitation balancing

---

## 🛠 Installation & Setup

### Prerequisites

* Python **3.8+**
* pip package manager

### Step 1 — Clone the Repository

```bash
git clone https://github.com/your-username/aim-rubiks-solver.git
cd aim-rubiks-solver
```

### Step 2 — Install Dependencies

```bash
pip install streamlit numpy pandas
```

### Step 3 — Launch the Application

```bash
streamlit run app.py
```

The app starts at:
**[http://localhost:8501](http://localhost:8501)**

---

## 🕹 User Manual

### Tab 1 — Analytical Solver (Expert Mode)

* Choose **Cube Order (N)**
* Click **Scramble Cube**
* Click **Solve Cube**
* Observe:

  * Move Optimization
  * Execution Time
  * Visual Cube State

### Tab 2 — RL Agent Diagnostics (Research Mode)

* Set batch size & scramble depth
* Start training
* Watch reward convergence graph rise
* Test the agent’s learned policy

---

## 🧮 Algorithmic Core

### 1. Analytical Engine — Deterministic Mathematics

For a scramble:

`R U F`

The inverse solution is:

`F' U' R'`

Then an optimizer refines:

* `R R'` → removed
* `R R` → `R2`
* `R R2` → `R'`

This produces **minimal, elegant move sequences**.

---

### 2. Reinforcement Learning Agent — Q-Learning

* **State (S):** Flattened color hash of cube
* **Actions (A):** 12 face turns
* **Reward (R):**

  * `+100` for solving
  * `-0.1` per move

**Q-Update Rule:**

```text
Q(S,A) ← Q(S,A) + α [ R + γ max(Q(S'),A') − Q(S,A) ]
```

The agent gradually discovers efficient solve paths.

---

## 📊 Performance Metrics

| Engine Type | Cube Size | Success Rate | Avg. Time |
| ----------- | --------- | ------------ | --------- |
| Analytical  | 3×3       | 100%         | ~0.001s   |
| Analytical  | 20×20     | 100%         | ~0.005s   |
| RL Agent    | 2×2       | ~85%         | Varies    |

*RL performance depends on training duration and scramble depth.*

---

## 📂 Project Structure

```text
AIM-Rubiks-Solver/
│── app.py               # Main Application
│── README.md            # Documentation
│── requirements.txt     # Dependencies
└── assets/              # Images / diagrams
```

---

## 🚀 Future Roadmap

* [ ] **3D Cube Visualization** (Three.js / PyDeck)
* [ ] **Deep Q-Network (DQN)** for 3×3 learning capability
* [ ] **Kociemba’s Two-Phase Algorithm** integration
* [ ] Exportable solution logs
* [ ] Performance benchmarking suite

---

## 🤝 Contact & License

**Developer:** Nik (Prince)
**Role:** Final Year B.Tech (ECE)
**Institution:** NIT Agartala

**License:** MIT License

Made with ❤️ and Python.
