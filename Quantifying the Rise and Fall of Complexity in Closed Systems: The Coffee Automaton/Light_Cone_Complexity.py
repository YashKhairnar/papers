import random
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import Counter

# ==========================================
# 1. THE CLASS (Physics + Math)
# ==========================================
class CoffeeAutomaton:
    def __init__(self, N=50):
        self.N = N
        # using numpy for speed and easier plotting
        self.state = np.zeros((N, N), dtype=int)
        self.state[N//2:, :] = 1  # Bottom half is Cream (1)
        
    def get_neighbor_string(self, x, y):
        """Returns 4-neighbor string for LCC calculation."""
        neighbors = ""
        # Up, Right, Down, Left
        dx = [0, 1, 0, -1] 
        dy = [-1, 0, 1, 0]
        
        for i in range(4):
            nx = (x + dx[i]) % self.N
            ny = (y + dy[i]) % self.N
            neighbors += str(self.state[ny, nx])
        return neighbors

    def get_grid_snapshot(self):
        """Returns list of all neighbor patterns."""
        snapshot = []
        for y in range(self.N):
            for x in range(self.N):
                snapshot.append(self.get_neighbor_string(x, y))
        return snapshot

    def step(self):
        """Performs ONE swap (Kawasaki dynamics)."""
        # Pick random cell
        x1, y1 = random.randint(0, self.N - 1), random.randint(0, self.N - 1)
        
        # Pick random neighbor
        dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        dx, dy = random.choice(dirs)
        
        x2 = (x1 + dx) % self.N
        y2 = (y1 + dy) % self.N

        # Swap
        self.state[y1, x1], self.state[y2, x2] = self.state[y2, x2], self.state[y1, x1]

    def shannon_entropy(self, data_list):
        counts = Counter(data_list)
        total = len(data_list)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def joint_entropy(self, list_A, list_B):
        pairs = list(zip(list_A, list_B))
        return self.shannon_entropy(pairs)

    def calculate_metrics(self, tau):
        """Runs the simulation for 'tau' steps to measure LCC."""
        past = self.get_grid_snapshot()
        
        # Micro-Evolution
        for _ in range(tau):
            self.step()
            
        future = self.get_grid_snapshot()
        
        H_past = self.shannon_entropy(past)
        H_future = self.shannon_entropy(future)
        H_joint = self.joint_entropy(past, future)
        
        LCC = H_past + H_future - H_joint
        return LCC, H_past

# ==========================================
# 2. PART 1: GENERATE THE GRAPH
# ==========================================
def run_analysis_experiment():
    print("\n--- Starting Complexity Analysis ---")
    
    # 1. Setup Parameters (The Golden Ratio)
    N = 100
    total_cells = N * N
    tau = int(total_cells * 0.4)    # Measurement Window
    evolution_gap = total_cells * 3 # Mixing Gap
    total_phases = 100               # Resolution

    sim = CoffeeAutomaton(N)
    
    # 2. Storage
    history_phases = []
    history_lcc = []
    history_entropy = []

    # 3. The Loop
    for phase in range(total_phases):
        lcc, entropy = sim.calculate_metrics(tau)
        
        history_phases.append(phase)
        history_lcc.append(lcc)
        history_entropy.append(entropy)
        
        if phase % 10 == 0:
            print(f"Phase {phase}/{total_phases}: Entropy={entropy:.2f}, LCC={lcc:.2f}")
        
        # Mixing Step
        for _ in range(evolution_gap):
            sim.step()

    # 4. Plotting
    print("Analysis complete. Generating Graph...")
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Time (Phases)')
    ax1.set_ylabel('Shannon Entropy', color=color)
    ax1.plot(history_phases, history_entropy, color=color, linewidth=2, label='Entropy')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Light-Cone Complexity', color=color)
    ax2.plot(history_phases, history_lcc, color=color, linewidth=2, linestyle='--', label='Complexity')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'Rise and Fall of Complexity (N={N})')
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig('complexity_graph.png') # Saves the graph
    print("Graph saved as 'complexity_graph.png'")
    # plt.show() # Uncomment to see popup

# ==========================================
# 3. PART 2: GENERATE THE VIDEO
# ==========================================
def run_video_recording():
    print("\n--- Starting Video Recording ---")
    
    N = 100
    sim = CoffeeAutomaton(N)
    
    # Video Settings
    steps_per_frame = 1000  # Speed up physics
    total_frames = 3300     # Length of video
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title("Coffee Automaton")
    ax.axis('off')
    
    # Initial Image
    im = ax.imshow(sim.state, cmap='gray', vmin=0, vmax=1)

    def update(frame):
        # Fast-forward physics
        for _ in range(steps_per_frame):
            sim.step()
        
        im.set_data(sim.state)
        ax.set_title(f"Step: {frame * steps_per_frame}")
        return [im]

    # Create Animation
    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=20, blit=True)
    
    # SAVE VIDEO
    # Note: Requires FFmpeg installed. If it fails, change to .gif and writer='pillow'
    try:
        ani.save('coffee_evolution.mp4', writer='ffmpeg', fps=30)
        print("Video saved as 'coffee_evolution.mp4'")
    except Exception as e:
        print(f"Could not save MP4 (FFmpeg missing?). Trying GIF...")
        ani.save('coffee_evolution.gif', writer='pillow', fps=30)
        print("Video saved as 'coffee_evolution.gif'")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # You can comment out one or the other if you don't need both
    # run_analysis_experiment()
    run_video_recording()