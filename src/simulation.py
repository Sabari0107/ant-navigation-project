import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from rover import AntInspiredRover

class NavigationSimulation:
    """
    Manages and runs navigation simulations with visualization.
    """
    
    def __init__(self):
        """Initialize simulation parameters."""
        # Simulation configuration
        self.sun_azimuth = 90  # Sun at North (90 degrees)
        self.home_position = (0, 0)  # Origin as home/nest
        
    def run_single_simulation(self, foraging_moves, sensor_noise=0.05, 
                            simulation_name="Simulation"):
        """
        Run a single navigation simulation.
        
        Parameters:
        -----------
        foraging_moves : list of tuples
            List of (turn_angle, distance) movements during foraging
        sensor_noise : float
            Noise level for polarized light sensor
        simulation_name : str
            Name for this simulation run
            
        Returns:
        --------
        AntInspiredRover : The rover object with complete path history
        """
        print(f"\n{'='*60}")
        print(f"Starting: {simulation_name}")
        print(f"{'='*60}")
        
        # Create rover at home position
        rover = AntInspiredRover(
            start_position=self.home_position,
            sensor_noise=sensor_noise
        )
        
        # Configure sun position in sensor
        rover.sensor.set_sun_position(self.sun_azimuth)
        print(f"Sun position set to: {self.sun_azimuth}° (North)")
        
        # Phase 1: Foraging (random walk away from home)
        rover.execute_foraging_run(foraging_moves)
        
        # Phase 2: Homing (direct return using path integration)
        steps_to_home = rover.execute_return_home(step_size=1.0)
        
        # Calculate performance metrics
        total_distance = rover.distance_traveled
        straight_line = np.linalg.norm(rover.position - rover.home)
        
        print(f"\n--- Performance Metrics ---")
        print(f"Total distance traveled: {total_distance:.2f}m")
        print(f"Straight-line distance: {straight_line:.2f}m")
        print(f"Homing accuracy: {straight_line:.2f}m from home")
        print(f"Steps to return: {steps_to_home}")
        
        return rover
    
    def visualize_simulation(self, rover, save_path='results/simulation.png'):
        """
        Create visualization of the rover's path.
        
        Parameters:
        -----------
        rover : AntInspiredRover
            Rover object with path history
        save_path : str
            Path to save the figure
        """
        # Extract path data
        path = np.array(rover.path_history)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Subplot 1: 2D Path Visualization
        ax1.plot(path[:, 0], path[:, 1], 'b-', linewidth=2, label='Rover Path')
        ax1.plot(path[0, 0], path[0, 1], 'go', markersize=15, 
                label='Start/Home', zorder=5)
        ax1.plot(path[-1, 0], path[-1, 1], 'ro', markersize=12, 
                label='Final Position', zorder=5)
        
        # Draw home vector at several points
        sample_indices = np.linspace(0, len(rover.home_vector_history)-1, 8, dtype=int)
        for idx in sample_indices:
            pos = rover.path_history[idx]
            hv = rover.home_vector_history[idx]
            # Draw arrow showing home vector at this point
            ax1.arrow(pos[0], pos[1], hv[0]*0.3, hv[1]*0.3,
                     head_width=1, head_length=0.5, fc='red', 
                     ec='red', alpha=0.3, linewidth=1.5)
        
        # Draw sun direction indicator
        sun_angle = np.deg2rad(self.sun_azimuth)
        sun_vec = np.array([np.cos(sun_angle), np.sin(sun_angle)]) * 20
        ax1.arrow(30, 30, sun_vec[0], sun_vec[1], 
                 head_width=2, head_length=1, fc='yellow', 
                 ec='orange', linewidth=2, label='Sun Direction')
        
        ax1.set_xlabel('X Position (meters)', fontsize=12)
        ax1.set_ylabel('Y Position (meters)', fontsize=12)
        ax1.set_title('Rover Navigation Path\n(Red arrows = Home Vector)', 
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # Subplot 2: Distance from Home over Time
        home_distances = [np.linalg.norm(pos - rover.home) 
                         for pos in rover.path_history]
        
        ax2.plot(home_distances, 'b-', linewidth=2)
        ax2.axhline(y=0.5, color='r', linestyle='--', 
                   label='Homing Threshold (0.5m)')
        ax2.set_xlabel('Step Number', fontsize=12)
        ax2.set_ylabel('Distance from Home (meters)', fontsize=12)
        ax2.set_title('Distance from Home Over Time', 
                     fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")
        plt.show()
    
    def compare_simulations(self, noise_levels=[0.01, 0.05, 0.1, 0.2]):
        """
        Run multiple simulations with different sensor noise levels.
        
        Parameters:
        -----------
        noise_levels : list
            Different noise levels to test
        """
        # Define standard foraging pattern
        foraging_moves = [
            (45, 15),   # Turn 45°, move 15m
            (-30, 20),  # Turn -30°, move 20m
            (60, 10),   # Turn 60°, move 10m
            (-45, 18),  # Turn -45°, move 18m
        ]
        
        results = []
        
        print("\n" + "="*60)
        print("COMPARATIVE ANALYSIS: Effect of Sensor Noise")
        print("="*60)
        
        for noise in noise_levels:
            rover = self.run_single_simulation(
                foraging_moves=foraging_moves,
                sensor_noise=noise,
                simulation_name=f"Sensor Noise = {noise}"
            )
            
            # Store results
            results.append({
                'noise': noise,
                'final_error': np.linalg.norm(rover.position - rover.home),
                'total_distance': rover.distance_traveled
            })
        
        # Visualize comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        noises = [r['noise'] for r in results]
        errors = [r['final_error'] for r in results]
        distances = [r['total_distance'] for r in results]
        
        # Error vs Noise
        ax1.bar(range(len(noises)), errors, color='coral', alpha=0.7)
        ax1.set_xticks(range(len(noises)))
        ax1.set_xticklabels([f"{n:.2f}" for n in noises])
        ax1.set_xlabel('Sensor Noise Level', fontsize=12)
        ax1.set_ylabel('Homing Error (meters)', fontsize=12)
        ax1.set_title('Effect of Sensor Noise on Homing Accuracy', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Distance vs Noise
        ax2.bar(range(len(noises)), distances, color='skyblue', alpha=0.7)
        ax2.set_xticks(range(len(noises)))
        ax2.set_xticklabels([f"{n:.2f}" for n in noises])
        ax2.set_xlabel('Sensor Noise Level', fontsize=12)
        ax2.set_ylabel('Total Distance Traveled (meters)', fontsize=12)
        ax2.set_title('Effect of Sensor Noise on Path Length', 
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('results/comparison.png', dpi=150, bbox_inches='tight')
        print("\nComparison plot saved to: results/comparison.png")
        plt.show()


def main():
    """
    Main function to run simulations.
    """
    # Create results directory if it doesn't exist
    import os
    os.makedirs('results', exist_ok=True)
    
    # Initialize simulation
    sim = NavigationSimulation()
    
    # Scenario 1: Simple foraging pattern
    print("\n" + "#"*60)
    print("# SCENARIO 1: Simple Foraging Pattern")
    print("#"*60)
    
    simple_moves = [
        (30, 20),   # Turn right 30°, move 20m
        (-45, 15),  # Turn left 45°, move 15m
        (60, 25),   # Turn right 60°, move 25m
    ]
    
    rover1 = sim.run_single_simulation(
        foraging_moves=simple_moves,
        sensor_noise=0.05,
        simulation_name="Simple Pattern"
    )
    sim.visualize_simulation(rover1, 'results/simple_pattern.png')
    
    # Scenario 2: Complex foraging pattern
    print("\n" + "#"*60)
    print("# SCENARIO 2: Complex Foraging Pattern")
    print("#"*60)
    
    complex_moves = [
        (45, 15),
        (-30, 20),
        (60, 10),
        (-45, 18),
        (90, 12),
        (-60, 22),
    ]
    
    rover2 = sim.run_single_simulation(
        foraging_moves=complex_moves,
        sensor_noise=0.05,
        simulation_name="Complex Pattern"
    )
    sim.visualize_simulation(rover2, 'results/complex_pattern.png')
    
    # Scenario 3: Comparative analysis
    print("\n" + "#"*60)
    print("# SCENARIO 3: Sensor Noise Analysis")
    print("#"*60)
    
    sim.compare_simulations(noise_levels=[0.01, 0.05, 0.1, 0.2])
    
    print("\n" + "="*60)
    print("ALL SIMULATIONS COMPLETE!")
    print("Check the 'results/' folder for output files.")
    print("="*60)


if __name__ == "__main__":
    main()