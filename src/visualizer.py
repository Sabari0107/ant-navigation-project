import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from rover import AntInspiredRover

class NavigationAnimator:
    """
    Creates animated visualizations of rover navigation.
    """
    
    def __init__(self, rover, sun_azimuth=90):
        """
        Initialize animator with rover data.
        
        Parameters:
        -----------
        rover : AntInspiredRover
            Rover object with complete path history
        sun_azimuth : float
            Sun position in degrees
        """
        self.rover = rover
        self.sun_azimuth = sun_azimuth
        self.path = np.array(rover.path_history)
        
    def create_animation(self, save_path='results/animation.gif', fps=10):
        """
        Create animated GIF of rover navigation.
        
        Parameters:
        -----------
        save_path : str
            Path to save animation
        fps : int
            Frames per second
        """
        # Setup figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Set plot limits with padding
        x_min, x_max = self.path[:, 0].min(), self.path[:, 0].max()
        y_min, y_max = self.path[:, 1].min(), self.path[:, 1].max()
        padding = max(x_max - x_min, y_max - y_min) * 0.2
        
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)
        
        # Initialize plot elements
        path_line, = ax.plot([], [], 'b-', linewidth=2, label='Path')
        rover_marker, = ax.plot([], [], 'ro', markersize=12, label='Rover')
        home_marker, = ax.plot([0], [0], 'g*', markersize=20, label='Home')
        
        # Home vector arrow - initialize as None
        home_arrow = None
        
        # Sun direction
        sun_angle = np.deg2rad(self.sun_azimuth)
        sun_vec = np.array([np.cos(sun_angle), np.sin(sun_angle)]) * (padding * 0.8)
        sun_x, sun_y = (x_max + x_min) / 2, (y_max + y_min) / 2
        ax.arrow(sun_x, sun_y, sun_vec[0], sun_vec[1],
                head_width=2, head_length=1.5, fc='yellow', 
                ec='orange', linewidth=3, label='Sun')
        
        # Text annotations
        step_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                           verticalalignment='top', fontsize=12,
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        distance_text = ax.text(0.02, 0.88, '', transform=ax.transAxes,
                               verticalalignment='top', fontsize=12,
                               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        ax.set_xlabel('X Position (meters)', fontsize=12)
        ax.set_ylabel('Y Position (meters)', fontsize=12)
        ax.set_title('Ant-Inspired Rover Navigation', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        def init():
            """Initialize animation."""
            path_line.set_data([], [])
            rover_marker.set_data([], [])
            step_text.set_text('')
            distance_text.set_text('')
            return path_line, rover_marker, step_text, distance_text
        
        def animate(frame):
            """Update animation for each frame."""
            nonlocal home_arrow  # Access outer scope variable
            
            # Update path line
            path_line.set_data(self.path[:frame+1, 0], self.path[:frame+1, 1])
            
            # Update rover position
            rover_marker.set_data([self.path[frame, 0]], [self.path[frame, 1]])
            
            # Update home vector arrow
            if frame < len(self.rover.home_vector_history):
                hv = self.rover.home_vector_history[frame]
                pos = self.path[frame]
                
                # Remove old arrow if it exists
                if home_arrow is not None:
                    try:
                        home_arrow.remove()
                    except:
                        pass  # Ignore if already removed
                
                # Draw new arrow
                if np.linalg.norm(hv) > 0.1:
                    scale = 0.5  # Scale factor for visualization
                    home_arrow = ax.arrow(pos[0], pos[1], 
                                        hv[0]*scale, hv[1]*scale,
                                        head_width=1, head_length=0.5,
                                        fc='red', ec='red', 
                                        linewidth=2, alpha=0.6)
            
            # Update text
            step_text.set_text(f'Step: {frame + 1} / {len(self.path)}')
            
            home_dist = np.linalg.norm(self.path[frame] - self.rover.home)
            distance_text.set_text(f'Distance to Home: {home_dist:.1f}m')
            
            return path_line, rover_marker, step_text, distance_text
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, init_func=init,
            frames=len(self.path),
            interval=1000/fps,  # milliseconds per frame
            blit=False,
            repeat=True
        )
        
        # Save animation
        print(f"Creating animation... This may take a minute.")
        try:
            anim.save(save_path, writer='pillow', fps=fps, dpi=100)
            print(f"✓ Animation saved to: {save_path}")
        except Exception as e:
            print(f"✗ Error saving animation: {e}")
            print("  Try installing pillow: pip install pillow")
        
        plt.close()


def create_animation_from_simulation():
    """
    Helper function to create animation from a simulation.
    Run this after running simulation.py
    """
    from simulation import NavigationSimulation
    
    # Run simulation
    sim = NavigationSimulation()
    
    foraging_moves = [
        (45, 15),
        (-30, 20),
        (60, 10),
        (-45, 18),
    ]
    
    rover = sim.run_single_simulation(
        foraging_moves=foraging_moves,
        sensor_noise=0.05,
        simulation_name="Animation Demo"
    )
    
    # Create animation
    animator = NavigationAnimator(rover, sun_azimuth=90)
    animator.create_animation('results/navigation_animation.gif', fps=8)
    
    print("\n✓ Animation created successfully!")


if __name__ == "__main__":
    create_animation_from_simulation()