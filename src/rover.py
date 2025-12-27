import numpy as np
from sensor import PolarizedLightSensor

class AntInspiredRover:
    """
    A rover that navigates using path integration, inspired by desert ants.
    
    Key behaviors modeled:
    1. Path Integration - maintaining a home vector
    2. Polarized light compass - using sun position for orientation
    3. Odometry - tracking distance traveled
    """
    
    def __init__(self, start_position, sensor_noise=0.05):
        """
        Initialize the rover at a starting position.
        
        Parameters:
        -----------
        start_position : tuple or list
            Initial (x, y) coordinates in meters
        sensor_noise : float
            Noise level for the polarized light sensor
        """
        # Rover's current position in 2D space (x, y)
        self.position = np.array(start_position, dtype=float)
        
        # Home position - where the rover starts (like ant's nest)
        self.home = np.array(start_position, dtype=float)
        
        # Home vector - direct line from current position to home
        # This is the key to path integration navigation
        self.home_vector = np.array([0.0, 0.0])
        
        # Rover's current heading in radians (0 = East, π/2 = North)
        self.heading = 0.0
        
        # Total distance traveled (odometry)
        self.distance_traveled = 0.0
        
        # Install polarized light sensor
        self.sensor = PolarizedLightSensor(noise_level=sensor_noise)
        
        # Movement parameters
        self.speed = 1.0  # meters per time step
        
        # History tracking for visualization
        self.path_history = [self.position.copy()]  # Record all positions
        self.home_vector_history = [self.home_vector.copy()]  # Record home vectors
        
    def move_forward(self, distance):
        """
        Move the rover forward by a specified distance in its current heading.
        
        Parameters:
        -----------
        distance : float
            Distance to move in meters
            
        This simulates the rover's motor commands and updates:
        1. Position based on current heading
        2. Home vector (path integration)
        3. Odometry (distance counter)
        """
        # Calculate movement vector based on heading
        # heading = 0° means move East (+x direction)
        # heading = 90° means move North (+y direction)
        movement = np.array([
            distance * np.cos(self.heading),  # x-component
            distance * np.sin(self.heading)   # y-component
        ])
        
        # Update position
        self.position += movement
        
        # Update home vector (subtract movement because home is getting farther)
        # This is the core of path integration: tracking displacement from home
        self.home_vector -= movement
        
        # Update odometry
        self.distance_traveled += distance
        
        # Record history for visualization
        self.path_history.append(self.position.copy())
        self.home_vector_history.append(self.home_vector.copy())
        
    def turn(self, angle_degrees):
        """
        Turn the rover by a specified angle.
        
        Parameters:
        -----------
        angle_degrees : float
            Angle to turn in degrees (positive = counterclockwise)
        """
        # Convert degrees to radians and update heading
        self.heading += np.deg2rad(angle_degrees)
        
        # Normalize heading to stay within [0, 2π) range
        self.heading = self.heading % (2 * np.pi)
        
    def align_to_sun(self, target_angle_to_sun=0):
        """
        Orient the rover relative to the sun using the polarized light sensor.
        
        Parameters:
        -----------
        target_angle_to_sun : float
            Desired angle relative to sun in degrees
            0° = facing sun, 90° = sun on right, -90° = sun on left
            
        Desert ants use this to maintain consistent direction during foraging.
        """
        # Get current sun direction from sensor
        sun_direction = self.sensor.get_sun_direction()
        
        # Calculate desired heading based on sun and target angle
        target_heading = sun_direction + np.deg2rad(target_angle_to_sun)
        
        # Calculate turn needed
        turn_needed = target_heading - self.heading
        
        # Normalize turn to [-π, π] range (shortest rotation)
        turn_needed = np.arctan2(np.sin(turn_needed), np.cos(turn_needed))
        
        # Execute turn
        self.heading = target_heading
        
    def get_home_direction(self):
        """
        Calculate the direction to home based on the home vector.
        
        Returns:
        --------
        float : Angle to home in radians
        
        This uses the accumulated home vector from path integration.
        """
        if np.linalg.norm(self.home_vector) < 0.01:
            # Already at home
            return self.heading
        
        # Calculate angle of home vector
        # arctan2 gives angle from x-axis to the vector
        return np.arctan2(self.home_vector[1], self.home_vector[0])
    
    def get_home_distance(self):
        """
        Calculate straight-line distance to home.
        
        Returns:
        --------
        float : Distance to home in meters
        """
        return np.linalg.norm(self.home_vector)
    
    def navigate_home(self):
        """
        Orient the rover to face directly toward home.
        This is called when the ant decides to return to the nest.
        """
        home_direction = self.get_home_direction()
        self.heading = home_direction
        
    def execute_foraging_run(self, moves):
        """
        Execute a series of movements during foraging phase.
        
        Parameters:
        -----------
        moves : list of tuples
            List of (angle, distance) movements
            Example: [(45, 10), (-30, 15)] means:
            - Turn 45° and move 10 meters
            - Turn -30° and move 15 meters
        """
        print("\n=== Foraging Phase ===")
        for i, (angle, distance) in enumerate(moves):
            self.turn(angle)
            self.move_forward(distance)
            print(f"Move {i+1}: Turn {angle}°, move {distance}m "
                  f"→ Position: ({self.position[0]:.1f}, {self.position[1]:.1f})")
        
        final_distance = self.get_home_distance()
        print(f"\nForaging complete. Distance from home: {final_distance:.2f}m")
        
    def execute_return_home(self, step_size=1.0):
        """
        Return home using path integration (home vector).
        
        Parameters:
        -----------
        step_size : float
            Distance to move in each step (meters)
            
        Returns:
        --------
        int : Number of steps taken to reach home
        """
        print("\n=== Homing Phase ===")
        steps = 0
        
        # Navigate until close to home (within 0.5 meters)
        while self.get_home_distance() > 0.5:
            # Orient toward home
            self.navigate_home()
            
            # Move forward one step
            move_distance = min(step_size, self.get_home_distance())
            self.move_forward(move_distance)
            
            steps += 1
            
            if steps % 10 == 0:  # Print progress every 10 steps
                print(f"Step {steps}: Distance to home: {self.get_home_distance():.2f}m")
        
        print(f"\nHome reached in {steps} steps!")
        print(f"Final position: ({self.position[0]:.2f}, {self.position[1]:.2f})")
        print(f"Home position: ({self.home[0]:.2f}, {self.home[1]:.2f})")
        
        return steps
    
    def get_path_efficiency(self):
        """
        Calculate navigation efficiency.
        
        Returns:
        --------
        float : Efficiency ratio (0 to 1)
                1.0 = perfectly straight path
                <1.0 = less efficient (more winding)
        """
        straight_line = np.linalg.norm(self.position - self.home)
        if self.distance_traveled == 0:
            return 1.0
        return straight_line / self.distance_traveled