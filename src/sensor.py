import numpy as np

class PolarizedLightSensor:
    """
    Simulates a polarized light sensor array that can determine sun direction.
    Desert ants have specialized photoreceptors in their compound eyes that
    detect the polarization pattern of skylight.
    """
    
    def __init__(self, noise_level=0.05):
        """
        Initialize the polarized light sensor.
        
        Parameters:
        -----------
        noise_level : float
            Amount of random noise in sensor readings (0.0 to 1.0)
            Simulates environmental factors and sensor imperfections
        """
        self.noise_level = noise_level
        self.sun_azimuth = 0.0
        
    def set_sun_position(self, azimuth_degrees):
        """
        Set the sun's position in the sky.
        
        Parameters:
        -----------
        azimuth_degrees : float
            Sun direction in degrees (0° = East, 90° = North, 180° = West)
        """
        self.sun_azimuth = np.deg2rad(azimuth_degrees)
        
    def get_sun_direction(self):
        """
        Get the current sun direction with simulated sensor noise.
        
        Returns:
        --------
        float : Sun azimuth in radians, with added noise
        """
        noise = np.random.normal(0, self.noise_level)
        return self.sun_azimuth + noise
    
    def get_sun_vector(self):
        """
        Get sun direction as a unit vector [x, y].
        
        Returns:
        --------
        numpy.array : 2D unit vector pointing toward the sun
        """
        angle = self.get_sun_direction()
        return np.array([np.cos(angle), np.sin(angle)])
    
    def calibrate(self, known_sun_position):
        """
        Calibrate the sensor with a known sun position.
        
        Parameters:
        -----------
        known_sun_position : float
            Known sun azimuth in degrees
        """
        self.set_sun_position(known_sun_position)
        print(f"Sensor calibrated: Sun at {known_sun_position}° azimuth")