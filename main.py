from xarm.wrapper import XArmAPI
import math

class XArmController:
    def __init__(self, config_path):
        self.arm = XArmAPI(self.IP_ADDRESS)
        self.setup()

    def setup(self):
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0) 
        self.arm.set_state(state=0)

    def move_to_position(self, x, y, z, speed=100):
        self.arm.set_position(x=x, y=y, z=z, speed=speed, wait=True)

    def draw_square(self, points):
        for point in points:
            self.move_to_position(*point)

    def draw_circle(self, center, radius, segments=24):
        angle_increment = 360 / segments
        for i in range(segments + 1):
            angle_rad = math.radians(i * angle_increment)
            x = center[0] + radius * math.cos(angle_rad)
            y = center[1] + radius * math.sin(angle_rad)
            z = center[2]
            # Use set_position for all segments, including the first
            self.arm.set_position(x=x, y=y, z=z, speed=100, wait=True)

    def disconnect(self):
        self.arm.disconnect()

def main():
    controller = XArmController('your_xarm_ip')  # Replace with your xArm's IP address
    
    try:
        # Define the corners of the canvas (assuming corrected for tilt)
        top_left = [546, 205, 835]
        top_right = [546, -333, 835]
        bottom_left = [ 508, -333, 88]
        bottom_right = [508, 205, 88]
        
        # Draw square around the canvas
        square_points = [top_left, top_right, bottom_right, bottom_left, top_left]
        controller.draw_square(square_points)
        
        # Define center of the canvas
        center_x = (top_left[0] + bottom_right[0]) / 2
        center_y = (top_left[1] + bottom_right[1]) / 2
        center_z = (top_left[2] + bottom_right[2]) / 2  

        # Draw three  circles
        for radius in [50, 100, 150]:  
            controller.draw_circle([center_x, center_y, center_z], radius)
        
    finally:
        # Ensure the arm is disconnected properly
        controller.disconnect()

if __name__ == '__main__':
    main()
