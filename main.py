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

    def draw_triangle(self, start_point, side_length):
        # Calculate the height of the equilateral triangle
        height = (math.sqrt(3) / 2) * side_length
        
        # Calculate points
        point1 = start_point
        point2 = [start_point[0] + side_length, start_point[1], start_point[2]]
        point3 = [start_point[0] + side_length / 2, start_point[1] + height, start_point[2]]
        
        # Draw the triangle
        self.move_to_position(*point1)
        self.move_to_position(*point2)
        self.move_to_position(*point3)
        self.move_to_position(*point1)  # Complete the triangle


    def disconnect(self):
        self.arm.disconnect()


def main():
    controller = XArmController('your_xarm_ip')  # Replace with your xArm's IP address
    
    try:
        # Define the corners of the canvas (assuming corrected for tilt)
        top_left = [546, 205, 835]
        top_right = [546, -333, 835]
        bottom_left = [ 508, 205, 88]
        bottom_right = [508, -333, 88]
        
        # Draw square around the canvas
        square_points = [top_left, top_right, bottom_right, bottom_left, top_left]
        controller.draw_square(square_points)

        
        # Canvas dimensions
        canvas_width = abs(546 - 508)
        canvas_height = abs(205 + 333)
        center_x = 508 + canvas_width / 2
        center_y = -333 + canvas_height / 2
        center_z = (835 + 88) / 2 
        
        # Draw triagnle
        side_lengths = [50, 100, 150]  # Example side lengths
        starting_points = []
        for length in side_lengths:
            start_x = center_x - length / 2
            start_y = center_y - ((math.sqrt(3) / 2) * length) / 2
            start_z = center_z
            starting_points.append([start_x, start_y, start_z])

        for i, start_point in enumerate(starting_points):
            controller.draw_triangle(start_point, side_lengths[i])
        
    finally:
        # Ensure the arm is disconnected properly
        controller.disconnect()

if __name__ == '__main__':
    main()
