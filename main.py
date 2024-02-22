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
        x = start_point[0]
        height = (math.sqrt(3) / 2) * side_length

        # Adjust points for Y-Z plane drawing
        point1 = start_point
        point2 = [x, start_point[1] + side_length, start_point[2]]  
        point3 = [x, start_point[1] + side_length / 2, start_point[2] - height]  
        triangle_points = [point1, point2, point3, point1]
        for point in triangle_points:
            self.move_to_position(*point)


    def disconnect(self):
        self.arm.disconnect()


def main():
    controller = XArmController('http://192.168.0.203/') 
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
        fixed_x = 546  # Example fixed X value, adjust based on your setup
        side_lengths = [50, 100, 150]  # 
        
        start_y = -100  # 
        start_z = 500  # 

        for length in side_lengths:
            start_point = [fixed_x, start_y, start_z]
            controller.draw_triangle(start_point, length)
            start_y -= length 
            start_z -= (math.sqrt(3) / 2) * length  #
        
    finally:
        controller.disconnect()

if __name__ == '__main__':
    main()
