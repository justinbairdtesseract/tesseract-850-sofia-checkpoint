from xarm.wrapper import XArmAPI
import math
import random

class XArmController:
    def __init__(self, ip_address):
        self.IP_ADDRESS = ip_address
        self.arm = XArmAPI(self.IP_ADDRESS)
        self.setup()
        self.canvas_corners = [
            [546, 205, 835],  # top_left
            [546, -333, 835], # top_right
            [508, 205, 88],   # bottom_left
            [508, -333, 88]   # bottom_right
        ]
        self.calculate_scaling_factors()
        self.off_canvas_distance = 50  # Distance to start away from the canvas on the x-axis

    def setup(self):
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)

    # FUNCTIONS TO CALCULATE X
    def calculate_scaling_factors(self):
        z1, x1 = self.canvas_corners[0][2], self.canvas_corners[0][0]  # Top left corner
        z2, x2 = self.canvas_corners[2][2], self.canvas_corners[2][0]  # Bottom left corner
        self.a = (x2 - x1) / (z2 - z1)
        self.b = x1 - self.a * z1

    def x_from_z(self, z):
        # Calculate x based on z
        return self.a * z + self.b



    # FUNCTIONS TO MOVE TO AND FROM CANVAS
    def approach_canvas(self, x, y, z, speed=100):
        # Move to a position slightly away from the canvas
        approach_x = x - self.off_canvas_distance
        self.arm.set_position(x=approach_x, y=y, z=z, speed=speed, wait=True)

    def retreat_from_canvas(self, x, y, z, speed=100):
        # Move to a position slightly away from the canvas after drawing
        retreat_x = x - self.off_canvas_distance
        self.arm.set_position(x=retreat_x, y=y, z=z, speed=speed, wait=True)

    def move_to_position_with_approach_and_retreat(self, x, y, z, speed=100):
        self.approach_canvas(x, y, z, speed)
        self.arm.set_position(x=x, y=y, z=z, speed=speed, wait=True)
        self.retreat_from_canvas(x, y, z, speed)



    # DRAW SPECIFIC SHAPES
    def draw_canvas_border(self):
        top_left, top_right, bottom_left, bottom_right = self.canvas_corners
        square_points = [top_left, top_right, bottom_right, bottom_left, top_left]
        self.draw_square(square_points)

    def draw_dot(self, x, y, z):
        x_scaled = self.x_from_z(z)
        self.move_to_position_with_approach_and_retreat(x_scaled, y, z, speed=100)

    def draw_random_dots(self, num_dots):
        for _ in range(num_dots):
            random_y = random.uniform(self.canvas_corners[3][1], self.canvas_corners[0][1])
            random_z = random.uniform(self.canvas_corners[3][2], self.canvas_corners[0][2])
            random_x = self.x_from_z(random_z)
            self.draw_dot(random_x, random_y, random_z)

    def draw_line(self, start, end, speed=100):
        self.approach_canvas(*start, speed)
        self.arm.set_position(x=start[0], y=start[1], z=start[2], speed=speed, wait=True)
        self.arm.set_position(x=end[0], y=end[1], z=end[2], speed=speed, wait=True)
        self.retreat_from_canvas(*end, speed)

    def draw_square(self, points, speed=100):
        scaled_points = [[self.x_from_z(point[2]), point[1], point[2]] for point in points]
        for i in range(len(scaled_points)):
            start_point = scaled_points[i]
            end_point = scaled_points[(i + 1) % len(scaled_points)] 
            self.draw_line(start_point, end_point, speed)
            
    def draw_random_lines(self, num_lines):
        for _ in range(num_lines):
                start_z = random.uniform(self.canvas_corners[3][2], self.canvas_corners[0][2])
                end_z = random.uniform(self.canvas_corners[3][2], self.canvas_corners[0][2])
                
                start_x = self.x_from_z(start_z)
                end_x = self.x_from_z(end_z)
                
                # Randomly choose Y within the limits of top left and bottom right corners
                start_y = random.uniform(self.canvas_corners[3][1], self.canvas_corners[0][1])
                end_y = random.uniform(self.canvas_corners[3][1], self.canvas_corners[0][1])
                
                # Approach the canvas before starting the line
                self.approach_canvas(start_x, start_y, start_z)
                
                # Draw the line from start to end point
                print(f"Drawing line from ({start_x}, {start_y}, {start_z}) to ({end_x}, {end_y}, {end_z})")
                self.arm.set_position(x=end_x, y=end_y, z=end_z, speed=100, wait=True)
                
                # Retreat from the canvas after drawing the line
                self.retreat_from_canvas(end_x, end_y, end_z)


    def disconnect(self):
        self.arm.disconnect()

def main():
    controller = XArmController('192.168.0.203')
    try:
        controller.draw_canvas_border()
        
        controller.draw_random_dots(10)
        
        controller.draw_random_lines(10)

    finally:
        controller.disconnect()


if __name__ == '__main__':
    main()
