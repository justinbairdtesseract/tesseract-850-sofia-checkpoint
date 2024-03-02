from xarm.wrapper import XArmAPI
from pythonosc.udp_client import SimpleUDPClient
import math
import random

MAXMSP_IP = '127.0.0.1' # IP address of laptop 
MAXMSP_PORT = 7654

# ADD CANVAS CORNERS
# ADD ARM IP, and make all arguments to Xarm Controller, then make controller in seperate file.. 

class XArmController:
    def __init__(self, ip_address,  osc_ip=MAXMSP_IP, osc_port=MAXMSP_PORT):
        self.IP_ADDRESS = ip_address
        self.arm = XArmAPI(self.IP_ADDRESS)
        self.setup()

        # Canvas corners must always stay in this order, top left, top right, bottom left, bottom right
        self.canvas_corners = [
            [546, 205, 835],  # top_left
            [546, -333, 835], # top_right
            [508, 205, 88],   # bottom_left
            [508, -333, 88]   # bottom_right
        ]
        self.calculate_scaling_factors()
        self.calculate_canvas_bounds()
        self.off_canvas_distance = 25  # Distance to start away from the canvas on the x-axis

        self.osc_client = SimpleUDPClient(osc_ip, osc_port)

    def setup(self):
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)

    # FUNCTIONS TO CALCULATE X and Bounds
    def calculate_scaling_factors(self):
        z1, x1 = self.canvas_corners[0][2], self.canvas_corners[0][0]  # Top left corner
        z2, x2 = self.canvas_corners[2][2], self.canvas_corners[2][0]  # Bottom left corner
        self.a = (x2 - x1) / (z2 - z1)
        self.b = x1 - self.a * z1

    def x_from_z(self, z):
        # Calculate x based on z
        return self.a * z + self.b

    def calculate_canvas_bounds(self):
            self.y_min = min(self.canvas_corners[0][1], self.canvas_corners[1][1])
            self.y_max = max(self.canvas_corners[0][1], self.canvas_corners[1][1])
            self.z_min = min(self.canvas_corners[2][2], self.canvas_corners[0][2])
            self.z_max = max(self.canvas_corners[2][2], self.canvas_corners[0][2])


    # FUNCTIONS TO MOVE TO AND FROM CANVAS
    def approach_canvas(self, x, y, z, speed=50):
        # Move to a position slightly away from the canvas
        approach_x = x - self.off_canvas_distance
        self.arm.set_position(x=approach_x, y=y, z=z, speed=speed, wait=True)
        self.osc_client.send_message("/arm/movement", "approaching")

    def retreat_from_canvas(self, x, y, z, speed=50):
        # Move to a position slightly away from the canvas after drawing
        retreat_x = x - self.off_canvas_distance
        self.arm.set_position(x=retreat_x, y=y, z=z, speed=speed, wait=True)
        self.osc_client.send_message("/arm/movement", "retreating")

    def move_to_position_with_approach_and_retreat(self, x, y, z, speed=50):
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

    def draw_random_arcs(self, quantity):
        for _ in range(quantity):
            start_y = random.uniform(self.y_min, self.y_max)
            start_z = random.uniform(self.z_min, self.z_max)
            start_x = self.x_from_z(start_z)

            end_y = random.uniform(self.y_min, self.y_max)
            end_z = random.uniform(self.z_min, self.z_max)
            end_x = self.x_from_z(end_z)

            # random mid point, will be interesting... 
            mid_y = random.uniform(min(start_y, end_y), max(start_y, end_y))
            mid_z = random.uniform(min(start_z, end_z), max(start_z, end_z))
            mid_x = self.x_from_z(mid_z)

            self.approach_canvas(start_x, start_y, start_z)
            self.arm.set_position(x=start_x, y=start_y, z=start_z, wait=True)
            self.arm.move_arc_lines([start_x, start_y, start_z, mid_x, mid_y, mid_z, end_x, end_y, end_z], radius=0, wait=True)
            self.retreat_from_canvas(end_x, end_y, end_z)

    def fill_canvas(self, line_spacing=10, speed=100):
        start_z = self.z_max
        end_z = self.z_min

        current_y = self.y_min
        while current_y <= self.y_max:
            start = [self.x_from_z(start_z), current_y, start_z]
            end = [self.x_from_z(end_z), current_y, end_z]

            self.draw_line(start, end, speed)
            current_y += line_spacing
            start_z, end_z = end_z, start_z

            
    def disconnect(self):
        self.arm.disconnect()



def main():
    controller = XArmController('192.168.0.203')
    try:
        # controller.draw_canvas_border()
        # controller.draw_random_dots(10)

        ### TEST, DOES THIS MOVE THAT JOINT THAT NEEDS TO BE MOVED?
        controller.arm.set_servo_angle(servo_id=5, angle=90, wait=True)


        ### TEST, fill_canvas
        # controller.fill_canvas(line_spacing=10)


        ### TEST DRAW ARCS
        controller.draw_random_arcs(10)


    finally:
        controller.disconnect()


if __name__ == '__main__':
    main()