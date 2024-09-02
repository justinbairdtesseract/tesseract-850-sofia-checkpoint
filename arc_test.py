from xarm.wrapper import XArmAPI
import math
import random
import time

class XArmController:
    def __init__(self, ip_address):
        self.IP_ADDRESS = ip_address
        self.arm = XArmAPI(self.IP_ADDRESS)
        self.setup()
        self.off_canvas_distance = 25  # Distance to start away from the canvas on the x-axis

        # Canvas corners must always stay in this order, top left, top right, bottom left, bottom right
        self.canvas_corners = [
            [546, 205, 835],  # top_left
            [546, -333, 835], # top_right
            [508, 205, 88],   # bottom_left
            [508, -333, 88]   # bottom_right
        ]
        self.calculate_scaling_factors()
        self.calculate_canvas_bounds()

    def setup(self):
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)

    def calculate_scaling_factors(self):
        z1, x1 = self.canvas_corners[0][2], self.canvas_corners[0][0]  # Top left corner
        z2, x2 = self.canvas_corners[2][2], self.canvas_corners[2][0]  # Bottom left corner
        self.a = (x2 - x1) / (z2 - z1)
        self.b = x1 - self.a * z1

    def x_from_z(self, z):
        return self.a * z + self.b

    def calculate_canvas_bounds(self):
        self.y_min = min(self.canvas_corners[0][1], self.canvas_corners[1][1])
        self.y_max = max(self.canvas_corners[0][1], self.canvas_corners[1][1])
        self.z_min = min(self.canvas_corners[2][2], self.canvas_corners[0][2])
        self.z_max = max(self.canvas_corners[2][2], self.canvas_corners[0][2])

    def generate_random_position(self, scale_factor=1.0):
        random_y = random.uniform(
            self.y_min + (self.y_max - self.y_min) * (1 - scale_factor) / 2,
            self.y_max - (self.y_max - self.y_min) * (1 - scale_factor) / 2
        )

        random_z = random.uniform(
            self.z_min + (self.z_max - self.z_min) * (1 - scale_factor) / 2,
            self.z_max - (self.z_max - self.z_min) * (1 - scale_factor) / 2
        )

        random_x = self.x_from_z(random_z)
        return (random_x, random_y, random_z)

    def approach_canvas(self, x, y, z, roll=-180, speed=50):
        approach_x = x - self.off_canvas_distance
        self.arm.set_position(x=approach_x, y=y, z=z, roll=roll, pitch=-90, yaw=0, speed=speed, wait=True)
        self.arm.set_position(x=x, y=y, z=z, roll=roll, pitch=-90, yaw=0, speed=speed, wait=True)

    def retreat_from_canvas(self, x, y, z, speed=50):
        retreat_x = x - self.off_canvas_distance
        self.arm.set_position(x=retreat_x, y=y, z=z, roll=-180, pitch=-90, yaw=0, speed=speed, wait=True)

    def draw_random_arcs(self, num_arcs, speed=200, mvacc=1000, circle_scaling_factor=0.5, wait=True):
        for _ in range(num_arcs):
            # Move to a random starting position on the canvas
            x_start, y_start, z_start = self.generate_random_position(circle_scaling_factor)

            # Approach the canvas at the random starting position
            self.approach_canvas(x_start, y_start, z_start, speed)

            # Generate pose1 and pose2 for the circle within canvas bounds
            x1, y1, z1 = self.generate_random_position(circle_scaling_factor)
            x2, y2, z2 = self.generate_random_position(circle_scaling_factor)

            pose1 = [x1, y1, z1, -180, -90, 0]  # Adjust orientation as needed
            pose2 = [x2, y2, z2, -180, -90, 0]  # Adjust orientation as needed

            print('Drawing arc with positions:')
            print(x_start, y_start, z_start)
            print(pose1)
            print(pose2)

            # Start drawing the arc
            self.arm.move_circle(pose1=pose1, pose2=pose2, percent=100, speed=speed, mvacc=mvacc, wait=False)

            # Wait for 1 second before stopping
            time.sleep(1)

            # Stop the arm mid-arc
            self.arm.set_state(state=4)
            print('state stop')
            time.sleep(.5)
            self.arm.set_state(state=0)
            print('state start')
            # Retreat from the canvas after stopping
            self.retreat_from_canvas(x_start, y_start, z_start)

    # def check_position_and_stop(self):
    #     position = self.arm.get_position()  # Get current position of the arm
    #     x, y, z = position[:3]
        
    #     # Calculate the canvas bounds with a margin of 10 units
    #     x_min = self.x_from_z(self.z_min) - 10
    #     x_max = self.x_from_z(self.z_max) + 10
    #     y_min = self.y_min - 10
    #     y_max = self.y_max + 10
    #     z_min = self.z_min - 10
    #     z_max = self.z_max + 10
        
    #     # Check if the current position is out of bounds
    #     if not (x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max):
    #         print(f"Arm is out of bounds: (x: {x}, y: {y}, z: {z}). Stopping arm.")
    #         self.arm.set_state(state=4)  # Stop the arm
    #         self.checking_position = False  # Stop the checking loop


    def disconnect(self):
        self.arm.disconnect()


# Example usage
controller = XArmController('192.168.0.203')
controller.arm.set_servo_angle(servo_id=5, angle=90, wait=True)
controller.draw_random_arcs(num_arcs=5)
controller.disconnect()
