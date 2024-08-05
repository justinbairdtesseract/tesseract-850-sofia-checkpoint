from xarm.wrapper import XArmAPI
from pythonosc.udp_client import SimpleUDPClient
from pythonosc import udp_client
from pythonosc import dispatcher
from pythonosc import osc_server
import math
import random
import time
import vision 


LINUX_IP = '192.168.0.110' # IP address of laptop 
MAC_IP = '192.168.0.128'
MAXMSP_TO_PORT = 7654
MAXMSP_FROM_PORT = 7655

MOVE_BETWEEN_SPEED = 500


face_detection_thread = vision.start_face_mesh('webcam', True)
# face_detection_thread = vision.start_face_mesh('filepath', True) # False to not display video


# ip_address is for arm
class XArmController:
    def __init__(self, ip_address,  osc_ip=MAC_IP, osc_port=MAXMSP_TO_PORT):
        self.IP_ADDRESS = ip_address
        self.arm = XArmAPI(self.IP_ADDRESS)
        self.setup()
        self.busy = False
        self.fingers_allowed = 4
        self.strokes_since_colour_change = 0

        self.allowed_to_play = True

        self.use_video = False
        self.camera_width = 640
        self.camera_height = 480

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

        self.paint_pressure_range = 5
        self.current_pressure = 0 # 0 full pressure

        self.osc_client = SimpleUDPClient(osc_ip, osc_port)

    def setup(self):
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)


    def pause_movement(self):
        print('started drawing')
        moving = controller.arm.get_is_moving() 
        print('controller is moving? ', moving)

        while controller.arm.get_is_moving():
            print('moving')
            time.sleep(.5)
        print('stopped moving')


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

    def generate_random_position(self, scale_factor=1.0):
        print('using video postion', self.use_video)
        """
        Generates a random position within the canvas boundaries scaled by a specified factor.

        Args:
            scale_factor (float): The factor by which to scale the canvas size for generating a random position. 
                                Default is 1.0 (100% of the canvas).

        Returns:
            A tuple (x, y, z) representing a random position within the scaled canvas bounds.
        """
        # Directly calculate the y and z coordinates within the scaled canvas size
        if self.use_video:
            cx, cy = vision.get_face_center()
            eye_rotation = self.calculate_eye_rotation()
            print('Center point from camera:', cx, cy)
            print('Eye rotation:', eye_rotation)
            random_x, random_y, random_z = self.scale_to_canvas(cx, eye_rotation)
            print('Scaled camera position:', random_x, random_y, random_z)
            random_x, random_y, random_z = 510, 100, 835 # comment out after testing camera 
        else: 
            random_y = random.uniform(
                self.y_min + (self.y_max - self.y_min) * (1 - scale_factor) / 2,
                self.y_max - (self.y_max - self.y_min) * (1 - scale_factor) / 2
            )

            random_z = random.uniform(
                self.z_min + (self.z_max - self.z_min) * (1 - scale_factor) / 2,
                self.z_max - (self.z_max - self.z_min) * (1 - scale_factor) / 2
            )

            random_x = self.x_from_z(random_z) - self.current_pressure # if you subtract getting further away 
        return (random_x, random_y, random_z)


    # VISION FUNCTIONS
    def scale_to_canvas(self, camera_x, eye_rotation):
        rotation_factor = (eye_rotation + math.pi/4) / (math.pi/2)  # Normalize angle to [0, 1] range
        scaled_y = self.y_min + rotation_factor * (self.y_max - self.y_min)

        y_scale_factor = 1.5

        scaled_z = self.z_min + (camera_x / self.camera_width) * (self.z_max - self.z_min) * y_scale_factor


        scaled_x = self.x_from_z(scaled_z)
        return scaled_x, scaled_y, scaled_z


    def calculate_eye_rotation(self):
        """
        Calculate the relative angle of the eyes.

        Returns:
            float: The relative angle of the eyes.
        """
        left_eye, right_eye = vision.get_eye_centers()
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = math.atan2(dy, dx)  # Calculate the angle in radians
        return angle


    # FUNCTIONS TO MOVE TO AND FROM CANVAS, 
    def approach_canvas(self, x, y, z, roll=-180, speed=50):
        # speed is for the move onto canvas, don't change
        # Move to a position slightly away from the canvas, 
        approach_x = x - self.off_canvas_distance
        self.arm.set_position(x=approach_x, y=y, z=z, roll=-180, pitch=-90, yaw=0,  speed=MOVE_BETWEEN_SPEED, wait=True)
        # and then move onto canvas
        self.arm.set_position(x=x, y=y, z=z, roll=-180, pitch=-90, yaw=0, speed=speed, wait=True)
        self.osc_client.send_message("/arm/movement", "approaching")

    def retreat_from_canvas(self, x, y, z, speed=50):
        # Move to a position slightly away from the canvas after drawing
        retreat_x = x - self.off_canvas_distance
        self.arm.set_position(x=retreat_x, y=y, z=z, roll=-180, pitch=-90, yaw=0, speed=speed, wait=True)
        self.osc_client.send_message("/arm/movement", "retreating")
        self.osc_client.send_message("/listen", 1)


    # DRAW SPECIFIC SHAPES
    def draw_canvas_border(self):
        top_left, top_right, bottom_left, bottom_right = self.canvas_corners
        square_points = [top_left, top_right, bottom_right, bottom_left, top_left]
        self.draw_square(square_points)

    def draw_dot(self, x, y, z, speed=200):
        self.approach_canvas(x, y, z, speed)
        self.arm.set_position(x=x, y=y, z=z, roll=-180, pitch=-90, yaw=0, speed=speed, wait=True)
        self.retreat_from_canvas(x, y, z, speed)

    def draw_random_dots(self, num_dots):
        for _ in range(num_dots):
            x, y, z = self.generate_random_position()  # Use the new method to generate a random position
            self.draw_dot(x, y, z)

    def draw_line(self, start, end, speed=100):
        self.approach_canvas(*start, speed)
        self.arm.set_position(x=start[0], y=start[1], z=start[2], roll=-180, pitch=-90, yaw=0, speed=speed, wait=True)
        self.arm.set_position(x=end[0], y=end[1], z=end[2], roll=-180, pitch=-90, yaw=0, speed=speed, wait=True)
        self.retreat_from_canvas(*end, speed)

    def draw_square(self, points, speed=100):
        scaled_points = [[self.x_from_z(point[2]), point[1], point[2]] for point in points]
        for i in range(len(scaled_points)):
            start_point = scaled_points[i]
            end_point = scaled_points[(i + 1) % len(scaled_points)] 
            self.draw_line(start_point, end_point, speed)


    def draw_random_arcs(self, num_arcs, speed=200, mvacc=1000, circle_scaling_factor=0.25, wait=True):
        for _ in range(num_arcs):
            # Move to a random starting position on the canvas
            x_start, y_start, z_start = self.generate_random_position(circle_scaling_factor)

            # Approach the canvas at the random starting position
            self.approach_canvas(x_start, y_start, z_start, speed)

            # Generate pose1 and pose2 for the circle within canvas bounds
            x1, y1, z1 = self.generate_random_position(circle_scaling_factor)
            x2, y2, z2 = self.generate_random_position(circle_scaling_factor)

            radius = math.sqrt((x1 - x_start) ** 2 + (y1 - y_start) ** 2 + (z1 - z_start) ** 2) 
            # Check if the radius is greater than 50
            print('predicted radius: ', radius)

            if radius > 25:
                print(f'Skipping arc with radius {radius} (greater than 50)')
                self.retreat_from_canvas(x_start, y_start, z_start)
                continue

            pose1 = [x1, y1, z1, -180, -90, 0]  # Adjust orientation as needed
            pose2 = [x2, y2, z2, -180, -90, 0]  # Adjust orientation as needed

            print('arc positions')
            print(x_start, y_start, z_start)
            print (pose1)
            print (pose2)

            # Randomly decide the percentage of the circle to draw
            # percent = random.randint(10, 100)  # Adjust range based on desired variability

            # Use the move_circle method to draw the arc
            self.arm.move_circle(pose1=pose1, pose2=pose2, percent=100, speed=speed, mvacc=mvacc, wait=wait)

            self.retreat_from_canvas(x2, y2, z2,)


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




# Arm IP address
controller = XArmController('192.168.0.203')
controller.arm.set_servo_angle(servo_id=5, angle=90, wait=True)
controller.osc_client.send_message("/listen", 1)

        # controller.draw_canvas_border()
        # controller.draw_random_dots(10)
        # controller.fill_canvas(line_spacing=10)
# controller.draw_random_arcs(10)

def interactive(name, *args):
    max_message = list(args)
    if max_message[0] != 'suggested_fingers':
        print(max_message)

    if max_message[0] == 'homing':

        if max_message[1] == 'draw_border':
            controller.draw_canvas_border()
        else: 
            if max_message[1] == 'top_left':
                controller.canvas_corners[0] = [max_message[2], max_message[3], max_message[4]]
            elif max_message[1] == 'top_right':
                controller.canvas_corners[1] = [max_message[2], max_message[3], max_message[4]]
            elif max_message[1] == 'bottom_left':
                controller.canvas_corners[2] = [max_message[2], max_message[3], max_message[4]]
            elif max_message[1] == 'bottom_right':
                controller.canvas_corners[3] = [max_message[2], max_message[3], max_message[4]]
            
            if max_message[5] == 1:
                controller.arm.set_position(x=max_message[2], y=max_message[3], z=max_message[4], roll=-180, pitch=-90, yaw=0,  speed=100, wait=True)
            controller.calculate_scaling_factors()
            controller.calculate_canvas_bounds()
            print('new canvas corners')
            print(controller.canvas_corners)

    if max_message[0] == 'video':
        if max_message[1] == 1:
            controller.use_video = True
        else:
            controller.use_video = False

    if max_message[0] == 'fingers':
        controller.fingers_allowed =  max_message[1]
        print('fingers allowed changed to ',  max_message[1])

    if max_message[0] == 'pressure':
        if max_message[1] <= controller.paint_pressure_range:
            controller.current_pressure = max_message[1]
            print('pressure changed to ',  max_message[1])
        else: 
            print('incorrect pressure value')

    if max_message[0] == 'suggested_fingers':
        suggested_fingers =  max_message[1]
        if controller.strokes_since_colour_change > 4:
            if suggested_fingers > controller.fingers_allowed:
                fingers_to_use =  controller.fingers_allowed
            else:
                fingers_to_use = suggested_fingers
            random_fingers = [1] * fingers_to_use + [0] * (4 - fingers_to_use)
            random.shuffle(random_fingers)
            controller.osc_client.send_message("/fingers", random_fingers)
            controller.strokes_since_colour_change = 0

        # else:
        #     print('to recent too change number of fingers')

    if controller.allowed_to_play:
        if max_message[0] == 'onset':
            controller.osc_client.send_message("/listen", 0)
            if max_message[1] == 'dots':
                print('drawing dots')
                controller.draw_random_dots(1)
            elif max_message[1] == 'lines':
                print('drawing lines')
                x1, y1, z1 = controller.generate_random_position(1)
                time.sleep(.5)
                x2, y2, z2 = controller.generate_random_position(1)
                controller.draw_line([x1, y1, z1], [x2, y2, z2])
            elif  max_message[1] == 'arcs':
                print('drawing arcs')
                controller.draw_random_arcs(1)
            else:
                print('incorrect drawing type')

            print('finished drawing, listening again')
            
            controller.strokes_since_colour_change += 1
    
    else:
        print("In PANIC MODE")
    
    
    if max_message[0] == 'panic':
        if max_message[1] == 1:
            controller.allowed_to_play = True
            controller.osc_client.send_message("/listen", 1)
            print('panic off, ready to play')
        else:
            controller.allowed_to_play = False
            controller.osc_client.send_message("/listen", 0)
            print('PANIC TIME!')

    






# OSC Listening
dispatcher = dispatcher.Dispatcher()
dispatcher.map("/macbook", interactive)

server = osc_server.ThreadingOSCUDPServer((LINUX_IP, MAXMSP_FROM_PORT), dispatcher)
print("Serving on {}".format(server.server_address))
server.serve_forever()

# close server when finished

def on_exit():
    controller.disconnect()
    server.server_close()
    
atexit.register(on_exit())