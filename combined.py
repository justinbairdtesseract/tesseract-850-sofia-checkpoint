from xarm.wrapper import XArmAPI
from pythonosc.udp_client import SimpleUDPClient
from pythonosc import udp_client
from pythonosc import dispatcher
from pythonosc import osc_server
import math
import random
import time
import vision
import atexit
import sys
import os
import re
import xml.etree.ElementTree as ET
import numpy as np

LINUX_IP = '192.168.0.110'  # IP address of Linux laptop
MAC_IP = '192.168.0.128'    # IP address of Mac (where Max is running)
MAXMSP_TO_PORT = 7654
MAXMSP_FROM_PORT = 7655

MOVE_BETWEEN_SPEED = 400
STROKE_SPEED = 50

face_detection_thread = vision.start_face_mesh('webcam', True)
# face_detection_thread = vision.start_face_mesh('filepath', True) # if you want to load from file

class XArmController:
    def __init__(self, ip_address, osc_ip=MAC_IP, osc_port=MAXMSP_TO_PORT):
        self.IP_ADDRESS = ip_address
        self.arm = XArmAPI(self.IP_ADDRESS)
        self.setup()

        # States
        self.busy = False
        self.fingers_allowed = 4
        self.strokes_since_colour_change = 0
        self.allowed_to_play = True

        # Vision usage
        self.use_video = False
        self.camera_width = 640
        self.camera_height = 480

        # Canvas corners must always stay in this order: top_left, top_right, bottom_left, bottom_right
        self.canvas_corners = [
            [400,  200, 780],  # top_left
            [400, -200, 780],  # top_right
            [400,  200, 100],  # bottom_left
            [400, -200, 100]   # bottom_right
        ]
        self.calculate_scaling_factors()
        self.calculate_canvas_bounds()

        self.off_canvas_distance = 25  # Distance to start away from the canvas (X direction)
        self.paint_pressure_range = 5
        self.current_pressure = 0  # 0 means full pressure
        self.osc_client = SimpleUDPClient(osc_ip, osc_port)

        # SVG-related
        self.svg_strokes = []  # Will hold path strings
        self.stroke_index = 0  # Which stroke we’re on
        # Default SVG size (in case not in file):
        self.svg_width = 600
        self.svg_height = 600
        self.safety_margin = 10  # to keep from going out of bounds
        self.bezier_steps = 15   # for smooth curves
        
        

    def setup(self):
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)

    def disconnect(self):
        self.arm.disconnect()

    def pause_movement(self):
        """Block until the arm finishes moving."""
        while self.arm.get_is_moving():
            time.sleep(0.5)

    def calculate_scaling_factors(self):
        z1, x1 = self.canvas_corners[0][2], self.canvas_corners[0][0]  # top_left
        z2, x2 = self.canvas_corners[2][2], self.canvas_corners[2][0]  # bottom_left
        self.a = (x2 - x1) / (z2 - z1)
        self.b = x1 - self.a * z1

    def x_from_z(self, z):
        """Given z, figure out the corresponding x for a flat plane on the canvas."""
        return self.a * z + self.b

    def calculate_canvas_bounds(self):
        self.y_min = min(self.canvas_corners[0][1], self.canvas_corners[1][1])
        self.y_max = max(self.canvas_corners[0][1], self.canvas_corners[1][1])
        self.z_min = min(self.canvas_corners[2][2], self.canvas_corners[0][2])
        self.z_max = max(self.canvas_corners[2][2], self.canvas_corners[0][2])

    # --------------
    # VISION HELPERS
    # --------------
    def calculate_eye_rotation(self):
        left_eye, right_eye = vision.get_eye_centers()
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = math.atan2(dy, dx)
        return angle

    def scale_to_canvas(self, camera_x, camera_y, eye_rotation):
        rotation_factor = (eye_rotation + math.pi/4) / (math.pi/2)  # normalize angle to [0,1]
        scaled_z = self.z_min + rotation_factor * (self.z_max - self.z_min)
        scaled_y = self.y_min + (camera_x / self.camera_width) * (self.y_max - self.y_min)
        scaled_x = self.x_from_z(scaled_z)
        return scaled_x, scaled_y, scaled_z

    # --------------
    # RANDOM DRAWING
    # --------------
    def generate_random_position(self, scale_factor=1.0):
        """Generate random position in the canvas region (optionally scaled)."""
        if self.use_video:
            cx, cy = vision.get_face_center()
            eye_rotation = self.calculate_eye_rotation()
            random_x, random_y, random_z = self.scale_to_canvas(cx, cy, eye_rotation)
        else:
            random_y = random.uniform(
                self.y_min + (self.y_max - self.y_min) * (1 - scale_factor) / 2,
                self.y_max - (self.y_max - self.y_min) * (1 - scale_factor) / 2
            )
            random_z = random.uniform(
                self.z_min + (self.z_max - self.z_min) * (1 - scale_factor) / 2,
                self.z_max - (self.z_max - self.z_min) * (1 - scale_factor) / 2
            )
            random_x = self.x_from_z(random_z) - self.current_pressure
        return (random_x, random_y, random_z)

    # --------------
    # MOVE ON/OFF CANVAS
    # --------------
    def approach_canvas(self, x, y, z, roll=-180, speed=50):
        """Approach the canvas from x - off_canvas_distance, then move onto it."""
        approach_x = x - self.off_canvas_distance
        self.arm.set_position(x=approach_x, y=y, z=z, roll=roll, pitch=-90, yaw=0,
                              speed=MOVE_BETWEEN_SPEED, wait=True)
        # Then move onto canvas
        self.arm.set_position(x=x, y=y, z=z, roll=roll, pitch=-90, yaw=0,
                              speed=speed, wait=True)
        self.osc_client.send_message("/arm/movement", "approaching")

    def retreat_from_canvas(self, x, y, z, speed=50, lift=5):
        # 1) lift a little
        self.arm.set_position(x=x, y=y, z=z + lift,
                            roll=-180, pitch=-90, yaw=0, speed=speed, wait=True)
        # 2) slide backwards in X, off the canvas
        retreat_x = x - self.off_canvas_distance
        self.arm.set_position(x=retreat_x, y=y, z=z + lift,
                            roll=-180, pitch=-90, yaw=0, speed=speed, wait=True)
        self.osc_client.send_message("/arm/movement", "retreating")
        self.osc_client.send_message("/listen", 1)

    # --------------
    # BASIC SHAPES
    # --------------
    def draw_dot(self, x, y, z, speed=200):
        self.approach_canvas(x, y, z, speed)
        self.arm.set_position(x=x, y=y, z=z, roll=-180, pitch=-90, yaw=0, speed=speed, wait=True)
        self.retreat_from_canvas(x, y, z, speed)

    def draw_random_dots(self, num_dots):
        for _ in range(num_dots):
            x, y, z = self.generate_random_position()
            self.draw_dot(x, y, z)

    def draw_line(self, start, end, speed=100):
        self.approach_canvas(*start, speed=speed)
        self.arm.set_position(x=start[0], y=start[1], z=start[2],
                              roll=-180, pitch=-90, yaw=0, speed=speed, wait=True)
        self.arm.set_position(x=end[0], y=end[1], z=end[2],
                              roll=-180, pitch=-90, yaw=0, speed=speed, wait=True)
        self.retreat_from_canvas(*end, speed=speed)

    def draw_random_arcs(self, num_arcs, speed=200, mvacc=1000, circle_scaling_factor=0.25, wait=True):
        for _ in range(num_arcs):
            x_start, y_start, z_start = self.generate_random_position(circle_scaling_factor)
            self.approach_canvas(x_start, y_start, z_start, speed)
            x1, y1, z1 = self.generate_random_position(circle_scaling_factor)
            x2, y2, z2 = self.generate_random_position(circle_scaling_factor)

            radius = math.sqrt((x1 - x_start)**2 + (y1 - y_start)**2 + (z1 - z_start)**2)
            # If radius is too large, skip
            if radius > 25:
                self.retreat_from_canvas(x_start, y_start, z_start)
                continue

            pose1 = [x1, y1, z1, -180, -90, 0]
            pose2 = [x2, y2, z2, -180, -90, 0]
            self.arm.move_circle(pose1=pose1, pose2=pose2, percent=100,
                                 speed=speed, mvacc=mvacc, wait=wait)
            self.retreat_from_canvas(x2, y2, z2)

    # --------------
    # SVG SUPPORT
    # --------------
    def parse_svg_file(self, svg_file_path):
        """
        Reads an SVG file, attempts to find all <path>, <rect>, <circle>, <ellipse>, <line>,
        <polyline>, <polygon> and returns them as a list of path-data strings.
        """
        def rect_to_path(rect):
            try:
                x = float(rect.attrib.get('x', 0))
                y = float(rect.attrib.get('y', 0))
                w = float(rect.attrib.get('width', 0))
                h = float(rect.attrib.get('height', 0))
                return f"M{x},{y} h{w} v{h} h{-w} Z"
            except:
                return None

        def circle_to_path(circle):
            try:
                cx = float(circle.attrib.get('cx', 0))
                cy = float(circle.attrib.get('cy', 0))
                r = float(circle.attrib.get('r', 0))
                c = r * 0.5522848
                # approximate circle with 4 cubic beziers
                return (
                  f"M{cx},{cy-r} "
                  f"C{cx+c},{cy-r} {cx+r},{cy-c} {cx+r},{cy} "
                  f"C{cx+r},{cy+c} {cx+c},{cy+r} {cx},{cy+r} "
                  f"C{cx-c},{cy+r} {cx-r},{cy+c} {cx-r},{cy} "
                  f"C{cx-r},{cy-c} {cx-c},{cy-r} {cx},{cy-r} Z"
                )
            except:
                return None

        def ellipse_to_path(ellipse):
            try:
                cx = float(ellipse.attrib.get('cx', 0))
                cy = float(ellipse.attrib.get('cy', 0))
                rx = float(ellipse.attrib.get('rx', 0))
                ry = float(ellipse.attrib.get('ry', 0))
                cpx = rx * 0.5522848
                cpy = ry * 0.5522848
                return (
                  f"M{cx},{cy-ry} "
                  f"C{cx+cpx},{cy-ry} {cx+rx},{cy-cpy} {cx+rx},{cy} "
                  f"C{cx+rx},{cy+cpy} {cx+cpx},{cy+ry} {cx},{cy+ry} "
                  f"C{cx-cpx},{cy+ry} {cx-rx},{cy+cpy} {cx-rx},{cy} "
                  f"C{cx-rx},{cy-cpy} {cx-cpx},{cy-ry} {cx},{cy-ry} Z"
                )
            except:
                return None

        def line_to_path(line):
            try:
                x1 = float(line.attrib.get('x1', 0))
                y1 = float(line.attrib.get('y1', 0))
                x2 = float(line.attrib.get('x2', 0))
                y2 = float(line.attrib.get('y2', 0))
                return f"M{x1},{y1} L{x2},{y2}"
            except:
                return None

        def polyline_to_path(poly):
            try:
                pts = poly.attrib.get('points', '')
                pairs = re.findall(r'(-?\d+\.?\d*)[,\s]+(-?\d+\.?\d*)', pts)
                if not pairs:
                    return None
                path_str = f"M{pairs[0][0]},{pairs[0][1]}"
                for (px, py) in pairs[1:]:
                    path_str += f" L{px},{py}"
                return path_str
            except:
                return None

        def polygon_to_path(poly):
            path_str = polyline_to_path(poly)
            if path_str:
                path_str += " Z"
            return path_str

        # Actual parsing:
        paths = []
        if not os.path.exists(svg_file_path):
            print(f"SVG file not found: {svg_file_path}")
            return paths

        tree = ET.parse(svg_file_path)
        root = tree.getroot()

        # Try to read width/height from SVG
        w = root.attrib.get('width')
        h = root.attrib.get('height')
        if w:
            try:
                self.svg_width = float(w.replace('px',''))
            except:
                pass
        if h:
            try:
                self.svg_height = float(h.replace('px',''))
            except:
                pass

        # Namespaces
        namespaces = {'svg': 'http://www.w3.org/2000/svg'}

        def find_all(el, tag):
            # attempt with namespace first
            found = el.findall(f".//svg:{tag}", namespaces)
            if not found:
                # fallback no namespace
                found = el.findall(f".//{tag}")
            return found

        # Collect <path> data
        for p in find_all(root, 'path'):
            d = p.attrib.get('d')
            if d:
                paths.append(d)

        # rect
        for r in find_all(root, 'rect'):
            rp = rect_to_path(r)
            if rp:
                paths.append(rp)

        # circle
        for c in find_all(root, 'circle'):
            cp = circle_to_path(c)
            if cp:
                paths.append(cp)

        # ellipse
        for e in find_all(root, 'ellipse'):
            ep = ellipse_to_path(e)
            if ep:
                paths.append(ep)

        # line
        for l in find_all(root, 'line'):
            lp = line_to_path(l)
            if lp:
                paths.append(lp)

        # polyline
        for pl in find_all(root, 'polyline'):
            plp = polyline_to_path(pl)
            if plp:
                paths.append(plp)

        # polygon
        for pg in find_all(root, 'polygon'):
            pgp = polygon_to_path(pg)
            if pgp:
                paths.append(pgp)

        print(f"Parsed {len(paths)} drawable paths from {svg_file_path}.")
        return paths

    def load_svg_strokes(self, svg_file_path):
        """
        Parse the given SVG file and store all the path data in self.svg_strokes.
        Reset the stroke_index to 0.
        """
        self.svg_strokes = self.parse_svg_file(svg_file_path)
        self.stroke_index = 0
        print("SVG loaded. Strokes ready to be drawn.")

    # --------------
    # STROKE-BY-STROKE DRAWING
    # --------------
    def draw_svg_stroke(self, stroke_data, speed: int = 20):
        """
        Draw a single SVG path (stroke).

        • Approaches the canvas at the first ‘M’.
        • Follows all subsequent draw commands while the pen is down.
        • Lifts the pen, then retreats from the point it actually finished on
        (so it never drags back over the stroke).
        """

        # -------------------------------------------------
        # tiny helpers
        # -------------------------------------------------
        def parse_path(path_data: str):
            """Return a list of (command, [float …]) tuples."""
            cmd_pat = r'([MLHVCSQTAZmlhvcsqtaz])([^MLHVCSQTAZmlhvcsqtaz]*)'
            num_pat = r'(-?\d+(?:\.\d+)?|\.\d+)'          # handles “.5” as well
            out = []
            for cmd, params in re.findall(cmd_pat, path_data):
                nums = [float(n) for n in re.findall(num_pat, params)]
                out.append((cmd, nums))
            return out

        def clamp_to_canvas(y: float, z: float):
            """Keep Y/Z inside the user‑defined canvas, recompute X from Z."""
            y = max(min(y, self.y_max), self.y_min)
            z = max(min(z, self.z_max), self.z_min)
            x = self.x_from_z(z)
            return x, y, z

        # -------------------------------------------------
        # main logic
        # -------------------------------------------------
        commands = parse_path(stroke_data)

        current_svg_pt   = None      # (x, y) in SVG space
        start_point_robot = None     # first ‘M’ in robot coords
        last_point_robot  = None     # updated after every pen‑down move
        first_move_done   = False

        for cmd, nums in commands:
            lower   = cmd.lower()
            is_rel  = (cmd == lower)

            # --------------- ‘M’ — move (pen up) -----------------
            if lower == 'm':
                for i in range(0, len(nums), 2):
                    x_svg, y_svg = nums[i], nums[i + 1]
                    if is_rel and current_svg_pt:
                        x_svg += current_svg_pt[0]
                        y_svg += current_svg_pt[1]
                    current_svg_pt = (x_svg, y_svg)

                    rx, ry, rz = self.map_svg_to_robot(x_svg, y_svg)
                    rx, ry, rz = clamp_to_canvas(ry, rz)

                    if not first_move_done:
                        # approach canvas (pen up), then touch down
                        self.approach_canvas(rx, ry, rz, speed=speed)
                        start_point_robot = (rx, ry, rz)
                        first_move_done   = True
                    else:
                        # we’re already on the canvas – lift a little, move, drop
                        self.arm.set_position(z=rz + 5,
                                            roll=-180, pitch=-90, yaw=0,
                                            speed=speed, wait=True)
                        self.arm.set_position(x=rx, y=ry,
                                            roll=-180, pitch=-90, yaw=0,
                                            z=rz + 5, speed=speed, wait=True)
                        self.arm.set_position(z=rz,
                                            roll=-180, pitch=-90, yaw=0,
                                            speed=speed, wait=True)
                    last_point_robot = (rx, ry, rz)

            # --------------- line commands (pen down) -------------
            elif lower in ('l', 'h', 'v'):
                # normalise the parameter list into (x, y) pairs
                pairs = []
                if lower == 'l':                # x, y pairs
                    for i in range(0, len(nums), 2):
                        pairs.append((nums[i], nums[i + 1]))
                elif lower == 'h':              # horizontal line(s): x only
                    for x_val in nums:
                        pairs.append((x_val, 0.0))
                elif lower == 'v':              # vertical line(s): y only
                    for y_val in nums:
                        pairs.append((0.0, y_val))

                for dx, dy in pairs:
                    if current_svg_pt is None:
                        continue
                    x_svg, y_svg = current_svg_pt
                    if lower == 'l':
                        if is_rel:
                            x_svg += dx
                            y_svg += dy
                        else:
                            x_svg, y_svg = dx, dy
                    elif lower == 'h':
                        x_svg = x_svg + dx if is_rel else dx
                    else:  # 'v'
                        y_svg = y_svg + dy if is_rel else dy

                    current_svg_pt = (x_svg, y_svg)
                    rx, ry, rz = self.map_svg_to_robot(x_svg, y_svg)
                    rx, ry, rz = clamp_to_canvas(ry, rz)

                    self.arm.set_position(x=rx, y=ry, z=rz,
                                        roll=-180, pitch=-90, yaw=0,
                                        speed=speed, wait=True)
                    last_point_robot = (rx, ry, rz)

            # --------------- ‘Z’ — close path ----------------------
            # elif lower == 'z' and start_point_robot:
            #     # lift a little, travel back to the start (in the air), drop, draw
            #     lift = 5
            #     self.arm.set_position(z=last_point_robot[2] + lift,
            #                         roll=-180, pitch=-90, yaw=0,
            #                         speed=speed, wait=True)
            #     self.arm.set_position(x=start_point_robot[0], y=start_point_robot[1],
            #                         z=last_point_robot[2] + lift,
            #                         roll=-180, pitch=-90, yaw=0,
            #                         speed=speed, wait=True)
            #     self.arm.set_position(z=start_point_robot[2],
            #                         roll=-180, pitch=-90, yaw=0,
            #                         speed=speed, wait=True)
            #     last_point_robot = start_point_robot

            # --------------- (any other command types ignored) ----

        # -------------------------------------------------
        # pen up → retreat off canvas
        # -------------------------------------------------
        if first_move_done and last_point_robot:
            self.retreat_from_canvas(*last_point_robot, speed=speed)

        return True

    
    def map_svg_to_robot(self, svg_x, svg_y):
        """
        Convert (svg_x, svg_y) in [0..svg_width, 0..svg_height]
        into (robot_x, robot_y, robot_z).

        Here we flip the vertical axis so that svg_y=0 corresponds
        to z_max (top) and svg_y=svg_height corresponds to z_min (bottom).
        """
        nx = svg_x / self.svg_width   # 0..1 (left to right)
        ny = svg_y / self.svg_height  # 0..1 (top to bottom)

        # # FLIP the vertical axis for z:
        # robot_z = self.z_max - ny * (self.z_max - self.z_min)
        # robot_y = self.y_min + nx * (self.y_max - self.y_min)

        # flipped
        robot_z = self.z_min + ny * (self.z_max - self.z_min)
        robot_y = self.y_max - nx * (self.y_max - self.y_min)

        # Finally, compute x from z
        robot_x = self.x_from_z(robot_z)
        return (robot_x, robot_y, robot_z)
    # --------------
    # TOP-LEVEL DRAW
    # --------------
    def draw_next_stroke(self, speed=STROKE_SPEED):
        """
        Draw exactly one path from self.svg_strokes[self.stroke_index],
        then increment stroke_index.
        """
        if self.stroke_index >= len(self.svg_strokes):
            print("No more SVG strokes to draw.")
            return

        stroke_data = self.svg_strokes[self.stroke_index]
        print(f"Drawing stroke {self.stroke_index+1}/{len(self.svg_strokes)}")
        self.draw_svg_stroke(stroke_data, speed=speed)
        self.stroke_index += 1
        
        self.osc_client.send_message("/stroke/done", self.stroke_index)
        print("Stroke done, sent OSC /stroke/done")

    # --------------
    # EXAMPLE RANDOM-FIGURE DRAWING
    # --------------
    def draw_canvas_border(self):
        # Just a quick example: top-left -> top-right -> bottom-right -> bottom-left -> top-left
        tl, tr, bl, br = self.canvas_corners
        square_points = [tl, tr, br, bl, tl]
        for i in range(len(square_points) - 1):
            self.draw_line(square_points[i], square_points[i+1], speed=100)


# Instantiate
controller = XArmController('192.168.0.203')
controller.arm.set_servo_angle(servo_id=5, angle=90, wait=True)
controller.osc_client.send_message("/listen", 1)


def interactive(address, *args):
    """
    Example OSC handler. We’ll add a “svg” sub-command for loading and stepping
    through the SVG, stroke by stroke.
    """
    max_msg = list(args)
    cmd = max_msg[0]

    if cmd != 'suggested_fingers':  # just so we see them
        print("OSC message:", max_msg)

    if cmd == 'homing':
        # example code
        if max_msg[1] == 'draw_border':
            controller.draw_canvas_border()
        else:
            # set corners
            corner_type = max_msg[1]  # top_left, top_right, etc.
            x_val = max_msg[2]
            y_val = max_msg[3]
            z_val = max_msg[4]
            if corner_type == 'top_left':
                controller.canvas_corners[0] = [x_val, y_val, z_val]
            elif corner_type == 'top_right':
                controller.canvas_corners[1] = [x_val, y_val, z_val]
            elif corner_type == 'bottom_left':
                controller.canvas_corners[2] = [x_val, y_val, z_val]
            elif corner_type == 'bottom_right':
                controller.canvas_corners[3] = [x_val, y_val, z_val]

            if max_msg[5] == 1:
                controller.arm.set_position(
                    x=x_val, y=y_val, z=z_val,
                    roll=-180, pitch=-90, yaw=0,
                    speed=100, wait=True
                )
            controller.calculate_scaling_factors()
            controller.calculate_canvas_bounds()
            print("New canvas corners:", controller.canvas_corners)

    elif cmd == 'video':
        controller.use_video = (max_msg[1] == 1)

    elif cmd == 'fingers':
        controller.fingers_allowed = max_msg[1]
        print("Fingers allowed changed to", max_msg[1])

    elif cmd == 'pressure':
        pval = max_msg[1]
        if pval <= controller.paint_pressure_range:
            controller.current_pressure = pval
            print("Pressure changed to", pval)
        else:
            print("Pressure out of range!", pval)

    elif cmd == 'suggested_fingers':
        # example logic
        suggested = max_msg[1]
        if controller.strokes_since_colour_change > 4:
            if suggested > controller.fingers_allowed:
                fingers_to_use = controller.fingers_allowed
            else:
                fingers_to_use = suggested
            arr = [1]*fingers_to_use + [0]*(4 - fingers_to_use)
            random.shuffle(arr)
            controller.osc_client.send_message("/fingers", arr)
            controller.strokes_since_colour_change = 0

    elif cmd == 'onset':
        controller.osc_client.send_message("/listen", 0)
        if max_msg[1] == 'dots':
            controller.draw_random_dots(1)
        elif max_msg[1] == 'lines':
            start = controller.generate_random_position()
            end   = controller.generate_random_position()
            controller.draw_line(start, end)
        elif max_msg[1] == 'arcs':
            controller.draw_random_arcs(1)
        else:
            print("Invalid drawing type:", max_msg[1])
        controller.strokes_since_colour_change += 1
        print("Finished drawing, listening again")

    elif cmd == 'panic':
        if max_msg[1] == 1:
            controller.allowed_to_play = True
            controller.osc_client.send_message("/listen", 1)
            print("Panic off, ready to play.")
        else:
            controller.allowed_to_play = False
            controller.osc_client.send_message("/listen", 0)
            print("PANIC MODE ON. Stopping new commands.")

    # ----------------
    # NEW: SVG COMMAND
    # ----------------
    elif cmd == 'svg':
        sub_cmd = max_msg[1]
        if sub_cmd == 'load':
            # E.g. /macbook svg load my_drawing.svg
            svg_file = max_msg[2]
            controller.load_svg_strokes(svg_file)
        elif sub_cmd == 'next':
            # E.g. /macbook svg next
            controller.osc_client.send_message("/listen", 0)
            controller.draw_next_stroke(speed=15)  # or pick your speed
        elif sub_cmd == 'clear':
            # Reset stroke index
            controller.stroke_index = 0
            print("Stroke index reset to 0.")
        else:
            print("Unrecognized subcommand for svg:", sub_cmd)

    else:
        print("Unrecognized OSC message:", max_msg)


# Setup OSC server
osc_dispatcher = dispatcher.Dispatcher()
osc_dispatcher.map("/macbook", interactive)

server = osc_server.ThreadingOSCUDPServer((LINUX_IP, MAXMSP_FROM_PORT), osc_dispatcher)
print("Serving on {}".format(server.server_address))


def on_exit():
    controller.disconnect()
    server.server_close()

atexit.register(on_exit)
server.serve_forever()
