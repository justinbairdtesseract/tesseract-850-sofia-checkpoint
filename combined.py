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
STROKE_SPEED = 500

face_detection_thread = vision.start_face_mesh('webcam', True)
# face_detection_thread = vision.start_face_mesh('filepath', True) # if you want to load from file

class XArmController:
    # AUDIO VARIABLE RANGES
    LOW_SPEED      = 20      # mm / s  when /audio speed = 0
    HIGH_SPEED     = 5000     # mm / s  when /audio speed = 127
    MAX_JITTER_MM  = 10.0     # ± mm  when /audio jitter = 127
    
    SMOOTH_MIN     = 5      # Bézier subdivisions when /audio smooth = 0
    SMOOTH_MAX     = 30      # Bézier subdivisions when /audio smooth = 127

    STIPPLE_THRESHOLD = 0.5  # Min stipple_level to activate effect
    STIPPLE_BASE_SEGMENT_MM = 5.0  # Base length of a stipple segment/gap in mm
    STIPPLE_LIFT_MM = 10.0          # How much to lift between stipples
    
    STIPPLE_MIN_SEGMENT_MM       = 0.3   # minimum spacing at maximum stipple
    STIPPLE_MIN_SEGMENT_MM_MAX   = 1.0   # spacing when stipple is off
    STIPPLE_RANGE_FACTOR         = 0.95  # how much of the base length can disappear


    
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
            [460,  150, 590],  # top_left
            [460, -125, 590],  # top_right
            [465,  150, 196],  # bottom_left
            [465, -125, 196]   # bottom_right
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
        self.svg_width = 768
        self.svg_height = 1080
        self.safety_margin = 10  # to keep from going out of bounds
        self.bezier_steps = 15   # for smooth curves

        # Brush attributes
        self.active_brush_index = 0
        self.brush_offset_mm = 25
        
        # --- live-audio modifiers (0–127 from Max) ---
        self.audio_params = {
            'speed':   127,   # default midpoint
            'jitter':  0,
            'smooth':  0,
            'stipple': 0,   # 0-127, where 0 is normal line, >0 means stippling
        }


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
        """
        Lift straight up by *lift* mm, then back away along +X by
        off_canvas_distance.  Y and Z stay exactly where the pen
        finished, so there’s no unexpected diagonal jump.
        """
        print('retreating')
        # 1) short vertical lift
        # self.arm.set_position(x=x, y=y, z=z + lift,
        #                     roll=-180, pitch=-90, yaw=0,
        #                     speed=speed, wait=True)

        # 2) slide straight back in X
        retreat_x = x - self.off_canvas_distance
        self.arm.set_position(x=retreat_x, y=y, z=z + lift,
                            roll=-180, pitch=-90, yaw=0,
                            speed=speed, wait=True)

        self.osc_client.send_message("/arm/movement", "retreating")
        self.osc_client.send_message("/listen", 1)

    def _lift_off_plane(self, x, y, z, speed, dist=None):
        """
        Pull the tip away from the canvas by <dist> mm along –x,
        leaving y / z unchanged.  Returns the new (x, y, z).
        """
        if dist is None:
            dist = self.STIPPLE_LIFT_MM
        new_x = x - dist           # –x is the normal to your canvas
        self.arm.set_position(x=new_x, y=y, z=z,
                            roll=-180, pitch=-90, yaw=0,
                            speed=speed, wait=True)
        return new_x, y, z

    # unified public wrapper for stipple retraction
    def retract_pen(self, x: float, y: float, z: float, speed: float):
        """
        Pull the brush away from the canvas by STIPPLE_LIFT_MM along –X
        (normal to the picture plane).  Returns the new lifted (x, y, z).
        """
        return self._lift_off_plane(x, y, z, speed)

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
        stipple_level = self._stipple_level()
        draw_spd      = speed

        if stipple_level > self.STIPPLE_THRESHOLD:
            p_start_rb = np.array(start, dtype=float)
            p_end_rb   = np.array(end,   dtype=float)
            dist       = np.linalg.norm(p_end_rb - p_start_rb)

            if dist < 0.1:                                   # too short – single dot
                self.draw_dot(*end, speed=draw_spd)
                return

            # decide number of dots
            lvl   = max(0.01, stipple_level)
            min_seg = self._stipple_min_spacing(lvl)
            seg   = max(min_seg,
                        self.STIPPLE_BASE_SEGMENT_MM *
                        (1.0 - 0.8 * lvl))
            n_dot = max(1, int(round(dist / seg)))

            # approach first point
            self.approach_canvas(*p_start_rb, speed=draw_spd)

            # first lift off-plane
            prev_x, prev_y, prev_z = self._lift_off_plane(
                p_start_rb[0], p_start_rb[1], p_start_rb[2], draw_spd)

            # iterate dots
            for i in range(n_dot):
                t    = (i + 1) / n_dot
                nx   = p_start_rb[0] + t * (p_end_rb[0] - p_start_rb[0])
                ny   = p_start_rb[1] + t * (p_end_rb[1] - p_start_rb[1])
                nz   = p_start_rb[2] + t * (p_end_rb[2] - p_start_rb[2])

                # move off-plane to new place
                self.arm.set_position(x=nx - self.STIPPLE_LIFT_MM,
                                      y=ny, z=nz,
                                      roll=-180, pitch=-90, yaw=0,
                                      speed=draw_spd, wait=True)

                # tap
                self.arm.set_position(x=nx, y=ny, z=nz,
                                      roll=-180, pitch=-90, yaw=0,
                                      speed=draw_spd/1.5, wait=True)

                # lift off again
                prev_x, prev_y, prev_z = self._lift_off_plane(
                    nx, ny, nz, draw_spd)

            # retreat (prev_* is already lifted)
            self.retreat_from_canvas(prev_x + self.STIPPLE_LIFT_MM,  # undo −x
                                     prev_y,
                                     prev_z, speed=draw_spd)

        else:
            # ordinary continuous line
            self.approach_canvas(*start, speed=draw_spd)
            self.arm.set_position(*end, roll=-180, pitch=-90, yaw=0,
                                  speed=draw_spd, wait=True)
            self.retreat_from_canvas(*end, speed=draw_spd)


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
        self.osc_client.send_message("/strokes/total", len(self.svg_strokes))
        print("SVG loaded. Strokes ready to be drawn.")

    # --------------
    # STROKE-BY-STROKE DRAWING
    # --------------
       # ------------------------------------------------------------------
    # FULL-FEATURE DRAW-SVG-STROKE  ──  supports M/L/H/V/C/S/Q/T/A/Z
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    #  FULL-FEATURE, AUDIO-RESPONSIVE  draw_svg_stroke
    # ------------------------------------------------------------------

    def draw_svg_stroke(self, path_data, speed=None):
        """
        Draw one SVG <path> using live-audio modifiers.

        Audio params (0–127):
            speed  →  20–200 mm/s      (self._speed_mm_s)
            jitter →  0–3 mm ±Y/±Z     (self._jitter_mm)
            smooth →  5–30 Bézier pts  (self._smooth_steps)
        """
        if speed is None:
            base_speed = self._speed_mm_s()
        else:
            base_speed = speed
        # ---------- helpers ----------
        cmd_pat = r'([MLHVCSQTAZmlhvcsqtaz])([^MLHVCSQTAZmlhvcsqtaz]*)'
        num_pat = r'(-?\d+(?:\.\d+)?|\.\d+)'

        def parse_path(d):
            for cmd, params in re.findall(cmd_pat, d):
                yield cmd, [float(n) for n in re.findall(num_pat, params)]

        def clamp(y, z):
            offset = self.active_brush_index * self.brush_offset_mm

            # Canvas band **shifted** by the same offset
            y_lower = self.y_min + self.safety_margin + offset
            y_upper = self.y_max - self.safety_margin + offset

            y = min(max(y, y_lower), y_upper)

            z = min(max(z, self.z_min + self.safety_margin),
                    self.z_max - self.safety_margin)

            return self.x_from_z(z), y, z
        # --------------------------------

        steps     = self._smooth_steps()   # live “smooth” (Bézier subdivisions)
        cur       = None                   # current SVG (x, y)
        start_rb  = None                   # robot point of current sub-path
        last_rb   = None                   # **last robot (x,y,z) sent**
        last_ctl  = None                   # for smooth Béziers
        pen_down  = False

        for cmd, nums in parse_path(path_data):
            op, rel  = cmd.lower(), cmd.islower()
            draw_spd = self._speed_mm_s()        # live speed
            jitter   = self._jitter_mm()         # live jitter (0–3 mm)
            stipple_level = self._stipple_level() # live stipple (0-1)

            # ========== MOVE (M/m) ==========
            if op == 'm':
                for i in range(0, len(nums), 2):
                    x, y = nums[i:i+2]
                    if rel and cur:
                        x += cur[0];  y += cur[1]
                    cur = (x, y)

                    rx, ry, rz = self.map_svg_to_robot(x, y)
                    rx, ry, rz = clamp(ry, rz)

                    if not pen_down:                     # first sub-path
                        self.approach_canvas(rx, ry, rz, speed=draw_spd)
                        pen_down  = True
                    else:                                # subsequent “M” inside <path>
                        lift = 5
                        self.arm.set_position(z=last_rb[2] + lift, roll=-180,
                                            pitch=-90, yaw=0, speed=draw_spd, wait=True)
                        self.arm.set_position(x=rx, y=ry, z=rz + lift, roll=-180,
                                            pitch=-90, yaw=0, speed=draw_spd, wait=True)
                        self.arm.set_position(z=rz, roll=-180, pitch=-90, yaw=0,
                                            speed=draw_spd, wait=True)
                    start_svg = (x, y)          # ← add this
                    start_rb = last_rb = (rx, ry, rz)
                last_ctl = None

            # ========== LINE / H / V ==========
            elif op in ('l', 'h', 'v'):
                raw_segs = [] # holds (dx,dy) or (abs_x, abs_y) based on op type for this L/H/V sequence
                if op == 'l':
                    raw_segs = [(nums[i], nums[i+1]) for i in range(0, len(nums), 2)]
                elif op == 'h': # Relative or absolute H moves
                    raw_segs = [(val, cur[1] if not rel else 0) for val in nums] # if rel, dy=0; if abs, y=cur[1]
                    # if absolute H, the 'val' is the new X, Y is current Y. We need to convert to (target_x, target_y) format.
                    # if relative H, the 'val' is dx, dy is 0.
                    # This will be handled when processing target_svg_points.
                else: # Relative or absolute V moves
                    raw_segs = [(cur[0] if not rel else 0, val) for val in nums]

                # Calculate all target SVG points for this command group
                target_svg_points = []
                temp_svg_cur = cur # Temporary SVG cursor for this sequence
                for i_seg_val in range(len(raw_segs)):
                    if op == 'l':
                        dx_or_abs_x, dy_or_abs_y = raw_segs[i_seg_val]
                        if rel:
                            next_svg_x, next_svg_y = temp_svg_cur[0] + dx_or_abs_x, temp_svg_cur[1] + dy_or_abs_y
                        else:
                            next_svg_x, next_svg_y = dx_or_abs_x, dy_or_abs_y
                    elif op == 'h':
                        val = nums[i_seg_val]
                        next_svg_x = temp_svg_cur[0] + val if rel else val
                        next_svg_y = temp_svg_cur[1] # Y does not change
                    else: # op == 'v'
                        val = nums[i_seg_val]
                        next_svg_x = temp_svg_cur[0] # X does not change
                        next_svg_y = temp_svg_cur[1] + val if rel else val

                    target_svg_points.append((next_svg_x, next_svg_y))
                    temp_svg_cur = (next_svg_x, next_svg_y)

                # Process each segment from the target_svg_points list
                segment_start_rb = last_rb
                for svg_idx, (x_svg_target, y_svg_target) in enumerate(target_svg_points):
                    cur = (x_svg_target, y_svg_target) # Official update to current SVG position

                    rx_target, ry_target, rz_target = self.map_svg_to_robot(x_svg_target, y_svg_target)
                    if jitter: # Apply jitter to the target robot coordinates
                        ry_target += random.uniform(-jitter, jitter)
                        rz_target += random.uniform(-jitter, jitter)
                    rx_target, ry_target, rz_target = clamp(ry_target, rz_target) # Clamp after jitter

                    segment_end_rb = (rx_target, ry_target, rz_target)

                    if stipple_level > self.STIPPLE_THRESHOLD:
                        # Ensure pen is lifted before starting stipple segment if it was drawing continuously
                        if pen_down and not getattr(self, '_stipple_was_active', False):
                             # Lift away from the canvas before stippling
                             segment_start_rb = self.retract_pen(segment_start_rb[0],
                                                                  segment_start_rb[1],
                                                                  segment_start_rb[2],
                                                                  draw_spd)

                        last_rb = self._stipple_segment_runner(segment_start_rb, segment_end_rb, draw_spd, stipple_level)
                        # _stipple_segment_runner now sets 'self._stipple_was_active' to True internally
                    else:
                        # If previous segment was stippling, need to move to start of continuous line and lower pen
                        if getattr(self, '_stipple_was_active', False):
                            # segment_start_rb is already lifted from previous stipple. Move to start XY, then lower.
                            self.arm.set_position(x=segment_start_rb[0], y=segment_start_rb[1], z=segment_start_rb[2],
                                                  roll=-180, pitch=-90, yaw=0, speed=draw_spd, wait=True) # move to XY (still lifted)
                            self.arm.set_position(x=segment_start_rb[0], y=segment_start_rb[1], z=segment_end_rb[2], # Lower to drawing Z
                                                  roll=-180, pitch=-90, yaw=0, speed=draw_spd / 1.5, wait=True)

                        # Normal continuous line segment
                        self.arm.set_position(x=segment_end_rb[0], y=segment_end_rb[1], z=segment_end_rb[2], roll=-180,
                                            pitch=-90, yaw=0, speed=draw_spd, wait=False)
                        last_rb = segment_end_rb
                        setattr(self, '_stipple_was_active', False)

                    segment_start_rb = last_rb # The end of the last segment is the start of the new one

                # After all segments in L/H/V command, if not stippling and arm is moving, wait.
                if not (stipple_level > self.STIPPLE_THRESHOLD) and self.arm.get_is_moving():
                    self.pause_movement()
                elif stipple_level > self.STIPPLE_THRESHOLD:
                    # If we ended on stippling, ensure the arm is responsive for the next command
                    # (stipple_segment_runner should leave it in a good state, lifted)
                    pass


                last_ctl = None

            # ========== QUADRATIC / SMOOTH Q ==========
            elif op in ('q', 't'):
                i = 0
                while i < len(nums):
                    if op == 'q':
                        cx, cy, x, y = nums[i:i+4];  i += 4
                    else:
                        x, y = nums[i:i+2];          i += 2
                        if last_ctl:
                            cx = 2*cur[0] - last_ctl[0]
                            cy = 2*cur[1] - last_ctl[1]
                        else:
                            cx, cy = cur
                    if rel:
                        cx += cur[0]; cy += cur[1]; x += cur[0]; y += cur[1]

                    p0 = np.array(cur)
                    p1 = np.array([cx, cy])
                    p2 = np.array([x, y])

                    # Determine points along the curve
                    curve_points_svg = []
                    for t_val in np.linspace(0, 1, steps):
                        pt_svg = (1-t_val)**2*p0 + 2*t_val*(1-t_val)*p1 + t_val**2*p2
                        curve_points_svg.append(pt_svg)

                    # Draw the curve (stippled or continuous)
                    segment_start_rb = last_rb
                    for k_pt, pt_svg in enumerate(curve_points_svg):
                        rx_target, ry_target, rz_target = self.map_svg_to_robot(pt_svg[0], pt_svg[1])
                        if jitter:
                            ry_target += random.uniform(-jitter, jitter)
                            rz_target += random.uniform(-jitter, jitter)
                        rx_target, ry_target, rz_target = clamp(ry_target, rz_target)
                        segment_end_rb = (rx_target, ry_target, rz_target)

                        if stipple_level > self.STIPPLE_THRESHOLD:
                            # Ensure pen is lifted if transitioning from continuous draw or previous stipple segment ended low
                            if k_pt == 0 and pen_down and not getattr(self, '_stipple_was_active', False):
                                segment_start_rb = self.retract_pen(segment_start_rb[0],
                                                                    segment_start_rb[1],
                                                                    segment_start_rb[2],
                                                                    draw_spd)

                            # Tap at segment_end_rb (which is the current curve point)
                            # Move to the point while staying lifted along –X
                            self.arm.set_position(x=segment_end_rb[0] - self.STIPPLE_LIFT_MM,
                                                  y=segment_end_rb[1], z=segment_end_rb[2],
                                                  roll=-180, pitch=-90, yaw=0, speed=draw_spd, wait=True)
                            # Tap down
                            self.arm.set_position(x=segment_end_rb[0], y=segment_end_rb[1], z=segment_end_rb[2],
                                                  roll=-180, pitch=-90, yaw=0, speed=draw_spd / 1.5, wait=True)
                            # Lift up along –X
                            last_rb = self.retract_pen(segment_end_rb[0], segment_end_rb[1],
                                                       segment_end_rb[2], draw_spd)
                            setattr(self, '_stipple_was_active', True)
                        else:
                            # Continuous drawing for curve segment
                            if k_pt == 0 and getattr(self, '_stipple_was_active', False): # Transition from stipple to continuous
                                # segment_start_rb is already lifted. Move to its XY, then lower to segment_end_rb's Z.
                                self.arm.set_position(x=segment_start_rb[0], y=segment_start_rb[1], z=segment_start_rb[2], # XY of start (lifted)
                                                      roll=-180, pitch=-90, yaw=0, speed=draw_spd, wait=True)
                                self.arm.set_position(x=segment_start_rb[0], y=segment_start_rb[1], z=segment_end_rb[2], # Lower to target Z
                                                      roll=-180, pitch=-90, yaw=0, speed=draw_spd / 1.5, wait=True)

                            self.arm.set_position(x=segment_end_rb[0], y=segment_end_rb[1], z=segment_end_rb[2], roll=-180,
                                                pitch=-90, yaw=0, speed=draw_spd, wait=False)
                            last_rb = segment_end_rb
                            setattr(self, '_stipple_was_active', False)

                        segment_start_rb = last_rb # for the next iteration or if this is the last point

                    # After all points in curve, if not stippling and arm is moving, wait.
                    if not (stipple_level > self.STIPPLE_THRESHOLD) and self.arm.get_is_moving():
                        self.pause_movement()

                    cur, last_ctl = (x, y), (cx, cy)

            # ========== CUBIC / SMOOTH C ==========
            elif op in ('c', 's'):
                i = 0
                while i < len(nums):
                    if op == 'c':
                        c1x, c1y, c2x, c2y, x, y = nums[i:i+6];  i += 6
                    else:
                        c2x, c2y, x, y = nums[i:i+4];  i += 4
                        if last_ctl:
                            c1x = 2*cur[0] - last_ctl[0]
                            c1y = 2*cur[1] - last_ctl[1]
                        else:
                            c1x, c1y = cur
                    if rel:
                        c1x += cur[0]; c1y += cur[1]
                        c2x += cur[0]; c2y += cur[1]
                        x   += cur[0]; y   += cur[1]

                    p0 = np.array(cur)
                    p1 = np.array([c1x, c1y])
                    p2 = np.array([c2x, c2y])
                    p3 = np.array([x, y])

                    # Determine points along the curve
                    curve_points_svg = []
                    for t_val in np.linspace(0, 1, steps):
                        pt_svg = (1-t_val)**3*p0 + 3*t_val*(1-t_val)**2*p1 \
                               + 3*(t_val**2)*(1-t_val)*p2 + (t_val**3)*p3
                        curve_points_svg.append(pt_svg)

                    # Draw the curve (stippled or continuous)
                    segment_start_rb = last_rb
                    for k_pt, pt_svg in enumerate(curve_points_svg):
                        rx_target, ry_target, rz_target = self.map_svg_to_robot(pt_svg[0], pt_svg[1])
                        if jitter:
                            ry_target += random.uniform(-jitter, jitter)
                            rz_target += random.uniform(-jitter, jitter)
                        rx_target, ry_target, rz_target = clamp(ry_target, rz_target)
                        segment_end_rb = (rx_target, ry_target, rz_target)

                        if stipple_level > self.STIPPLE_THRESHOLD:
                            if k_pt == 0 and pen_down and not getattr(self, '_stipple_was_active', False):
                                segment_start_rb = self.retract_pen(segment_start_rb[0],
                                                                    segment_start_rb[1],
                                                                    segment_start_rb[2],
                                                                    draw_spd)

                                self.arm.set_position(x=segment_end_rb[0] - self.STIPPLE_LIFT_MM,
                                                      y=segment_end_rb[1], z=segment_end_rb[2],
                                                      roll=-180, pitch=-90, yaw=0, speed=draw_spd, wait=True)
                                self.arm.set_position(x=segment_end_rb[0], y=segment_end_rb[1], z=segment_end_rb[2],
                                                      roll=-180, pitch=-90, yaw=0, speed=draw_spd / 1.5, wait=True)
                                last_rb = self.retract_pen(segment_end_rb[0], segment_end_rb[1],
                                                           segment_end_rb[2], draw_spd)
                                setattr(self, '_stipple_was_active', True)
                        else:
                            if k_pt == 0 and getattr(self, '_stipple_was_active', False):
                                self.arm.set_position(x=segment_start_rb[0], y=segment_start_rb[1], z=segment_start_rb[2],
                                                      roll=-180, pitch=-90, yaw=0, speed=draw_spd, wait=True)
                                self.arm.set_position(x=segment_start_rb[0], y=segment_start_rb[1], z=segment_end_rb[2],
                                                      roll=-180, pitch=-90, yaw=0, speed=draw_spd / 1.5, wait=True)

                            self.arm.set_position(x=segment_end_rb[0], y=segment_end_rb[1], z=segment_end_rb[2], roll=-180,
                                                pitch=-90, yaw=0, speed=draw_spd, wait=False)
                            last_rb = segment_end_rb
                            setattr(self, '_stipple_was_active', False)

                        segment_start_rb = last_rb

                    if not (stipple_level > self.STIPPLE_THRESHOLD) and self.arm.get_is_moving():
                        self.pause_movement()

                    cur, last_ctl = (x, y), (c2x, c2y)

            # ========== ELLIPTICAL ARC (A/a) ==========
            elif op == 'a':
                i = 0
                while i + 6 < len(nums):
                    x, y = nums[i+5], nums[i+6];  i += 7
                    if rel:
                        x += cur[0]; y += cur[1]
                    cur = (x, y)
                    rx, ry, rz = self.map_svg_to_robot(x, y)
                    if jitter:
                        ry += random.uniform(-jitter, jitter)
                        rz += random.uniform(-jitter, jitter)
                    rx_target, ry_target, rz_target = clamp(ry, rz) # rx,ry,rz were from self.map_svg_to_robot(x,y)

                    segment_end_rb = (rx_target, ry_target, rz_target)

                    if stipple_level > self.STIPPLE_THRESHOLD:
                        # Ensure pen is lifted before starting stipple segment
                        if pen_down and not getattr(self, '_stipple_was_active', False):
                             last_rb = self.retract_pen(last_rb[0], last_rb[1], last_rb[2], draw_spd)

                        last_rb = self._stipple_segment_runner(last_rb, segment_end_rb, draw_spd, stipple_level)
                    else:
                        # If previous segment was stippling, need to move to start of continuous line and lower pen
                        if getattr(self, '_stipple_was_active', False):
                            self.arm.set_position(x=last_rb[0], y=last_rb[1], z=last_rb[2], # XY of current (lifted), Z of current (lifted)
                                                  roll=-180, pitch=-90, yaw=0, speed=draw_spd, wait=True)
                            self.arm.set_position(x=last_rb[0], y=last_rb[1], z=segment_end_rb[2], # Lower to target Z for drawing
                                                  roll=-180, pitch=-90, yaw=0, speed=draw_spd / 1.5, wait=True)

                        self.arm.set_position(x=segment_end_rb[0], y=segment_end_rb[1], z=segment_end_rb[2], roll=-180,
                                            pitch=-90, yaw=0, speed=draw_spd, wait=True) # Arcs are usually single moves
                        last_rb = segment_end_rb
                        setattr(self, '_stipple_was_active', False)
                last_ctl = None

            # ========== CLOSE PATH (Z/z) ==========
            elif op == 'z' and start_rb: # start_rb is the robot coords of the start of the current sub-path
                # Target for Z command is start_rb
                segment_end_rb = start_rb

                # Current position is last_rb
                # We need to draw/stipple a line from last_rb to start_rb

                if stipple_level > self.STIPPLE_THRESHOLD:
                    # Ensure pen is lifted before stippling the closing segment
                    if pen_down and not getattr(self, '_stipple_was_active', False):
                        last_rb = self.retract_pen(last_rb[0], last_rb[1], last_rb[2], draw_spd)

                    closed_path_final_rb = self._stipple_segment_runner(last_rb, segment_end_rb, draw_spd, stipple_level)
                    # After stippling, arm is lifted at segment_end_rb (which is start_rb)
                    # For Z, we want to ensure it's at the drawing depth of start_rb if it were to continue,
                    # but retreat will handle the final lift. So, keep it lifted is fine.
                    last_rb = closed_path_final_rb
                else:
                    # Continuous line for Z
                    if getattr(self, '_stipple_was_active', False): # Transition from stipple
                        self.arm.set_position(x=last_rb[0], y=last_rb[1], z=last_rb[2], # XY of current (lifted)
                                              roll=-180, pitch=-90, yaw=0, speed=draw_spd, wait=True)
                        self.arm.set_position(x=last_rb[0], y=last_rb[1], z=segment_end_rb[2], # Lower to target Z
                                              roll=-180, pitch=-90, yaw=0, speed=draw_spd / 1.5, wait=True)

                    self.arm.set_position(x=segment_end_rb[0], y=segment_end_rb[1], z=segment_end_rb[2], roll=-180,
                                        pitch=-90, yaw=0, speed=draw_spd, wait=True) # wait=True for Z
                    last_rb = segment_end_rb
                    setattr(self, '_stipple_was_active', False)

                cur = start_svg # SVG current point resets to start of sub-path
                last_ctl = None

            # ---- unrecognised commands ignored ----

        # --------------- PEN UP & RETREAT ---------------
        # If the last operation was stippling, last_rb is already lifted.
        # retreat_from_canvas expects the Z at drawing depth for its initial lift calculation.
        if pen_down and last_rb:
            final_z_for_retreat = last_rb[2]
            if getattr(self, '_stipple_was_active', False):
                # retract_pen keeps Z constant, so no adjustment needed
                final_z_for_retreat = last_rb[2]

            self.retreat_from_canvas(last_rb[0], last_rb[1], final_z_for_retreat, speed=self._speed_mm_s())
            # Clear the stippling state for the next independent stroke
            if hasattr(self, '_stipple_was_active'): # Check attribute on self
                delattr(self, '_stipple_was_active') # Delete attribute from self


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
         # robot_z = self.z_min + ny * (self.z_max - self.z_min)
        robot_z = self.z_max - ny * (self.z_max - self.z_min)

        total_offset = self.active_brush_index * self.brush_offset_mm
        
        robot_y = self.y_min + nx * (self.y_max - self.y_min) + total_offset

        # robot_y = self.y_min + nx * (self.y_max - self.y_min)


         # Finally, compute x from z
        robot_x = self.x_from_z(robot_z)
        return (robot_x, robot_y, robot_z)
    
    
   

        
    
    # --------------
    # TOP-LEVEL DRAW
    # --------------
    def draw_next_stroke(self, count=1, speed=None):
            """
            Draw *count* successive SVG paths (default = 1) and bump
            self.stroke_index accordingly.  Stops early if we run out.
            """
            drawn = 0
            while drawn < count and self.stroke_index < len(self.svg_strokes):
                stroke_data = self.svg_strokes[self.stroke_index]
                print(f"Drawing stroke {self.stroke_index+1}/{len(self.svg_strokes)}")
                self.draw_svg_stroke(stroke_data, speed=speed)
                self.stroke_index += 1
                drawn += 1
                self.osc_client.send_message("/stroke/done", self.stroke_index)

            if self.stroke_index >= len(self.svg_strokes):
                print("SVG finished – no more strokes left.")

    # --------------
    # EXAMPLE RANDOM-FIGURE DRAWING
    # --------------
    def draw_canvas_border(self):
        # Just a quick example: top-left -> top-right -> bottom-right -> bottom-left -> top-left
        tl, tr, bl, br = self.canvas_corners
        square_points = [tl, tr, br, bl, tl]
        for i in range(len(square_points) - 1):
            self.draw_line(square_points[i], square_points[i+1], speed=100)

    ## AUDIO VARIATIONS
    # translate 0-127 ➜ robot units
    def _speed_mm_s(self):
        """0–127 → LOW_SPEED … HIGH_SPEED (mm/s)"""
        v = self.audio_params['speed']
        return self.LOW_SPEED + (v / 127) * (self.HIGH_SPEED - self.LOW_SPEED)

    def _jitter_mm(self):
        """0–127 → 0 … MAX_JITTER_MM (mm)"""
        return (self.audio_params['jitter'] / 127) * self.MAX_JITTER_MM

    def _smooth_steps(self):
        """0–127 → SMOOTH_MIN … SMOOTH_MAX (integer steps)"""
        v = self.audio_params['smooth']
        return int(self.SMOOTH_MIN + (v / 127) * (self.SMOOTH_MAX - self.SMOOTH_MIN))

    def _stipple_level(self):
        """0–127 → 0.0 … 1.0 (stipple intensity/probability)"""
        # We can refine the scaling later if needed (e.g., non-linear)
        return self.audio_params['stipple'] / 127.0

    def _stipple_min_spacing(self, level: float) -> float:
        """Return the dynamic minimum dot spacing based on stipple level."""
        hi = self.STIPPLE_MIN_SEGMENT_MM_MAX
        lo = self.STIPPLE_MIN_SEGMENT_MM
        return hi - (hi - lo) * level

    def _stipple_segment_runner(self, p_start_rb, p_end_rb, speed, current_stipple_level):
        """
        Draw a stippled segment from p_start_rb → p_end_rb, lifting off-plane
        (−x) between dots.  Returns the final lifted (x,y,z).
        """
        dx = p_end_rb[0] - p_start_rb[0]
        dy = p_end_rb[1] - p_start_rb[1]
        dz = p_end_rb[2] - p_start_rb[2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)

        if dist < 0.5:                                              # tiny → single tap
            self._lift_off_plane(*p_start_rb, speed=speed)
            self.arm.set_position(*p_start_rb, roll=-180, pitch=-90, yaw=0,
                                  speed=speed/1.5, wait=True)
            lifted = self._lift_off_plane(*p_start_rb, speed=speed)
            setattr(self, '_stipple_was_active', True)
            return lifted

        lvl      = max(0.01, current_stipple_level)
        min_seg  = self._stipple_min_spacing(lvl)
        seg_len  = max(min_seg,
                       self.STIPPLE_BASE_SEGMENT_MM *
                       (1.0 - self.STIPPLE_RANGE_FACTOR * (lvl**2)))
        n_dot    = max(2, int(math.ceil(dist / seg_len)))

        # first dot
        cur_x, cur_y, cur_z = self._lift_off_plane(*p_start_rb, speed=speed)
        self.arm.set_position(*p_start_rb, roll=-180, pitch=-90, yaw=0,
                              speed=speed/1.5, wait=True)
        cur_x, cur_y, cur_z = self._lift_off_plane(*p_start_rb, speed=speed)

        # remaining dots
        for i in range(1, n_dot):
            t  = i / (n_dot - 1)
            nx = p_start_rb[0] + t*dx
            ny = p_start_rb[1] + t*dy
            nz = p_start_rb[2] + t*dz

            self.arm.set_position(x=nx - self.STIPPLE_LIFT_MM, y=ny, z=nz,
                                  roll=-180, pitch=-90, yaw=0,
                                  speed=speed, wait=True)
            self.arm.set_position(x=nx, y=ny, z=nz,
                                  roll=-180, pitch=-90, yaw=0,
                                  speed=speed/1.5, wait=True)
            cur_x, cur_y, cur_z = self._lift_off_plane(nx, ny, nz, speed)

        setattr(self, '_stipple_was_active', True)
        return cur_x, cur_y, cur_z


    def get_current_arm_position(self):
        """Gets the current arm position and returns x, y, z coordinates."""
        code, position = self.arm.get_position()
        if code == 0:
            return position[0], position[1], position[2]
        else:
            print(f"Error getting arm position: Code {code}")
            # Potentially send an OSC error message back
            self.osc_client.send_message("/arm/error", f"get_position_failed_code_{code}")
            return None, None, None


# Global controller instance, to be initialized in main
controller = None

def interactive(address, *args):
    """
    Example OSC handler. We’ll add a “svg” sub-command for loading and stepping
    through the SVG, stroke by stroke.
    """
    global controller # Ensure we're using the global controller initialized in main
    if not controller:
        print("Error: Controller not initialized.")
        return

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
            controller.draw_next_stroke()
        elif sub_cmd == 'goto':             # /macbook svg goto 45
            idx = int(max_msg[2])
            if 0 <= idx < len(controller.svg_strokes):
                controller.stroke_index = idx
                print("Stroke index set to", idx)
            else:
                print("Stroke index out of range")
        elif sub_cmd == 'exec':             # /macbook svg exec 45
            idx = int(max_msg[2])
            if 0 <= idx < len(controller.svg_strokes):
                controller.osc_client.send_message("/listen", 0)
                controller.draw_svg_stroke(controller.svg_strokes[idx],
                                            speed=STROKE_SPEED)
                controller.osc_client.send_message("/stroke/done", idx+1)
            else:
                print("Stroke index out of range")      
                
        elif sub_cmd == 'clear':
            # Reset stroke index
            controller.stroke_index = 0
            print("Stroke index reset to 0.")
        else:
            print("Unrecognized subcommand for svg:", sub_cmd)
            
    elif cmd == 'audio':        # /macbook audio <name> <0-127>
        name  = max_msg[1]      # 'speed' / 'jitter' / 'smooth' / 'stipple'
        value = int(max_msg[2])
        if name in controller.audio_params:
            controller.audio_params[name] = max(0, min(127, value))
            # print(f"Audio-param {name} set to {value}")
        else:
            print("Unknown audio-param", name)

    elif cmd == 'brush':
        brush_idx = int(max_msg[1])
        if 0 <= brush_idx <= 3:  # Assuming 4 brushes, so indices 0-3
            controller.active_brush_index = brush_idx
            print(f"Active brush index set to {brush_idx}")
        else:
            print(f"Invalid brush index: {brush_idx}. Must be between 0 and 3.")
            
    elif cmd == 'stipple_lift':
        try:
            new_lift = float(max_msg[1])
        except (IndexError, ValueError):
            print("stipple_lift: need a numeric value")
            return

        # optional safety clamp – change the limits if you wish
        if not (0.5 <= new_lift <= 50.0):
            print(f"stipple_lift {new_lift} mm out of range (0.5–50)")
            return

        controller.STIPPLE_LIFT_MM = new_lift
        print(f"Stipple-lift height set to {new_lift:.1f} mm")
        controller.osc_client.send_message("/stipple/lift", new_lift)



    elif cmd == 'get_position':
        x, y, z = controller.get_current_arm_position()
        if x is not None: # If no error getting position
            controller.osc_client.send_message("/arm/current_position", [float(x), float(y), float(z)])
            print(f"Sent current position: x={x}, y={y}, z={z}")
        # If there was an error, get_current_arm_position already sent an error OSC msg
            
    else:
        print("Unrecognized OSC message:", max_msg)

# This global variable will hold the server instance
osc_server_instance = None

def on_exit(controller_instance, server_instance):
    print("Exiting program...")
    if controller_instance:
        print("Disconnecting XArm controller...")
        controller_instance.disconnect()
    if server_instance:
        print("Closing OSC server...")
        server_instance.server_close()
    print("Cleanup finished.")

def main():
    global controller # Declare controller as global to modify it
    global osc_server_instance

    # Instantiate controller
    controller = XArmController('192.168.0.203') # Or get IP from config/env
    controller.arm.set_servo_angle(servo_id=5, angle=90, wait=True) # Example setup
    controller.osc_client.send_message("/listen", 1)

    # Setup OSC server
    osc_dispatcher = dispatcher.Dispatcher()
    osc_dispatcher.map("/macbook", interactive) # interactive will use the global controller

    osc_server_instance = osc_server.ThreadingOSCUDPServer((LINUX_IP, MAXMSP_FROM_PORT), osc_dispatcher)
    print("Serving on {}".format(osc_server_instance.server_address))

    # Register exit handler
    # Pass controller and server to on_exit using a lambda or functools.partial
    # because atexit doesn't pass arguments to registered functions directly.
    atexit.register(lambda: on_exit(controller, osc_server_instance))

    try:
        osc_server_instance.serve_forever()
    except KeyboardInterrupt:
        print("Keyboard interrupt received, shutting down.")
    finally:
        # Ensure cleanup is called if serve_forever() exits unexpectedly
        on_exit(controller, osc_server_instance)

if __name__ == '__main__':
    main()
