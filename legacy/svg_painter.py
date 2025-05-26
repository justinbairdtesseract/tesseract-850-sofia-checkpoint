import sys
import time
import os
import re
import math
import numpy as np
import xml.etree.ElementTree as ET

from xarm.wrapper import XArmAPI

class SVGPainter:
    def __init__(self, ip_address='192.168.0.203'):
        self.IP_ADDRESS = ip_address
        self.arm = XArmAPI(self.IP_ADDRESS)
        self.setup()
        
        # Canvas corners must always stay in this order: top left, top right, bottom left, bottom right
        self.canvas_corners = [
            [546, 205, 835],  # top_left
            [546, -333, 835], # top_right
            [508, 205, 88],   # bottom_left
            [508, -333, 88]   # bottom_right
        ]
        
        # SVG scaling parameters
        self.svg_width = 600   # Default SVG width
        self.svg_height = 600  # Default SVG height
        
        # Robot parameters
        self.off_canvas_distance = 25  # Distance to start away from the canvas on the x-axis
        self.safety_margin = 10        # Safety margin in mm from canvas edges
        self.calculate_scaling_factors()
        self.calculate_canvas_bounds()
        
        # Drawing parameters
        self.bezier_steps = 15  # For smoother curves

    def setup(self):
        """Initialize the robot arm."""
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)
        
        # Lower speed for initial setup to ensure safety
        self.arm.set_servo_angle(servo_id=5, angle=90, speed=20, wait=True)
        
        # Enable collision detection (if supported by your model)
        try:
            self.arm.set_collision_sensitivity(1)
        except:
            print("Collision sensitivity setting not supported on this model")
        
        # Wait for robot to be ready
        time.sleep(1)

    def disconnect(self):
        """Disconnect from the robot arm."""
        print("Disconnecting from robot arm...")
        self.arm.disconnect()
        print("Disconnected successfully.")

    def calculate_scaling_factors(self):
        """Calculate scaling factors for the robot based on the canvas corners."""
        z1, x1 = self.canvas_corners[0][2], self.canvas_corners[0][0]  # top-left corner
        z2, x2 = self.canvas_corners[2][2], self.canvas_corners[2][0]  # bottom-left corner
        self.a = (x2 - x1) / (z2 - z1)
        self.b = x1 - self.a * z1

    def x_from_z(self, z):
        """Calculate x coordinate based on z-coordinate in robot space."""
        return self.a * z + self.b

    def calculate_canvas_bounds(self):
        """Calculate the bounds of the canvas in robot coordinates with safety margins."""
        self.y_min = min(self.canvas_corners[0][1], self.canvas_corners[1][1]) + self.safety_margin
        self.y_max = max(self.canvas_corners[0][1], self.canvas_corners[1][1]) - self.safety_margin
        self.z_min = min(self.canvas_corners[2][2], self.canvas_corners[3][2]) + self.safety_margin
        self.z_max = max(self.canvas_corners[0][2], self.canvas_corners[1][2]) - self.safety_margin
        
        print(f"Canvas bounds: Y: [{self.y_min}, {self.y_max}], Z: [{self.z_min}, {self.z_max}]")

    def map_svg_to_robot(self, svg_x, svg_y):
        """
        Map SVG coordinates to robot coordinates.
        
        Args:
            svg_x (float): X-coordinate in SVG space (0 -> width)
            svg_y (float): Y-coordinate in SVG space (0 -> height)
            
        Returns:
            tuple: (x, y, z) coordinates in robot space
        """
        # Convert SVG coordinates to normalized coordinates (0-1)
        norm_y = min(max(svg_y / self.svg_height, 0), 1)  # clamp to [0,1]
        norm_x = min(max(svg_x / self.svg_width, 0), 1)   # clamp to [0,1]
        
        # Map to robot coordinates
        # Note the flip: SVG X -> robot Y, and SVG Y -> robot Z
        robot_y = self.y_min + (1 - norm_x) * (self.y_max - self.y_min)
        robot_z = self.z_min + norm_y * (self.z_max - self.z_min)
        
        # Calculate corresponding x coordinate based on z
        robot_x = self.x_from_z(robot_z)
        
        return (robot_x, robot_y, robot_z)

    def map_robot_to_svg(self, robot_x, robot_y, robot_z):
        """
        Convert robot coordinates back to SVG coordinates.
        Useful for debugging or closing paths.
        """
        norm_z = (robot_z - self.z_min) / (self.z_max - self.z_min)
        norm_y = 1 - ((robot_y - self.y_min) / (self.y_max - self.y_min))
        
        svg_y = norm_z * self.svg_height
        svg_x = norm_y * self.svg_width
        
        return (svg_x, svg_y)

    def approach_canvas(self, x, y, z, roll=-180, speed=20):
        """
        Approach the canvas with the end effector, with safety checks.
        Moves first to x - off_canvas_distance, then onto the canvas.
        """
        # Ensure coordinates are within safe bounds
        y = max(min(y, self.y_max), self.y_min)
        z = max(min(z, self.z_max), self.z_min)
        x = self.x_from_z(z)
        
        print(f"Moving to position: ({x:.2f}, {y:.2f}, {z:.2f})")
        
        # Move to a position slightly away from the canvas
        approach_x = x - self.off_canvas_distance
        print(f"Approach position: ({approach_x:.2f}, {y:.2f}, {z:.2f})")
        result = self.arm.set_position(x=approach_x, y=y, z=z, roll=roll, pitch=-90, yaw=0, speed=speed, wait=True)
        print(f"Approach movement result: {result}")
        
        # Then move onto the canvas
        result = self.arm.set_position(x=x, y=y, z=z, roll=roll, pitch=-90, yaw=0, speed=speed, wait=True)
        print(f"Canvas contact movement result: {result}")
        
        time.sleep(0.2)  # short pause

    def retreat_from_canvas(self, x, y, z, speed=20):
        """Lift straight back along X only, keeping the actual Y and Z."""
        # keep Y and Z exactly as they are
        y = max(min(y, self.y_max), self.y_min)
        z = max(min(z, self.z_max), self.z_min)
        print('retreat, not down?')
        # move off the canvas by off_canvas_distance
        retreat_x = x - self.off_canvas_distance

        self.arm.set_position(x=retreat_x, y=y, z=z,
                            roll=-180, pitch=-90, yaw=0,
                            speed=speed, wait=True)
        time.sleep(0.2)

    def parse_path(self, path_data):
        """
        Parse an SVG path data string to extract commands and parameters.
        Returns a list of (command, [params]) tuples.
        """
        command_pattern = r'([MLHVCSQTAZmlhvcsqtaz])[\s,]*([^MLHVCSQTAZmlhvcsqtaz]*)'
        number_pattern = r'(-?\d+\.?\d*|\.\d+)'
        
        path_commands = []
        matches = re.finditer(command_pattern, path_data)
        
        for match in matches:
            cmd = match.group(1)
            params_str = match.group(2).strip()
            params = re.findall(number_pattern, params_str)
            params = [float(p) for p in params]
            path_commands.append((cmd, params))
            
        return path_commands

    def draw_svg_path(self, path_data, speed=20):
        """
        Draw a single SVG path on the canvas.

        Args:
            path_data (str): SVG path data (‘d’ attribute)
            speed (int): Drawing speed in % of the controller scale
        """
        path_commands = self.parse_path(path_data)

        current_point = None               # current point in SVG coordinates
        subpath_start_point = None         # first robot point of the current sub-path
        last_robot_pos = None              # last robot (x, y, z) actually reached
        last_control_point = None          # for smooth Béziers

        for cmd, params in path_commands:
            cmd_lower = cmd.lower()
            relative = cmd.islower()

            # ---------------------------------------------------------  M / m
            if cmd_lower == 'm':
                # lift from the previous sub-path (if any)
                if last_robot_pos is not None:
                    self.retreat_from_canvas(*last_robot_pos, speed=speed)

                # ‘M’ may contain multiple coordinate pairs
                for i in range(0, len(params), 2):
                    if i + 1 >= len(params):
                        break
                    x, y = params[i:i + 2]
                    if relative and current_point:
                        x += current_point[0]
                        y += current_point[1]
                    current_point = (x, y)

                    rx, ry, rz = self.map_svg_to_robot(x, y)
                    self.approach_canvas(rx, ry, rz, speed=speed)

                    subpath_start_point = last_robot_pos = (rx, ry, rz)

            # ---------------------------------------------------------  L / l
            elif cmd_lower == 'l':
                for i in range(0, len(params), 2):
                    if i + 1 >= len(params):
                        break
                    x, y = params[i:i + 2]
                    if relative and current_point:
                        x += current_point[0]
                        y += current_point[1]
                    current_point = (x, y)

                    rx, ry, rz = self.map_svg_to_robot(x, y)
                    ry = max(min(ry, self.y_max), self.y_min)
                    rz = max(min(rz, self.z_max), self.z_min)
                    rx = self.x_from_z(rz)

                    self.arm.set_position(x=rx, y=ry, z=rz,
                                        roll=-180, pitch=-90, yaw=0,
                                        speed=speed, wait=True)
                    last_robot_pos = (rx, ry, rz)

            # ---------------------------------------------------------  H / h
            elif cmd_lower == 'h':
                for x in params:
                    if relative and current_point:
                        x += current_point[0]
                    current_point = (x, current_point[1])

                    rx, ry, rz = self.map_svg_to_robot(x, current_point[1])
                    ry = max(min(ry, self.y_max), self.y_min)
                    rz = max(min(rz, self.z_max), self.z_min)
                    rx = self.x_from_z(rz)

                    self.arm.set_position(x=rx, y=ry, z=rz,
                                        roll=-180, pitch=-90, yaw=0,
                                        speed=speed, wait=True)
                    last_robot_pos = (rx, ry, rz)

            # ---------------------------------------------------------  V / v
            elif cmd_lower == 'v':
                for y in params:
                    if relative and current_point:
                        y += current_point[1]
                    current_point = (current_point[0], y)

                    rx, ry, rz = self.map_svg_to_robot(current_point[0], y)
                    ry = max(min(ry, self.y_max), self.y_min)
                    rz = max(min(rz, self.z_max), self.z_min)
                    rx = self.x_from_z(rz)

                    self.arm.set_position(x=rx, y=ry, z=rz,
                                        roll=-180, pitch=-90, yaw=0,
                                        speed=speed, wait=True)
                    last_robot_pos = (rx, ry, rz)

            # ---------------------------------------------------------  Q / q
            elif cmd_lower == 'q' and current_point is not None:
                if len(params) >= 4:
                    cx, cy, x, y = params[:4]
                    if relative:
                        cx += current_point[0]
                        cy += current_point[1]
                        x += current_point[0]
                        y += current_point[1]

                    last_control_point = (cx, cy)
                    start_x, start_y = current_point
                    steps = self.bezier_steps

                    for t in np.linspace(0, 1, steps):
                        px = (1 - t) ** 2 * start_x + 2 * t * (1 - t) * cx + t ** 2 * x
                        py = (1 - t) ** 2 * start_y + 2 * t * (1 - t) * cy + t ** 2 * y

                        rx, ry, rz = self.map_svg_to_robot(px, py)
                        ry = max(min(ry, self.y_max), self.y_min)
                        rz = max(min(rz, self.z_max), self.z_min)
                        rx = self.x_from_z(rz)

                        self.arm.set_position(x=rx, y=ry, z=rz,
                                            roll=-180, pitch=-90, yaw=0,
                                            speed=speed, wait=True)
                        last_robot_pos = (rx, ry, rz)

                    current_point = (x, y)

            # ---------------------------------------------------------  T / t
            elif cmd_lower == 't' and current_point is not None:
                if len(params) >= 2:
                    x, y = params[:2]
                    if relative:
                        x += current_point[0]
                        y += current_point[1]

                    if last_control_point:
                        cx = 2 * current_point[0] - last_control_point[0]
                        cy = 2 * current_point[1] - last_control_point[1]
                    else:
                        cx, cy = current_point

                    last_control_point = (cx, cy)
                    start_x, start_y = current_point
                    steps = self.bezier_steps

                    for t in np.linspace(0, 1, steps):
                        px = (1 - t) ** 2 * start_x + 2 * t * (1 - t) * cx + t ** 2 * x
                        py = (1 - t) ** 2 * start_y + 2 * t * (1 - t) * cy + t ** 2 * y

                        rx, ry, rz = self.map_svg_to_robot(px, py)
                        ry = max(min(ry, self.y_max), self.y_min)
                        rz = max(min(rz, self.z_max), self.z_min)
                        rx = self.x_from_z(rz)

                        self.arm.set_position(x=rx, y=ry, z=rz,
                                            roll=-180, pitch=-90, yaw=0,
                                            speed=speed, wait=True)
                        last_robot_pos = (rx, ry, rz)

                    current_point = (x, y)

            # ---------------------------------------------------------  C / c
            elif cmd_lower == 'c' and current_point is not None:
                if len(params) >= 6:
                    c1x, c1y, c2x, c2y, x, y = params[:6]
                    if relative:
                        c1x += current_point[0]
                        c1y += current_point[1]
                        c2x += current_point[0]
                        c2y += current_point[1]
                        x += current_point[0]
                        y += current_point[1]

                    last_control_point = (c2x, c2y)
                    start_x, start_y = current_point
                    steps = self.bezier_steps

                    for t in np.linspace(0, 1, steps):
                        px = ((1 - t) ** 3 * start_x +
                            3 * t * (1 - t) ** 2 * c1x +
                            3 * t ** 2 * (1 - t) * c2x +
                            t ** 3 * x)
                        py = ((1 - t) ** 3 * start_y +
                            3 * t * (1 - t) ** 2 * c1y +
                            3 * t ** 2 * (1 - t) * c2y +
                            t ** 3 * y)

                        rx, ry, rz = self.map_svg_to_robot(px, py)
                        ry = max(min(ry, self.y_max), self.y_min)
                        rz = max(min(rz, self.z_max), self.z_min)
                        rx = self.x_from_z(rz)

                        self.arm.set_position(x=rx, y=ry, z=rz,
                                            roll=-180, pitch=-90, yaw=0,
                                            speed=speed, wait=True)
                        last_robot_pos = (rx, ry, rz)

                    current_point = (x, y)

            # ---------------------------------------------------------  S / s
            elif cmd_lower == 's' and current_point is not None:
                if len(params) >= 4:
                    c2x, c2y, x, y = params[:4]
                    if relative:
                        c2x += current_point[0]
                        c2y += current_point[1]
                        x += current_point[0]
                        y += current_point[1]

                    if last_control_point:
                        c1x = 2 * current_point[0] - last_control_point[0]
                        c1y = 2 * current_point[1] - last_control_point[1]
                    else:
                        c1x, c1y = current_point

                    last_control_point = (c2x, c2y)
                    start_x, start_y = current_point
                    steps = self.bezier_steps

                    for t in np.linspace(0, 1, steps):
                        px = ((1 - t) ** 3 * start_x +
                            3 * t * (1 - t) ** 2 * c1x +
                            3 * t ** 2 * (1 - t) * c2x +
                            t ** 3 * x)
                        py = ((1 - t) ** 3 * start_y +
                            3 * t * (1 - t) ** 2 * c1y +
                            3 * t ** 2 * (1 - t) * c2y +
                            t ** 3 * y)

                        rx, ry, rz = self.map_svg_to_robot(px, py)
                        ry = max(min(ry, self.y_max), self.y_min)
                        rz = max(min(rz, self.z_max), self.z_min)
                        rx = self.x_from_z(rz)

                        self.arm.set_position(x=rx, y=ry, z=rz,
                                            roll=-180, pitch=-90, yaw=0,
                                            speed=speed, wait=True)
                        last_robot_pos = (rx, ry, rz)

                    current_point = (x, y)

            # ---------------------------------------------------------  A / a
            elif cmd_lower == 'a' and current_point is not None:
                # still approximated as a straight line
                if len(params) >= 7:
                    x, y = params[5], params[6]
                    if relative:
                        x += current_point[0]
                        y += current_point[1]
                    current_point = (x, y)

                    rx, ry, rz = self.map_svg_to_robot(x, y)
                    ry = max(min(ry, self.y_max), self.y_min)
                    rz = max(min(rz, self.z_max), self.z_min)
                    rx = self.x_from_z(rz)

                    self.arm.set_position(x=rx, y=ry, z=rz,
                                        roll=-180, pitch=-90, yaw=0,
                                        speed=speed, wait=True)
                    last_robot_pos = (rx, ry, rz)

            # ---------------------------------------------------------  Z / z
            elif cmd_lower == 'z':
                if subpath_start_point:
                    self.arm.set_position(x=subpath_start_point[0],
                                        y=subpath_start_point[1],
                                        z=subpath_start_point[2],
                                        roll=-180, pitch=-90, yaw=0,
                                        speed=speed, wait=True)
                    last_robot_pos = subpath_start_point
                    current_point = None
                    subpath_start_point = None

        # -------- end for -------------------------------------------------------

        # lift away from the canvas where we actually ended
        if last_robot_pos:
            self.retreat_from_canvas(*last_robot_pos, speed=speed)

        return True


    def rect_to_path(self, rect):
        """Convert a rectangle element to path data."""
        try:
            x = float(rect.attrib.get('x', 0))
            y = float(rect.attrib.get('y', 0))
            width = float(rect.attrib.get('width', 0))
            height = float(rect.attrib.get('height', 0))
            return f"M{x},{y} h{width} v{height} h{-width} Z"
        except (ValueError, TypeError) as e:
            print(f"Error converting rectangle to path: {e}")
            return None

    def circle_to_path(self, circle):
        """Convert a circle element to path data (approx. 4 Bezier arcs)."""
        try:
            cx = float(circle.attrib.get('cx', 0))
            cy = float(circle.attrib.get('cy', 0))
            r = float(circle.attrib.get('r', 0))
            control = r * 0.5522848  # approximation
            
            path_data = (
                f"M{cx},{cy-r} "
                f"C{cx+control},{cy-r} {cx+r},{cy-control} {cx+r},{cy} "
                f"C{cx+r},{cy+control} {cx+control},{cy+r} {cx},{cy+r} "
                f"C{cx-control},{cy+r} {cx-r},{cy+control} {cx-r},{cy} "
                f"C{cx-r},{cy-control} {cx-control},{cy-r} {cx},{cy-r} "
                f"Z"
            )
            return path_data
        except (ValueError, TypeError) as e:
            print(f"Error converting circle to path: {e}")
            return None

    def ellipse_to_path(self, ellipse):
        """Convert an ellipse element to path data (approx. 4 Bezier arcs)."""
        try:
            cx = float(ellipse.attrib.get('cx', 0))
            cy = float(ellipse.attrib.get('cy', 0))
            rx = float(ellipse.attrib.get('rx', 0))
            ry = float(ellipse.attrib.get('ry', 0))
            
            control_x = rx * 0.5522848
            control_y = ry * 0.5522848
            
            path_data = (
                f"M{cx},{cy-ry} "
                f"C{cx+control_x},{cy-ry} {cx+rx},{cy-control_y} {cx+rx},{cy} "
                f"C{cx+rx},{cy+control_y} {cx+control_x},{cy+ry} {cx},{cy+ry} "
                f"C{cx-control_x},{cy+ry} {cx-rx},{cy+control_y} {cx-rx},{cy} "
                f"C{cx-rx},{cy-control_y} {cx-control_x},{cy-ry} {cx},{cy-ry} "
                f"Z"
            )
            return path_data
        except (ValueError, TypeError) as e:
            print(f"Error converting ellipse to path: {e}")
            return None

    def line_to_path(self, line):
        """Convert a line element to path data."""
        try:
            x1 = float(line.attrib.get('x1', 0))
            y1 = float(line.attrib.get('y1', 0))
            x2 = float(line.attrib.get('x2', 0))
            y2 = float(line.attrib.get('y2', 0))
            return f"M{x1},{y1} L{x2},{y2}"
        except (ValueError, TypeError) as e:
            print(f"Error converting line to path: {e}")
            return None

    def polyline_to_path(self, polyline):
        """Convert a polyline element to path data."""
        try:
            points = polyline.attrib.get('points', '')
            if not points:
                return None
            pairs = re.findall(r'(-?\d+\.?\d*)[,\s](-?\d+\.?\d*)', points)
            if not pairs:
                return None
            path_data = f"M{pairs[0][0]},{pairs[0][1]}"
            for x, y in pairs[1:]:
                path_data += f" L{x},{y}"
            return path_data
        except (ValueError, TypeError) as e:
            print(f"Error converting polyline to path: {e}")
            return None

    def polygon_to_path(self, polygon):
        """Convert a polygon element to a closed path."""
        try:
            path_data = self.polyline_to_path(polygon)
            if path_data:
                path_data += " Z"
            return path_data
        except (ValueError, TypeError) as e:
            print(f"Error converting polygon to path: {e}")
            return None

    def parse_svg_file(self, svg_file_path):
        """
        Parse an SVG file and extract all drawable elements (paths, rects, circles, etc.),
        converting them to path data.
        """
        try:
            tree = ET.parse(svg_file_path)
            root = tree.getroot()
            
            # Get SVG dimensions if specified
            width = root.attrib.get('width')
            height = root.attrib.get('height')
            
            if width:
                self.svg_width = float(width.replace('px', ''))
            if height:
                self.svg_height = float(height.replace('px', ''))
            
            print(f"SVG dimensions: {self.svg_width} x {self.svg_height}")
            
            paths = []
            namespaces = {'svg': 'http://www.w3.org/2000/svg'}
            
            def find_elements(el, tag):
                # Attempt with namespace first
                found = el.findall(f'.//svg:{tag}', namespaces)
                if not found:
                    # Fallback to no namespace
                    found = el.findall(f'.//{tag}')
                return found
            
            # Grab existing <path> elements
            for p in find_elements(root, 'path'):
                d = p.attrib.get('d')
                if d:
                    paths.append(d)
            
            # Rects
            for rect in find_elements(root, 'rect'):
                rect_path = self.rect_to_path(rect)
                if rect_path:
                    paths.append(rect_path)
            
            # Circles
            for circle in find_elements(root, 'circle'):
                circle_path = self.circle_to_path(circle)
                if circle_path:
                    paths.append(circle_path)
            
            # Ellipses
            for ellipse in find_elements(root, 'ellipse'):
                ellipse_path = self.ellipse_to_path(ellipse)
                if ellipse_path:
                    paths.append(ellipse_path)
            
            # Lines
            for line in find_elements(root, 'line'):
                line_path = self.line_to_path(line)
                if line_path:
                    paths.append(line_path)
            
            # Polylines
            for polyline in find_elements(root, 'polyline'):
                polyline_path = self.polyline_to_path(polyline)
                if polyline_path:
                    paths.append(polyline_path)
            
            # Polygons
            for polygon in find_elements(root, 'polygon'):
                polygon_path = self.polygon_to_path(polygon)
                if polygon_path:
                    paths.append(polygon_path)
            
            print(f"Found {len(paths)} drawable elements in SVG.")
            return paths
        
        except Exception as e:
            print(f"Error parsing SVG file: {e}")
            return []

    def paint_svg_from_string(self, svg_string, speed=20):
        """
        Paint an SVG from a string content.
        
        Args:
            svg_string (str): SVG content as a string
            speed (int): Drawing speed
        """
        try:
            print("Starting to paint SVG from string")
            
            # Parse SVG string
            root = ET.fromstring(svg_string)
            
            # Get SVG dimensions
            width = root.attrib.get('width')
            height = root.attrib.get('height')
            
            if width:
                self.svg_width = float(width.replace('px', ''))
            if height:
                self.svg_height = float(height.replace('px', ''))
            
            print(f"SVG dimensions: {self.svg_width} x {self.svg_height}")
            
            # Try both with and without namespace
            namespaces = {'svg': 'http://www.w3.org/2000/svg'}
            
            def find_elements(el, tag):
                elems = el.findall(f'.//svg:{tag}', namespaces)
                if not elems:
                    elems = el.findall(f'.//{tag}')
                return elems
            
            paths = []
            
            # <path>
            for path_el in find_elements(root, 'path'):
                d = path_el.attrib.get('d')
                if d:
                    paths.append(d)
            
            # <rect>
            for rect in find_elements(root, 'rect'):
                rect_path = self.rect_to_path(rect)
                if rect_path:
                    paths.append(rect_path)
            
            # <circle>
            for circle in find_elements(root, 'circle'):
                circle_path = self.circle_to_path(circle)
                if circle_path:
                    paths.append(circle_path)
            
            # <ellipse>
            for ellipse in find_elements(root, 'ellipse'):
                ellipse_path = self.ellipse_to_path(ellipse)
                if ellipse_path:
                    paths.append(ellipse_path)
            
            # <line>
            for line in find_elements(root, 'line'):
                line_path = self.line_to_path(line)
                if line_path:
                    paths.append(line_path)
            
            # <polyline>
            for polyline in find_elements(root, 'polyline'):
                polyline_path = self.polyline_to_path(polyline)
                if polyline_path:
                    paths.append(polyline_path)
            
            # <polygon>
            for polygon in find_elements(root, 'polygon'):
                polygon_path = self.polygon_to_path(polygon)
                if polygon_path:
                    paths.append(polygon_path)
            
            if not paths:
                print("No drawable elements found in SVG string.")
                return False
            
            print(f"Found {len(paths)} drawable elements in the SVG string.")
            
            # Move to a safe position first
            safe_z = (self.z_min + self.z_max) / 2
            safe_y = (self.y_min + self.y_max) / 2
            safe_x = self.x_from_z(safe_z) - self.off_canvas_distance
            print(f"Moving to initial safety position: ({safe_x:.2f}, {safe_y:.2f}, {safe_z:.2f})")
            
            result = self.arm.set_position(
                x=safe_x, y=safe_y, z=safe_z, 
                roll=-180, pitch=-90, yaw=0, 
                speed=speed, wait=True
            )
            if result != 0:
                print(f"Warning: Initial safety move resulted in code {result}")
            
            # Draw each path
            for i, path_data in enumerate(paths):
                print(f"Drawing path {i+1}/{len(paths)}")
                success = self.draw_svg_path(path_data, speed)
                if not success:
                    print(f"Warning: Problem drawing path {i+1}")
                time.sleep(0.5)
            
            # Return to safe position
            print("Finished painting, returning to safe position.")
            self.arm.set_position(
                x=safe_x, y=safe_y, z=safe_z,
                roll=-180, pitch=-90, yaw=0,
                speed=speed, wait=True
            )
            
            return True

        except Exception as e:
            print(f"Error during painting from string: {e}")
            # Attempt to recover
            try:
                print("Attempting to move to a safe position...")
                self.arm.set_position(x=400, y=0, z=400, roll=-180, pitch=-90, yaw=0, speed=20, wait=True)
            except:
                pass
            # Emergency stop if recovery failed
            print("Executing emergency stop")
            self.arm.emergency_stop()
            return False

    def paint_svg(self, svg_file_path, speed=20):
        """
        Paint an SVG file on the canvas.
        
        Args:
            svg_file_path (str): Path to the SVG file
            speed (int): Drawing speed
        """
        try:
            print(f"Starting to paint SVG file: {svg_file_path}")
            state = self.arm.get_state()
            print(f"Robot state: {state}")
            if state[0] != 0:
                print("Warning: Robot is not in a ready state, attempting to reset.")
                self.arm.set_state(state=0)
                time.sleep(1)
            
            paths = self.parse_svg_file(svg_file_path)
            if not paths:
                print("No drawable elements found in SVG file.")
                return False
            
            # Move to a safe position
            safe_z = (self.z_min + self.z_max) / 2
            safe_y = (self.y_min + self.y_max) / 2
            safe_x = self.x_from_z(safe_z) - self.off_canvas_distance
            
            print(f"Moving to initial safety position: ({safe_x:.2f}, {safe_y:.2f}, {safe_z:.2f})")
            result = self.arm.set_position(
                x=safe_x, y=safe_y, z=safe_z,
                roll=-180, pitch=-90, yaw=0,
                speed=speed, wait=True
            )
            if result != 0:
                print(f"Warning: Initial safety move resulted in code {result}")
            
            # Draw each path
            for i, path_data in enumerate(paths):
                print(f"Drawing path {i+1}/{len(paths)}")
                success = self.draw_svg_path(path_data, speed)
                if not success:
                    print(f"Warning: Problem drawing path {i+1}")
                time.sleep(0.5)
            
            # Return to safe position
            print("Finished painting, returning to safe position.")
            self.arm.set_position(
                x=safe_x, y=safe_y, z=safe_z,
                roll=-180, pitch=-90, yaw=0,
                speed=speed, wait=True
            )
            
            return True

        except Exception as e:
            print(f"Error during painting: {e}")
            # Attempt to recover
            try:
                print("Attempting to move to a safe position...")
                self.arm.set_position(x=400, y=0, z=400, roll=-180, pitch=-90, yaw=0, speed=20, wait=True)
            except:
                pass
            # Emergency stop if needed
            print("Executing emergency stop")
            self.arm.emergency_stop()
            return False


if __name__ == "__main__":
    svg_file = "simple.svg"  # Default SVG file
    ip_address = "192.168.0.203"  # Default IP address
    speed = 15  # Default speed
    
    if len(sys.argv) > 1:
        svg_file = sys.argv[1]
    if len(sys.argv) > 2:
        ip_address = sys.argv[2]
    if len(sys.argv) > 3:
        try:
            speed = int(sys.argv[3])
        except ValueError:
            print(f"Invalid speed value, using default: {speed}")
    
    # Check if SVG file exists
    if not os.path.exists(svg_file):
        print(f"Error: SVG file '{svg_file}' not found.")
        sys.exit(1)
    
    print(f"Connecting to xArm at {ip_address}...")
    painter = SVGPainter(ip_address)
    
    try:
        print("Testing robot connection and movement...")
        # Move to a safe test position
        result = painter.arm.set_position(
            x=400, y=0, z=400, roll=-180, pitch=-90, yaw=0,
            speed=20, wait=True
        )
        print(f"Initial movement test result: {result}")
        time.sleep(1)
        
        # Start painting
        print(f"\n=== Starting to paint {svg_file} ===")
        print(f"Using drawing speed: {speed}")
        success = painter.paint_svg(svg_file, speed=speed)
        
        if success:
            print("\n=== Painting completed successfully ===")
        else:
            print("\n=== Painting encountered issues ===")
            
    except KeyboardInterrupt:
        print("\nOperation interrupted by user.")
    except Exception as e:
        print(f"Error during execution: {e}")
    finally:
        painter.disconnect()
