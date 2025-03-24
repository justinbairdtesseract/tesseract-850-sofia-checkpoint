from xarm.wrapper import XArmAPI
import xml.etree.ElementTree as ET
import re
import math
import time
import numpy as np

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
        self.bezier_steps = 15  # Increased from 10 for smoother curves
        
        # Namespace for SVG parsing
        self.ns = {'svg': 'http://www.w3.org/2000/svg'}

    def setup(self):
        """Initialize the robot arm."""
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)
        
        # Lower speed for initial setup to ensure safety
        self.arm.set_servo_angle(servo_id=5, angle=90, speed=20, wait=True)
        
        # Enable collision detection (if supported by your model)
        self.arm.set_collision_sensitivity(1)
        
        # Wait for robot to be ready
        time.sleep(1)

    def calculate_scaling_factors(self):
        """Calculate scaling factors for the robot based on the canvas corners."""
        z1, x1 = self.canvas_corners[0][2], self.canvas_corners[0][0]  # Top left corner
        z2, x2 = self.canvas_corners[2][2], self.canvas_corners[2][0]  # Bottom left corner
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
            svg_x (float): X-coordinate in SVG space (0-width)
            svg_y (float): Y-coordinate in SVG space (0-height)
            
        Returns:
            tuple: (x, y, z) coordinates in robot space
        """
        # Map SVG coordinates to robot coordinates
        # SVG origin is top-left, robot canvas has different orientation
        
        # Convert SVG coordinates to normalized coordinates (0-1)
        norm_y = min(max(svg_y / self.svg_height, 0), 1)  # Clamp to [0,1]
        norm_x = min(max(svg_x / self.svg_width, 0), 1)   # Clamp to [0,1]
        
        # Map to robot coordinates
        robot_y = self.y_min + (1 - norm_x) * (self.y_max - self.y_min)  # SVG x maps to robot y (flipped)
        robot_z = self.z_min + norm_y * (self.z_max - self.z_min)  # SVG y maps to robot z
        
        # Calculate corresponding x coordinate based on z (to maintain contact with the canvas)
        robot_x = self.x_from_z(robot_z)
        
        return (robot_x, robot_y, robot_z)

    def approach_canvas(self, x, y, z, roll=-180, speed=20):
        """Approach the canvas with the end effector, with safety checks."""
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
        
        # Then move onto canvas
        result = self.arm.set_position(x=x, y=y, z=z, roll=roll, pitch=-90, yaw=0, speed=speed, wait=True)
        print(f"Canvas contact movement result: {result}")
        
        # Short pause to ensure proper contact
        time.sleep(0.2)

    def retreat_from_canvas(self, x, y, z, speed=20):
        """Retreat from the canvas after drawing, with safety checks."""
        # Ensure coordinates are within safe bounds
        y = max(min(y, self.y_max), self.y_min)
        z = max(min(z, self.z_max), self.z_min)
        x = self.x_from_z(z)  # Recalculate x to ensure it's on the canvas plane
        
        retreat_x = x - self.off_canvas_distance
        self.arm.set_position(x=retreat_x, y=y, z=z, roll=-180, pitch=-90, yaw=0, speed=speed, wait=True)
        
        # Short pause to ensure proper lift
        time.sleep(0.2)

    def parse_path(self, path_data):
        """
        Parse SVG path data to extract points.
        
        Args:
            path_data (str): SVG path data string
            
        Returns:
            list: List of commands and their parameters
        """
        # Regular expressions to match path commands and parameters
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
        Draw an SVG path on the canvas with safety checks.
        
        Args:
            path_data (str): SVG path data
            speed (int): Drawing speed
            
        Returns:
            bool: True if successful, False otherwise
        """
        path_commands = self.parse_path(path_data)
        
        current_point = None  # Current position in SVG coordinates
        start_point = None    # Start position in robot coordinates
        
        # Last control point for smooth curves
        last_control_point = None
        
        for cmd, params in path_commands:
            cmd_lower = cmd.lower()
            
            # Check if this is a relative or absolute command
            relative = cmd.islower()
            
            if cmd_lower == 'm':  # Move
                if len(params) >= 2:
                    x, y = params[0], params[1]
                    if relative and current_point:
                        x += current_point[0]
                        y += current_point[1]
                    
                    # Update the current SVG point
                    current_point = (x, y)
                    
                    # Convert SVG coordinates to robot coordinates
                    robot_x, robot_y, robot_z = self.map_svg_to_robot(x, y)
                    
                    # If this is the first point, we need to approach the canvas
                    if start_point is None:
                        self.approach_canvas(robot_x, robot_y, robot_z, speed=speed)
                    else:
                        # Lift pen slightly, move, and lower again
                        self.retreat_from_canvas(start_point[0], start_point[1], start_point[2], speed=speed)
                        self.approach_canvas(robot_x, robot_y, robot_z, speed=speed)
                    
                    start_point = (robot_x, robot_y, robot_z)
                    
                    # Handle polyline (multiple points after M)
                    for i in range(2, len(params), 2):
                        if i+1 < len(params):
                            x, y = params[i], params[i+1]
                            if relative:
                                x += current_point[0]
                                y += current_point[1]
                            current_point = (x, y)
                            robot_x, robot_y, robot_z = self.map_svg_to_robot(x, y)
                            self.arm.set_position(x=robot_x, y=robot_y, z=robot_z, roll=-180, pitch=-90, yaw=0, speed=speed, wait=True)
                    
            elif cmd_lower == 'l':  # Line
                for i in range(0, len(params), 2):
                    if i+1 < len(params):
                        x, y = params[i], params[i+1]
                        if relative and current_point:
                            x += current_point[0]
                            y += current_point[1]
                        
                        current_point = (x, y)
                        robot_x, robot_y, robot_z = self.map_svg_to_robot(x, y)
                        
                        # Safety check - ensure robot stays within bounds
                        robot_y = max(min(robot_y, self.y_max), self.y_min)
                        robot_z = max(min(robot_z, self.z_max), self.z_min)
                        robot_x = self.x_from_z(robot_z)
                        
                        self.arm.set_position(x=robot_x, y=robot_y, z=robot_z, roll=-180, pitch=-90, yaw=0, speed=speed, wait=True)
            
            elif cmd_lower == 'h':  # Horizontal line
                for x in params:
                    if relative and current_point:
                        x += current_point[0]
                    
                    current_point = (x, current_point[1])
                    robot_x, robot_y, robot_z = self.map_svg_to_robot(x, current_point[1])
                    
                    # Safety check
                    robot_y = max(min(robot_y, self.y_max), self.y_min)
                    robot_z = max(min(robot_z, self.z_max), self.z_min)
                    robot_x = self.x_from_z(robot_z)
                    
                    self.arm.set_position(x=robot_x, y=robot_y, z=robot_z, roll=-180, pitch=-90, yaw=0, speed=speed, wait=True)
            
            elif cmd_lower == 'v':  # Vertical line
                for y in params:
                    if relative and current_point:
                        y += current_point[1]
                    
                    current_point = (current_point[0], y)
                    robot_x, robot_y, robot_z = self.map_svg_to_robot(current_point[0], y)
                    
                    # Safety check
                    robot_y = max(min(robot_y, self.y_max), self.y_min)
                    robot_z = max(min(robot_z, self.z_max), self.z_min)
                    robot_x = self.x_from_z(robot_z)
                    
                    self.arm.set_position(x=robot_x, y=robot_y, z=robot_z, roll=-180, pitch=-90, yaw=0, speed=speed, wait=True)
            
            elif cmd_lower == 'q':  # Quadratic Bezier
                if len(params) >= 4:
                    # Get control point and end point
                    cx, cy = params[0], params[1]
                    x, y = params[2], params[3]
                    
                    if relative and current_point:
                        cx += current_point[0]
                        cy += current_point[1]
                        x += current_point[0]
                        y += current_point[1]
                    
                    # Store this control point for future smooth curves
                    last_control_point = (cx, cy)
                    
                    # Interpolate the quadratic Bezier curve
                    steps = self.bezier_steps
                    start_x, start_y = current_point
                    
                    for t in np.linspace(0, 1, steps):
                        # Quadratic Bezier formula: B(t) = (1-t)²*P0 + 2t(1-t)*P1 + t²*P2
                        px = (1-t)**2 * start_x + 2*t*(1-t)*cx + t**2*x
                        py = (1-t)**2 * start_y + 2*t*(1-t)*cy + t**2*y
                        
                        robot_x, robot_y, robot_z = self.map_svg_to_robot(px, py)
                        
                        # Safety check
                        robot_y = max(min(robot_y, self.y_max), self.y_min)
                        robot_z = max(min(robot_z, self.z_max), self.z_min)
                        robot_x = self.x_from_z(robot_z)
                        
                        self.arm.set_position(x=robot_x, y=robot_y, z=robot_z, roll=-180, pitch=-90, yaw=0, speed=speed, wait=True)
                    
                    current_point = (x, y)
            
            elif cmd_lower == 't':  # Smooth quadratic Bezier
                if len(params) >= 2:
                    x, y = params[0], params[1]
                    
                    if relative and current_point:
                        x += current_point[0]
                        y += current_point[1]
                    
                    # Calculate the reflection of the last control point
                    if last_control_point and current_point:
                        cx = 2 * current_point[0] - last_control_point[0]
                        cy = 2 * current_point[1] - last_control_point[1]
                    else:
                        # If no previous control point, use current point (collinear)
                        cx, cy = current_point
                    
                    # Store this control point for future smooth curves
                    last_control_point = (cx, cy)
                    
                    # Interpolate the quadratic Bezier curve
                    steps = self.bezier_steps
                    start_x, start_y = current_point
                    
                    for t in np.linspace(0, 1, steps):
                        # Quadratic Bezier formula: B(t) = (1-t)²*P0 + 2t(1-t)*P1 + t²*P2
                        px = (1-t)**2 * start_x + 2*t*(1-t)*cx + t**2*x
                        py = (1-t)**2 * start_y + 2*t*(1-t)*cy + t**2*y
                        
                        robot_x, robot_y, robot_z = self.map_svg_to_robot(px, py)
                        
                        # Safety check
                        robot_y = max(min(robot_y, self.y_max), self.y_min)
                        robot_z = max(min(robot_z, self.z_max), self.z_min)
                        robot_x = self.x_from_z(robot_z)
                        
                        self.arm.set_position(x=robot_x, y=robot_y, z=robot_z, roll=-180, pitch=-90, yaw=0, speed=speed, wait=True)
                    
                    current_point = (x, y)
            
            elif cmd_lower == 'c':  # Cubic Bezier
                if len(params) >= 6:
                    # Get control points and end point
                    c1x, c1y = params[0], params[1]
                    c2x, c2y = params[2], params[3]
                    x, y = params[4], params[5]
                    
                    if relative and current_point:
                        c1x += current_point[0]
                        c1y += current_point[1]
                        c2x += current_point[0]
                        c2y += current_point[1]
                        x += current_point[0]
                        y += current_point[1]
                    
                    # Store the second control point for future smooth curves
                    last_control_point = (c2x, c2y)
                    
                    # Interpolate the cubic Bezier curve
                    steps = self.bezier_steps
                    start_x, start_y = current_point
                    
                    for t in np.linspace(0, 1, steps):
                        # Cubic Bezier formula: B(t) = (1-t)³*P0 + 3t(1-t)²*P1 + 3t²(1-t)*P2 + t³*P3
                        px = (1-t)**3 * start_x + 3*t*(1-t)**2 * c1x + 3*t**2*(1-t) * c2x + t**3 * x
                        py = (1-t)**3 * start_y + 3*t*(1-t)**2 * c1y + 3*t**2*(1-t) * c2y + t**3 * y
                        
                        robot_x, robot_y, robot_z = self.map_svg_to_robot(px, py)
                        
                        # Safety check
                        robot_y = max(min(robot_y, self.y_max), self.y_min)
                        robot_z = max(min(robot_z, self.z_max), self.z_min)
                        robot_x = self.x_from_z(robot_z)
                        
                        self.arm.set_position(x=robot_x, y=robot_y, z=robot_z, roll=-180, pitch=-90, yaw=0, speed=speed, wait=True)
                    
                    current_point = (x, y)
            
            elif cmd_lower == 's':  # Smooth cubic Bezier
                if len(params) >= 4:
                    # Get control points and end point
                    c2x, c2y = params[0], params[1]
                    x, y = params[2], params[3]
                    
                    if relative and current_point:
                        c2x += current_point[0]
                        c2y += current_point[1]
                        x += current_point[0]
                        y += current_point[1]
                    
                    # Calculate the reflection of the last control point
                    if last_control_point and current_point:
                        c1x = 2 * current_point[0] - last_control_point[0]
                        c1y = 2 * current_point[1] - last_control_point[1]
                    else:
                        # If no previous control point, use current point (collinear)
                        c1x, c1y = current_point
                    
                    # Store this control point for future smooth curves
                    last_control_point = (c2x, c2y)
                    
                    # Interpolate the cubic Bezier curve
                    steps = self.bezier_steps
                    start_x, start_y = current_point
                    
                    for t in np.linspace(0, 1, steps):
                        # Cubic Bezier formula: B(t) = (1-t)³*P0 + 3t(1-t)²*P1 + 3t²(1-t)*P2 + t³*P3
                        px = (1-t)**3 * start_x + 3*t*(1-t)**2 * c1x + 3*t**2*(1-t) * c2x + t**3 * x
                        py = (1-t)**3 * start_y + 3*t*(1-t)**2 * c1y + 3*t**2*(1-t) * c2y + t**3 * y
                        
                        robot_x, robot_y, robot_z = self.map_svg_to_robot(px, py)
                        
                        # Safety check
                        robot_y = max(min(robot_y, self.y_max), self.y_min)
                        robot_z = max(min(robot_z, self.z_max), self.z_min)
                        robot_x = self.x_from_z(robot_z)
                        
                        self.arm.set_position(x=robot_x, y=robot_y, z=robot_z, roll=-180, pitch=-90, yaw=0, speed=speed, wait=True)
                    
                    current_point = (x, y)
            
            elif cmd_lower == 'a':  # Elliptical arc
                # Skip for now but print a warning
                print(f"Warning: Arc commands (A/a) are not fully implemented")
                # Approximate as a line to the end point if possible
                if len(params) >= 7:
                    x, y = params[5], params[6]
                    if relative and current_point:
                        x += current_point[0]
                        y += current_point[1]
                    current_point = (x, y)
                    robot_x, robot_y, robot_z = self.map_svg_to_robot(x, y)
                    self.arm.set_position(x=robot_x, y=robot_y, z=robot_z, roll=-180, pitch=-90, yaw=0, speed=speed, wait=True)
            
            elif cmd_lower == 'z':  # Close path
                if start_point and current_point != (start_point[0], start_point[1]):
                    x, y = self.map_robot_to_svg(start_point[0], start_point[1], start_point[2])
                    current_point = (x, y)
                    self.arm.set_position(x=start_point[0], y=start_point[1], z=start_point[2], 
                                         roll=-180, pitch=-90, yaw=0, speed=speed, wait=True)
        
        # Always retreat from the canvas after drawing a path
        if start_point:
            self.retreat_from_canvas(start_point[0], start_point[1], start_point[2], speed=speed)
        
        return True

    def map_robot_to_svg(self, robot_x, robot_y, robot_z):
        """
        Convert robot coordinates back to SVG coordinates.
        Useful for debugging and path closing.
        """
        # Normalize Z
        norm_z = (robot_z - self.z_min) / (self.z_max - self.z_min)
        
        # Normalize Y (with flipping)
        norm_y = 1 - ((robot_y - self.y_min) / (self.y_max - self.y_min))
        
        # Convert back to SVG coordinates
        svg_y = norm_z * self.svg_height
        svg_x = norm_y * self.svg_width
        
        return (svg_x, svg_y)