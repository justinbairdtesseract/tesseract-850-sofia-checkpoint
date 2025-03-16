from xarm.wrapper import XArmAPI
import xml.etree.ElementTree as ET
import re
import math
import time
import numpy as np

class SVGPainter:
    def __init__(self, ip_address='192.168.0.203'):
        self.debug = True
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
        self.svg_width = 600  # Default SVG width
        self.svg_height = 600  # Default SVG height
    
        
        # Robot parameters
        self.off_canvas_distance = 25  # Distance to start away from the canvas on the x-axis
        self.safety_margin = 10  # Safety margin in mm from canvas edges
        self.calculate_scaling_factors()
        self.calculate_canvas_bounds()
        


    def setup(self):
        """Initialize the robot arm."""
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)
        self.arm.set_servo_angle(servo_id=5, angle=90, wait=True)

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
        self.z_min = min(self.canvas_corners[2][2], self.canvas_corners[0][2]) + self.safety_margin
        self.z_max = max(self.canvas_corners[2][2], self.canvas_corners[0][2]) - self.safety_margin
        
        if self.debug:
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
        # Convert SVG coordinates to normalized coordinates (0-1)
        norm_x = min(max(svg_x / self.svg_width, 0), 1)   # Clamp to [0,1]
        norm_y = min(max(svg_y / self.svg_height, 0), 1)  # Clamp to [0,1]
        
        # Map to robot coordinates - SVG coordinates need to be mapped to robot's coordinate system
        # Assuming the canvas is positioned so that SVG's top-left corresponds to robot's 
        # front-left corner (max y, min z) and bottom-right corresponds to the back-right (min y, max z)
        
        robot_y = self.y_max - norm_x * (self.y_max - self.y_min)  # SVG x → robot y (right is negative in robot space)
        robot_z = self.z_min + norm_y * (self.z_max - self.z_min)  # SVG y → robot z (down is increasing in robot space)
        
        # Calculate corresponding x coordinate based on z (to maintain contact with the canvas)
        robot_x = self.x_from_z(robot_z)
        
        if self.debug:
            print(f"SVG ({svg_x}, {svg_y}) → Robot ({robot_x:.2f}, {robot_y:.2f}, {robot_z:.2f})")
            
        return (robot_x, robot_y, robot_z)

    def approach_canvas(self, x, y, z, roll=-180, speed=50):
        """Approach the canvas with the end effector, with safety checks."""
        # Ensure coordinates are within safe bounds
        y = max(min(y, self.y_max), self.y_min)
        z = max(min(z, self.z_max), self.z_min)
        x = self.x_from_z(z)
        
        if self.debug:
            print(f"Approaching canvas at: ({x:.2f}, {y:.2f}, {z:.2f})")
        
        # Move to a position slightly away from the canvas
        approach_x = x - self.off_canvas_distance
        result = self.arm.set_position(x=approach_x, y=y, z=z, roll=roll, pitch=-90, yaw=0, speed=speed, wait=True)
        
        if result[0] != 0:
            print(f"Warning: Approach movement error: {result}")
        
        # Then move onto canvas
        result = self.arm.set_position(x=x, y=y, z=z, roll=roll, pitch=-90, yaw=0, speed=speed, wait=True)
        
        if result[0] != 0:
            print(f"Warning: Canvas contact movement error: {result}")
            
        return result

    def retreat_from_canvas(self, x, y, z, speed=50):
        """Retreat from the canvas after drawing, with safety checks."""
        # Ensure coordinates are within safe bounds
        y = max(min(y, self.y_max), self.y_min)
        z = max(min(z, self.z_max), self.z_min)
        x = self.x_from_z(z)  # Recalculate x to ensure it's on the canvas plane
        
        if self.debug:
            print(f"Retreating from canvas at: ({x:.2f}, {y:.2f}, {z:.2f})")
        
        retreat_x = x - self.off_canvas_distance
        result = self.arm.set_position(x=retreat_x, y=y, z=z, roll=-180, pitch=-90, yaw=0, speed=speed, wait=True)
        
        if result[0] != 0:
            print(f"Warning: Retreat movement error: {result}")
            
        return result

    def parse_path(self, path_data):
        """
        Parse SVG path data to extract commands and coordinates.
        
        Args:
            path_data (str): SVG path data string
            
        Returns:
            list: List of commands and their parameters
        """
        # Regular expression for path commands and coordinates
        path_commands = []
        
        # Pattern for commands (M,L,Q,C,Z) and their parameters
        pattern = r'([MLHVCSQTAZmlhvcsqtaz])([^MLHVCSQTAZmlhvcsqtaz]*)'
        matches = re.findall(pattern, path_data)
        
        for cmd, params_str in matches:
            # Extract numerical parameters
            params = re.findall(r'(-?\d+\.?\d*|\.\d+)', params_str)
            params = [float(p) for p in params]
            path_commands.append((cmd, params))
        
        if self.debug:
            print(f"Parsed path commands: {path_commands}")
            
        return path_commands

    def draw_svg_path(self, path_data, speed=20):
        """
        Draw an SVG path on the canvas.
        
        Args:
            path_data (str): SVG path data
            speed (int): Drawing speed
            
        Returns:
            bool: True if successful, False otherwise
        """
        path_commands = self.parse_path(path_data)
        
        current_point = None  # Current point in SVG coordinates
        start_point = None  # First point in SVG coordinates (for 'Z' command)
        current_robot_point = None  # Current point in robot coordinates
        
        pen_down = False  # Track whether the pen is on the canvas or not
        
        for i, (cmd, params) in enumerate(path_commands):
            cmd_lower = cmd.lower()
            relative = cmd.islower()
            
            if self.debug:
                print(f"Processing command: {cmd} with params: {params}")
            
            if cmd_lower == 'm':  # Move
                if len(params) >= 2:
                    # Get the first point
                    x, y = params[0], params[1]
                    
                    if relative and current_point:
                        x += current_point[0]
                        y += current_point[1]
                    
                    # Update current point in SVG coordinates
                    current_point = (x, y)
                    
                    # If this is the first point in the path, save it as the start point
                    if start_point is None:
                        start_point = current_point
                    
                    # Convert to robot coordinates
                    robot_x, robot_y, robot_z = self.map_svg_to_robot(x, y)
                    
                    # Move to this point (with pen up)
                    if pen_down:
                        # If pen was down, lift it before moving
                        if current_robot_point:
                            self.retreat_from_canvas(current_robot_point[0], current_robot_point[1], current_robot_point[2], speed=speed)
                        pen_down = False
                    
                    # Approach the canvas at the new point
                    self.approach_canvas(robot_x, robot_y, robot_z, speed=speed)
                    pen_down = True
                    current_robot_point = (robot_x, robot_y, robot_z)
                    
                    # Handle implicit line-to commands that follow move-to
                    if len(params) > 2:
                        for j in range(2, len(params), 2):
                            if j+1 < len(params):
                                # These are treated as 'L' commands
                                line_x, line_y = params[j], params[j+1]
                                
                                if relative:
                                    line_x += current_point[0]
                                    line_y += current_point[1]
                                
                                current_point = (line_x, line_y)
                                robot_x, robot_y, robot_z = self.map_svg_to_robot(line_x, line_y)
                                
                                # Move to this point with pen down
                                result = self.arm.set_position(x=robot_x, y=robot_y, z=robot_z, 
                                                             roll=-180, pitch=-90, yaw=0, 
                                                             speed=speed, wait=True)
                                if result[0] != 0:
                                    print(f"Warning: Line movement error: {result}")
                                
                                current_robot_point = (robot_x, robot_y, robot_z)
            
            elif cmd_lower == 'l':  # Line
                if current_point is None:
                    print("Warning: Line command without a current point")
                    continue
                
                for j in range(0, len(params), 2):
                    if j+1 < len(params):
                        x, y = params[j], params[j+1]
                        
                        if relative:
                            x += current_point[0]
                            y += current_point[1]
                        
                        current_point = (x, y)
                        robot_x, robot_y, robot_z = self.map_svg_to_robot(x, y)
                        
                        # Ensure coordinates are within bounds
                        robot_y = max(min(robot_y, self.y_max), self.y_min)
                        robot_z = max(min(robot_z, self.z_max), self.z_min)
                        robot_x = self.x_from_z(robot_z)
                        
                        # If pen is not down, approach the canvas first
                        if not pen_down:
                            self.approach_canvas(robot_x, robot_y, robot_z, speed=speed)
                            pen_down = True
                        else:
                            # Move to this point with pen down
                            result = self.arm.set_position(x=robot_x, y=robot_y, z=robot_z, 
                                                         roll=-180, pitch=-90, yaw=0, 
                                                         speed=speed, wait=True)
                            if result[0] != 0:
                                print(f"Warning: Line movement error: {result}")
                        
                        current_robot_point = (robot_x, robot_y, robot_z)
            
            elif cmd_lower == 'h':  # Horizontal line
                if current_point is None:
                    print("Warning: H command without a current point")
                    continue
                
                for x_param in params:
                    x = x_param
                    if relative:
                        x += current_point[0]
                    
                    # Keep the same y coordinate
                    y = current_point[1]
                    current_point = (x, y)
                    
                    robot_x, robot_y, robot_z = self.map_svg_to_robot(x, y)
                    
                    # Ensure coordinates are within bounds
                    robot_y = max(min(robot_y, self.y_max), self.y_min)
                    robot_z = max(min(robot_z, self.z_max), self.z_min)
                    robot_x = self.x_from_z(robot_z)
                    
                    # If pen is not down, approach the canvas first
                    if not pen_down:
                        self.approach_canvas(robot_x, robot_y, robot_z, speed=speed)
                        pen_down = True
                    else:
                        # Move to this point with pen down
                        result = self.arm.set_position(x=robot_x, y=robot_y, z=robot_z, 
                                                     roll=-180, pitch=-90, yaw=0, 
                                                     speed=speed, wait=True)
                    
                    current_robot_point = (robot_x, robot_y, robot_z)
            
            elif cmd_lower == 'v':  # Vertical line
                if current_point is None:
                    print("Warning: V command without a current point")
                    continue
                
                for y_param in params:
                    y = y_param
                    if relative:
                        y += current_point[1]
                    
                    # Keep the same x coordinate
                    x = current_point[0]
                    current_point = (x, y)
                    
                    robot_x, robot_y, robot_z = self.map_svg_to_robot(x, y)
                    
                    # Ensure coordinates are within bounds
                    robot_y = max(min(robot_y, self.y_max), self.y_min)
                    robot_z = max(min(robot_z, self.z_max), self.z_min)
                    robot_x = self.x_from_z(robot_z)
                    
                    # If pen is not down, approach the canvas first
                    if not pen_down:
                        self.approach_canvas(robot_x, robot_y, robot_z, speed=speed)
                        pen_down = True
                    else:
                        # Move to this point with pen down
                        result = self.arm.set_position(x=robot_x, y=robot_y, z=robot_z, 
                                                     roll=-180, pitch=-90, yaw=0, 
                                                     speed=speed, wait=True)
                    
                    current_robot_point = (robot_x, robot_y, robot_z)
            
            elif cmd_lower == 'q':  # Quadratic Bezier
                if current_point is None or len(params) < 4:
                    print("Warning: Invalid Q command")
                    continue
                
                # Process each quadratic bezier segment
                for j in range(0, len(params), 4):
                    if j+3 < len(params):
                        # Control point
                        cx, cy = params[j], params[j+1]
                        # End point
                        x, y = params[j+2], params[j+3]
                        
                        if relative:
                            cx += current_point[0]
                            cy += current_point[1]
                            x += current_point[0]
                            y += current_point[1]
                        
                        # Start point is the current point
                        sx, sy = current_point
                        
                        # Interpolate the curve with multiple segments
                        steps = 20  # Number of line segments to approximate the curve
                        
                        for t in np.linspace(0, 1, steps):
                            # Quadratic Bezier formula
                            px = (1-t)**2 * sx + 2*(1-t)*t * cx + t**2 * x
                            py = (1-t)**2 * sy + 2*(1-t)*t * cy + t**2 * y
                            
                            robot_x, robot_y, robot_z = self.map_svg_to_robot(px, py)
                            
                            # Ensure coordinates are within bounds
                            robot_y = max(min(robot_y, self.y_max), self.y_min)
                            robot_z = max(min(robot_z, self.z_max), self.z_min)
                            robot_x = self.x_from_z(robot_z)
                            
                            # If pen is not down, approach the canvas first
                            if not pen_down:
                                self.approach_canvas(robot_x, robot_y, robot_z, speed=speed)
                                pen_down = True
                            else:
                                # Move to this point with pen down
                                result = self.arm.set_position(x=robot_x, y=robot_y, z=robot_z, 
                                                             roll=-180, pitch=-90, yaw=0, 
                                                             speed=speed, wait=True)
                            
                            current_robot_point = (robot_x, robot_y, robot_z)
                        
                        # Update the current point to the end of this bezier curve
                        current_point = (x, y)
            
            elif cmd_lower == 'c':  # Cubic Bezier
                if current_point is None or len(params) < 6:
                    print("Warning: Invalid C command")
                    continue
                
                # Process each cubic bezier segment
                for j in range(0, len(params), 6):
                    if j+5 < len(params):
                        # Control points
                        c1x, c1y = params[j], params[j+1]
                        c2x, c2y = params[j+2], params[j+3]
                        # End point
                        x, y = params[j+4], params[j+5]
                        
                        if relative:
                            c1x += current_point[0]
                            c1y += current_point[1]
                            c2x += current_point[0]
                            c2y += current_point[1]
                            x += current_point[0]
                            y += current_point[1]
                        
                        # Start point is the current point
                        sx, sy = current_point
                        
                        # Interpolate the curve with multiple segments
                        steps = 20  # Number of line segments to approximate the curve
                        
                        for t in np.linspace(0, 1, steps):
                            # Cubic Bezier formula
                            px = (1-t)**3 * sx + 3*(1-t)**2*t * c1x + 3*(1-t)*t**2 * c2x + t**3 * x
                            py = (1-t)**3 * sy + 3*(1-t)**2*t * c1y + 3*(1-t)*t**2 * c2y + t**3 * y
                            
                            robot_x, robot_y, robot_z = self.map_svg_to_robot(px, py)
                            
                            # Ensure coordinates are within bounds
                            robot_y = max(min(robot_y, self.y_max), self.y_min)
                            robot_z = max(min(robot_z, self.z_max), self.z_min)
                            robot_x = self.x_from_z(robot_z)
                            
                            # If pen is not down, approach the canvas first
                            if not pen_down:
                                self.approach_canvas(robot_x, robot_y, robot_z, speed=speed)
                                pen_down = True
                            else:
                                # Move to this point with pen down
                                result = self.arm.set_position(x=robot_x, y=robot_y, z=robot_z, 
                                                             roll=-180, pitch=-90, yaw=0, 
                                                             speed=speed, wait=True)
                            
                            current_robot_point = (robot_x, robot_y, robot_z)
                        
                        # Update the current point to the end of this bezier curve
                        current_point = (x, y)
            
            elif cmd_lower == 'z':  # Close path
                if start_point and current_point != start_point:
                    # Line from current point to start point
                    robot_x, robot_y, robot_z = self.map_svg_to_robot(start_point[0], start_point[1])
                    
                    # Ensure coordinates are within bounds
                    robot_y = max(min(robot_y, self.y_max), self.y_min)
                    robot_z = max(min(robot_z, self.z_max), self.z_min)
                    robot_x = self.x_from_z(robot_z)
                    
                    # If pen is not down, approach the canvas first
                    if not pen_down:
                        self.approach_canvas(robot_x, robot_y, robot_z, speed=speed)
                        pen_down = True
                    else:
                        # Move to start point with pen down
                        result = self.arm.set_position(x=robot_x, y=robot_y, z=robot_z, 
                                                     roll=-180, pitch=-90, yaw=0, 
                                                     speed=speed, wait=True)
                    
                    current_robot_point = (robot_x, robot_y, robot_z)
                    current_point = start_point
            
            # Check if this is the last command OR if the next command is a move command
            # In these cases, we should lift the pen
            is_last_command = i == len(path_commands) - 1
            next_is_move = not is_last_command and path_commands[i+1][0].lower() == 'm'
            
            if (is_last_command or next_is_move) and pen_down and current_robot_point:
                self.retreat_from_canvas(current_robot_point[0], current_robot_point[1], current_robot_point[2], speed=speed)
                pen_down = False
        
        # Make sure the pen is up when we're done with the path
        if pen_down and current_robot_point:
            self.retreat_from_canvas(current_robot_point[0], current_robot_point[1], current_robot_point[2], speed=speed)
        
        return True

    def parse_svg_file(self, svg_file_path):
        """
        Parse an SVG file and extract paths.
        
        Args:
            svg_file_path (str): Path to the SVG file
            
        Returns:
            list: List of path data strings
        """
        try:
            tree = ET.parse(svg_file_path)
            root = tree.getroot()
            
            # Get SVG namespace if it exists
            namespace = {'svg': 'http://www.w3.org/2000/svg'}
            
            # Get SVG dimensions
            width = root.attrib.get('width')
            height = root.attrib.get('height')
            
            if width:
                self.svg_width = float(width.replace('px', ''))
            if height:
                self.svg_height = float(height.replace('px', ''))
                
            if self.debug:
                print(f"SVG dimensions: {self.svg_width} x {self.svg_height}")
                
            # Extract all path elements
            paths = []
            
            # Try with namespace first
            path_elements = root.findall('.//svg:path', namespace)
            
            # If no paths found, try without namespace
            if not path_elements:
                path_elements = root.findall('.//path')
            
            for path in path_elements:
                path_data = path.attrib.get('d')
                if path_data:
                    paths.append(path_data)
            
            if self.debug:
                print(f"Found {len(paths)} paths in SVG file")
                
            return paths
        except Exception as e:
            print(f"Error parsing SVG file: {e}")
            return []

    def paint_svg(self, svg_file_path, speed=20):
        """
        Paint an SVG file on the canvas.
        
        Args:
            svg_file_path (str): Path to the SVG file
            speed (int): Drawing speed
            
        Returns:
            bool: True if painting completed successfully, False otherwise
        """
        try:
            print(f"Loading SVG file: {svg_file_path}")
            paths = self.parse_svg_file(svg_file_path)
            
            if not paths:
                print("No paths found in SVG file")
                return False
            
            # Initial safety move - go to a position above the canvas center
            safe_z = (self.z_min + self.z_max) / 2
            safe_y = (self.y_min + self.y_max) / 2
            safe_x = self.x_from_z(safe_z) - self.off_canvas_distance
            
            print(f"Moving to safe position: ({safe_x:.2f}, {safe_y:.2f}, {safe_z:.2f})")
            result = self.arm.set_position(x=safe_x, y=safe_y, z=safe_z, roll=-180, pitch=-90, yaw=0, speed=speed, wait=True)
            
            if result[0] != 0:
                print(f"Error moving to safe position: {result}")
                return False
            
            # Paint each path
            for i, path_data in enumerate(paths):
                print(f"Painting path {i+1}/{len(paths)}")
                success = self.draw_svg_path(path_data, speed)
                if not success:
                    print(f"Error painting path {i+1}")
                    return False
                
                # Brief pause between paths
                time.sleep(0.5)
            
            # Return to safe position when done
            print("Returning to safe position")
            self.arm.set_position(x=safe_x, y=safe_y, z=safe_z, roll=-180, pitch=-90, yaw=0, speed=speed, wait=True)
            
            print("SVG painting completed successfully")
            return True
            
        except Exception as e:
            print(f"Error during SVG painting: {e}")
            # Emergency stop if something goes wrong
            self.arm.emergency_stop()
            return False
    
    def paint_svg_from_string(self, svg_string, speed=20):
        """
        Paint an SVG from a string content.
        
        Args:
            svg_string (str): SVG content as a string
            speed (int): Drawing speed
            
        Returns:
            bool: True if painting completed successfully, False otherwise
        """
        try:
            print("Loading SVG from string")
            root = ET.fromstring(svg_string)
            
            # Get SVG dimensions
            width = root.attrib.get('width')
            height = root.attrib.get('height')
            
            if width:
                self.svg_width = float(width.replace('px', ''))
            if height:
                self.svg_height = float(height.replace('px', ''))
                
            # Extract all path elements
            paths = []
            
            # Try with namespace
            namespace = {'svg': 'http://www.w3.org/2000/svg'}
            path_elements = root.findall('.//svg:path', namespace)
            
            # If no paths found, try without namespace
            if not path_elements:
                path_elements = root.findall('.//path')
            
            for path in path_elements:
                path_data = path.attrib.get('d')
                if path_data:
                    paths.append(path_data)
            
            if not paths:
                print("No paths found in SVG content")
                return False
                
            # Initial safety move - go to a position above the canvas
            safe_z = (self.z_min + self.z_max) / 2
            safe_y = (self.y_min + self.y_max) / 2
            safe_x = self.x_from_z(safe_z) - self.off_canvas_distance
            
            print(f"Moving to safe position: ({safe_x:.2f}, {safe_y:.2f}, {safe_z:.2f})")
            result = self.arm.set_position(x=safe_x, y=safe_y, z=safe_z, roll=-180, pitch=-90, yaw=0, speed=speed, wait=True)
            
            if result[0] != 0:
                print(f"Error moving to safe position: {result}")
                return False
            
            # Paint each path
            for i, path_data in enumerate(paths):
                print(f"Painting path {i+1}/{len(paths)}")
                success = self.draw_svg_path(path_data, speed)
                if not success:
                    print(f"Error painting path {i+1}")
                    return False
                
                # Brief pause between paths
                time.sleep(0.5)
            
            # Return to safe position when done
            print("Returning to safe position")
            self.arm.set_position(x=safe_x, y=safe_y, z=safe_z, roll=-180, pitch=-90, yaw=0, speed=speed, wait=True)
            
            print("SVG painting completed successfully")
            return True
            
        except Exception as e:
            print(f"Error during SVG painting from string: {e}")
            # Emergency stop if something goes wrong
            self.arm.emergency_stop()
    
        
    def test_drawing(self, test_type="square", size=100, speed=20):
        """
        Test the robot's drawing capabilities with simple shapes.
        
        Args:
            test_type (str): Type of test to run - "square", "circle", "zigzag", or "calibration"
            size (float): Size of the test pattern in mm
            speed (int): Drawing speed
            
        Returns:
            bool: True if test completed successfully
        """
        print(f"Running {test_type} test with size {size}mm at speed {speed}")
        
        # Go to canvas center
        center_z = (self.z_min + self.z_max) / 2
        center_y = (self.y_min + self.y_max) / 2
        center_x = self.x_from_z(center_z)
        
        # Move to a safe position first
        safe_x = center_x - self.off_canvas_distance
        result = self.arm.set_position(x=safe_x, y=center_y, z=center_z, roll=-180, pitch=-90, yaw=0, speed=speed, wait=True)
        
        if result[0] != 0:
            print(f"Error moving to center: {result}")
            return False
        
        if test_type == "square":
            # Calculate square corners in robot coordinates
            half_size = size / 2
            corners = [
                (center_y - half_size, center_z - half_size),  # Bottom left
                (center_y - half_size, center_z + half_size),  # Top left
                (center_y + half_size, center_z + half_size),  # Top right
                (center_y + half_size, center_z - half_size),  # Bottom right
                (center_y - half_size, center_z - half_size)   # Back to start
            ]
            
            # Move to first corner
            y, z = corners[0]
            x = self.x_from_z(z)
            self.approach_canvas(x, y, z, speed=speed)
            
            # Draw the square
            for y, z in corners[1:]:
                x = self.x_from_z(z)
                self.arm.set_position(x=x, y=y, z=z, roll=-180, pitch=-90, yaw=0, speed=speed, wait=True)
            
            # Lift pen
            self.retreat_from_canvas(x, y, z, speed=speed)
            
        elif test_type == "circle":
            # Draw a circle centered at the canvas center
            radius = size / 2
            steps = 36  # Number of segments to approximate the circle
            
            # Move to first point of circle
            first_y = center_y + radius
            first_z = center_z
            first_x = self.x_from_z(first_z)
            self.approach_canvas(first_x, first_y, first_z, speed=speed)
            
            # Draw the circle
            for angle in np.linspace(0, 2*np.pi, steps)[1:]:
                y = center_y + radius * np.cos(angle)
                z = center_z + radius * np.sin(angle)
                x = self.x_from_z(z)
                self.arm.set_position(x=x, y=y, z=z, roll=-180, pitch=-90, yaw=0, speed=speed, wait=True)
            
            # Lift pen
            self.retreat_from_canvas(x, y, z, speed=speed)
            
        elif test_type == "zigzag":
            # Draw a zigzag pattern across the canvas
            zigzag_width = size
            zigzag_height = size / 2
            steps = 10  # Number of zigzag segments
            
            # Start at bottom left of zigzag
            start_y = center_y - zigzag_width/2
            start_z = center_z - zigzag_height/2
            start_x = self.x_from_z(start_z)
            self.approach_canvas(start_x, start_y, start_z, speed=speed)
            
            # Draw zigzag
            for i in range(steps):
                # Alternate between top and bottom points
                z = center_z + zigzag_height/2 if i % 2 == 0 else center_z - zigzag_height/2
                y = start_y + (i+1) * zigzag_width / steps
                x = self.x_from_z(z)
                self.arm.set_position(x=x, y=y, z=z, roll=-180, pitch=-90, yaw=0, speed=speed, wait=True)
            
            # Lift pen
            self.retreat_from_canvas(x, y, z, speed=speed)
            
        elif test_type == "calibration":
            # Draw a calibration pattern that tests the edges of the canvas
            margin = 20  # mm from edge
            
            # Define the calibration pattern points as (y, z) coordinates
            cal_points = [
                (self.y_min + margin, self.z_min + margin),      # Bottom left
                (self.y_min + margin, self.z_max - margin),      # Top left 
                (self.y_max - margin, self.z_max - margin),      # Top right
                (self.y_max - margin, self.z_min + margin),      # Bottom right
                (self.y_min + margin, self.z_min + margin),      # Back to bottom left
                # Center cross
                (center_y, center_z - size/2),                   # Bottom of cross
                (center_y, center_z + size/2),                   # Top of cross
                (center_y, center_z),                            # Center
                (center_y - size/2, center_z),                   # Left of cross
                (center_y + size/2, center_z)                    # Right of cross
            ]
            
            # Draw each segment of the calibration pattern
            for i, (y, z) in enumerate(cal_points):
                x = self.x_from_z(z)
                
                if i == 0:  # First point
                    self.approach_canvas(x, y, z, speed=speed)
                elif i == 5:  # Start of cross
                    self.retreat_from_canvas(x, y, z, speed=speed)
                    self.approach_canvas(x, y, z, speed=speed)
                elif i == 7:  # Middle of cross
                    self.retreat_from_canvas(x, y, z, speed=speed)
                    self.approach_canvas(x, y, z, speed=speed)
                else:
                    self.arm.set_position(x=x, y=y, z=z, roll=-180, pitch=-90, yaw=0, speed=speed, wait=True)
            
            # Lift pen at the end
            self.retreat_from_canvas(x, y, z, speed=speed)
        
        # Return to safe position
        self.arm.set_position(x=safe_x, y=center_y, z=center_z, roll=-180, pitch=-90, yaw=0, speed=speed, wait=True)
        
        print(f"{test_type} test completed")
        return True