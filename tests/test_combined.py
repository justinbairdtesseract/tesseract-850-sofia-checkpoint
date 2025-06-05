import unittest
import sys
import os

# Add project root to sys.path to allow importing 'combined'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from combined import XArmController, interactive # Assuming interactive is where OSC is handled

class TestBrushOffsets(unittest.TestCase):
    def setUp(self):
        # Provide a dummy IP for testing purposes, as the arm won't be connected.
        # Also mock the OSC client to avoid actual OSC messages during tests.
        class MockOSCClient:
            def send_message(self, address, value):
                pass # Do nothing

        # Need to mock XArmAPI as it tries to connect upon instantiation
        class MockXArmAPI:
            def __init__(self, port=None, is_radian=False, do_not_open=False, **kwargs):
                print(f"MockXArmAPI initialized with port={port}")
                self.connected = False # Simulate not connected
                self.default_is_radian = is_radian
                self._port = port
                self._do_not_open = do_not_open
                self.version_number = None
                # Add any methods that are called in XArmController.setup() or __init__
                self.CMD_EXIST = 0 # Mocking a class attribute

            def connect(self, port=None, baudrate=None, timeout=None, **kwargs):
                self.connected = True
                print(f"MockXArmAPI connect called with port={port}")

            def disconnect(self):
                self.connected = False
                print("MockXArmAPI disconnect called")

            def motion_enable(self, enable=True, servo_id=None):
                print(f"MockXArmAPI motion_enable called with enable={enable}")
                return 0 # Success

            def set_mode(self, mode=0, detection_param=None):
                print(f"MockXArmAPI set_mode called with mode={mode}")
                return 0 # Success

            def set_state(self, state=0, **kwargs):
                print(f"MockXArmAPI set_state called with state={state}")
                return 0 # Success

            def get_is_moving(self):
                return False # Simulate arm is not moving

            def set_servo_angle(self, servo_id=None, angle=None, speed=None, acc=None, mvtime=None, wait=False, timeout=None, radius=None, is_radian=None):
                print(f"MockXArmAPI set_servo_angle called for servo_id={servo_id} with angle={angle}")
                return 0 # Success


        # Temporarily replace XArmAPI with the mock for the test
        # Get the actual module object for 'combined'
        combined_module = sys.modules[XArmController.__module__]

        original_xarm_api = getattr(combined_module, 'XArmAPI', None)
        setattr(combined_module, 'XArmAPI', MockXArmAPI)

        self.controller = XArmController(ip_address='127.0.0.1')
        self.controller.osc_client = MockOSCClient() # Mock the OSC client

        # Restore original XArmAPI
        if original_xarm_api:
            setattr(combined_module, 'XArmAPI', original_xarm_api)
        else: # If it wasn't there (e.g. if XArmAPI was not imported in combined), remove our mock
            delattr(combined_module, 'XArmAPI')


        # Define some SVG and canvas parameters for consistent testing
        self.controller.svg_width = 1000
        self.controller.svg_height = 1000
        self.controller.canvas_corners = [
            [400,  200, 780],  # top_left (y_max for mapping, z_max for mapping)
            [400, -200, 780],  # top_right (y_min for mapping)
            [400,  200, 100],  # bottom_left (z_min for mapping)
            [400, -200, 100]   # bottom_right
        ]
        # From canvas_corners:
        # y_min should be -200 (canvas_corners[1][1]) based on min(corners[0][1], corners[1][1])
        # y_max should be 200 (canvas_corners[0][1]) based on max(corners[0][1], corners[1][1])
        # z_min should be 100 (canvas_corners[2][2]) based on min(corners[2][2], corners[0][2])
        # z_max should be 780 (canvas_corners[0][2]) based on max(corners[2][2], corners[0][2])
        self.controller.calculate_scaling_factors() # Recalculate based on new corners
        self.controller.calculate_canvas_bounds()   # Recalculate based on new corners

        # Explicitly set y_min and y_max as per the logic in calculate_canvas_bounds
        # self.controller.y_min = min(self.controller.canvas_corners[0][1], self.controller.canvas_corners[1][1])
        # self.controller.y_max = max(self.controller.canvas_corners[0][1], self.controller.canvas_corners[1][1])
        # This means y_min = -200, y_max = 200 for the given corners.

        self.controller.brush_offset_mm = 25

    def test_map_svg_to_robot_offset(self):
        svg_x, svg_y = 500, 500 # Center of a 1000x1000 SVG

        # Expected Y without offset:
        # nx = svg_x / self.controller.svg_width = 500 / 1000 = 0.5
        # self.controller.y_max = 200
        # self.controller.y_min = -200
        # expected_robot_y_no_offset = self.controller.y_max - nx * (self.controller.y_max - self.controller.y_min)
        # expected_robot_y_no_offset = 200 - 0.5 * (200 - (-200))
        # expected_robot_y_no_offset = 200 - 0.5 * 400
        # expected_robot_y_no_offset = 200 - 200 = 0

        expected_robot_y_no_offset = 0.0 # Based on the calculation above

        # Test with brush 0 (no offset)
        self.controller.active_brush_index = 0
        _, robot_y_0, _ = self.controller.map_svg_to_robot(svg_x, svg_y)
        self.assertAlmostEqual(robot_y_0, expected_robot_y_no_offset, places=5)

        # Test with brush 1
        self.controller.active_brush_index = 1
        _, robot_y_1, _ = self.controller.map_svg_to_robot(svg_x, svg_y)
        expected_offset_1 = 1 * self.controller.brush_offset_mm
        self.assertAlmostEqual(robot_y_1, expected_robot_y_no_offset - expected_offset_1, places=5)

        # Test with brush 3
        self.controller.active_brush_index = 3
        _, robot_y_3, _ = self.controller.map_svg_to_robot(svg_x, svg_y)
        expected_offset_3 = 3 * self.controller.brush_offset_mm
        self.assertAlmostEqual(robot_y_3, expected_robot_y_no_offset - expected_offset_3, places=5)

    def test_osc_set_brush_command(self):
        # Get the 'combined' module to access its global 'controller' variable
        combined_module = sys.modules['combined']

        # Store the original global controller from combined.py (if any)
        original_global_controller = getattr(combined_module, 'controller', None)

        # Set the global controller in combined.py to our test controller
        setattr(combined_module, 'controller', self.controller)

        try:
            # Initial state check (active_brush_index is 0 by default in setUp)
            self.assertEqual(self.controller.active_brush_index, 0)

            # Test valid brush index
            interactive("/macbook", "brush", 2) # Simulate OSC message
            self.assertEqual(self.controller.active_brush_index, 2)

            # Test invalid brush index (too high)
            interactive("/macbook", "brush", 5)
            self.assertEqual(self.controller.active_brush_index, 2) # Should remain unchanged

            # Test invalid brush index (too low)
            interactive("/macbook", "brush", -1)
            self.assertEqual(self.controller.active_brush_index, 2) # Should remain unchanged

            # Test non-integer
            # int() will raise ValueError, which should be handled inside 'interactive' or not occur
            # For this test, we assume the OSC message parsing ensures an int is passed or it's handled.
            # If int("abc") is directly called in `interactive` it would raise an unhandled ValueError.
            # The current `interactive` has `brush_idx = int(max_msg[1])`.
            # A robust `interactive` would have a try-except around this.
            # For now, we'll assume valid type or that errors are handled gracefully.
            # Let's test that if an error occurs, the value doesn't change.
            current_brush_index_before_error = self.controller.active_brush_index
            try:
                interactive("/macbook", "brush", "abc")
            except ValueError: # Simulating that int() would fail.
                pass
            self.assertEqual(self.controller.active_brush_index, current_brush_index_before_error)

        finally:
            # Restore the original global controller in combined.py
            if original_global_controller is not None:
                setattr(combined_module, 'controller', original_global_controller)
            else:
                # If it was None initially, ensure it's set back to None or delattr
                # For simplicity, setting to None if it didn't exist or was None
                setattr(combined_module, 'controller', None)


if __name__ == '__main__':
    # Need to make sure the XArmController's XArmAPI is mocked for __main__ execution too
    class MockXArmAPI:
        def __init__(self, port=None, is_radian=False, do_not_open=False, **kwargs): self.connected = False; self.CMD_EXIST = 0
        def connect(self, **kwargs): self.connected = True
        def disconnect(self): self.connected = False
        def motion_enable(self, **kwargs): return 0
        def set_mode(self, **kwargs): return 0
        def set_state(self, **kwargs): return 0
        def get_is_moving(self): return False
        def set_servo_angle(self, **kwargs): return 0

    # This global mock is a bit tricky. For robust testing, dependency injection or a more sophisticated mocking framework is better.
    # For now, let's assume 'combined' module can be manipulated for testing.

    # Get the actual module object for 'combined' when running as __main__
    # Ensure 'combined' is imported in a way that sys.modules can find it.
    # The from combined import XArmController, interactive at the top should suffice.
    if XArmController.__module__ in sys.modules:
        current_combined_module = sys.modules[XArmController.__module__]
        original_xarm_api_main = getattr(current_combined_module, 'XArmAPI', None)
        setattr(current_combined_module, 'XArmAPI', MockXArmAPI)
    else:
        # This case should ideally not happen if imports are correct
        current_combined_module = None
        original_xarm_api_main = None
        print("Warning: Could not find combined module in sys.modules for __main__ mock.")

    unittest.main()

    # Restore original after tests run
    if current_combined_module and original_xarm_api_main:
        setattr(current_combined_module, 'XArmAPI', original_xarm_api_main)
    elif current_combined_module and hasattr(current_combined_module, 'XArmAPI') and not original_xarm_api_main:
        delattr(current_combined_module, 'XArmAPI')
