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
import json
import threading
import struct
import xml.etree.ElementTree as ET
import zlib
import numpy as np
from pathlib import Path

from brush_selector import BrushSelector, BrushSelectorError
try:
    from command_video_recorder import CommandVideoRecorder
except Exception as recorder_import_error:
    COMMAND_VIDEO_RECORDER_IMPORT_ERROR = str(recorder_import_error)

    class CommandVideoRecorder:
        @classmethod
        def from_env(cls):
            return cls()

        def describe(self):
            print(
                f"Command video recorder: unavailable ({COMMAND_VIDEO_RECORDER_IMPORT_ERROR})",
                flush=True,
            )

        def start(self, _command):
            return None

        def stop(self):
            return None

SVG_NUMBER_RE = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
CANVAS_CORNER_INDEX = {
    'top_left': 0,
    'top_right': 1,
    'bottom_left': 2,
    'bottom_right': 3,
}

LINUX_IP = os.getenv('TESSERACT_OSC_BIND_IP', '0.0.0.0')
MAC_IP = os.getenv('TESSERACT_OSC_TARGET_IP', '192.168.0.128')
MAXMSP_TO_PORT = int(os.getenv('TESSERACT_OSC_TARGET_PORT', '7654'))
MAXMSP_FROM_PORT = int(os.getenv('TESSERACT_OSC_BIND_PORT', '7655'))
MUSE_OSC_ENABLED = os.getenv('TESSERACT_MUSE_OSC_ENABLED', '1').strip().lower() in {
    '1',
    'true',
    'yes',
    'on',
}
MUSE_OSC_BIND_IP = os.getenv('TESSERACT_MUSE_OSC_BIND_IP', '0.0.0.0')
MUSE_OSC_BIND_PORT = int(os.getenv('TESSERACT_MUSE_OSC_BIND_PORT', '7400'))
MUSE_ATTENTION_LOW = float(os.getenv('TESSERACT_MUSE_ATTENTION_LOW', '0.33'))
MUSE_ATTENTION_HIGH = float(os.getenv('TESSERACT_MUSE_ATTENTION_HIGH', '0.66'))
MUSE_HEAD_GESTURE_THRESHOLD = float(os.getenv('TESSERACT_MUSE_HEAD_GESTURE_THRESHOLD', '0.5'))
MUSE_HEAD_GESTURE_COOLDOWN_SEC = float(os.getenv('TESSERACT_MUSE_HEAD_GESTURE_COOLDOWN_SEC', '1.5'))
MUSE_BRUSH_CHANGE_STROKE_LOCKOUT = int(os.getenv('TESSERACT_MUSE_BRUSH_CHANGE_STROKE_LOCKOUT', '2'))
BCI_COLOR_BRUSH_MIN = int(os.getenv('TESSERACT_BCI_COLOR_BRUSH_MIN', '2'))
BCI_COLOR_BRUSH_MAX = int(os.getenv('TESSERACT_BCI_COLOR_BRUSH_MAX', '6'))
ROBOT_IP = os.getenv('TESSERACT_ROBOT_IP', '192.168.0.203')
DEFAULT_CANVAS_CORNERS_PATH = (
    Path(__file__).resolve().parents[1] / 'Brush-selector' / 'canvas_corners.json'
)
CANVAS_CORNERS_PATH = Path(
    os.getenv('TESSERACT_CANVAS_CORNERS_PATH', str(DEFAULT_CANVAS_CORNERS_PATH))
).expanduser()
DEFAULT_BCI_PAINT_STATE_PATH = (
    Path(__file__).resolve().parents[1] / 'Brush-selector-Arduino-Gripper' / 'bci_paint_state.json'
)
BCI_PAINT_STATE_PATH = Path(
    os.getenv('TESSERACT_BCI_PAINT_STATE_PATH', str(DEFAULT_BCI_PAINT_STATE_PATH))
).expanduser()
DEFAULT_PAINT_Z_OFFSETS_PATH = (
    Path(__file__).resolve().parents[1] / 'Brush-selector-Arduino-Gripper' / 'paint_z_offsets.json'
)
PAINT_Z_OFFSETS_PATH = Path(
    os.getenv('TESSERACT_PAINT_Z_OFFSETS_PATH', str(DEFAULT_PAINT_Z_OFFSETS_PATH))
).expanduser()
PAINT_Z_NUDGE_STEP_MM = float(os.getenv('TESSERACT_Z_NUDGE_STEP_MM', '0.1'))
PAINT_Z_NUDGE_LIMIT_MM = float(os.getenv('TESSERACT_Z_NUDGE_LIMIT_MM', '10.0'))
PAINT_Z_BRUSH_COUNT = int(os.getenv('TESSERACT_PAINT_Z_BRUSH_COUNT', '6'))
CANVAS_REFERENCE_BRUSH = int(os.getenv('TESSERACT_CANVAS_REFERENCE_BRUSH', '1'))
PAINT_SURFACE_MODE = os.getenv('TESSERACT_PAINT_SURFACE_MODE', 'easel').strip().lower()
PAINT_Z_COMMANDS = {'z_up', 'z-up', 'zup', 'z_down', 'z-down', 'zdown', 'z', 'paint_z'}
BCI_LIVE_SUBCOMMANDS = {'emg', 'movement', 'status', 'stop'}
BCI_SVG_CANVAS_LAYOUT = os.getenv('TESSERACT_BCI_SVG_CANVAS_LAYOUT', 'full').strip().lower()
BCI_SVG_CANVAS_PADDING = float(os.getenv('TESSERACT_BCI_SVG_CANVAS_PADDING', '0.03'))
BCI_LAYER_PREVIEW_DIR = Path(
    os.getenv(
        'TESSERACT_BCI_LAYER_PREVIEW_DIR',
        str(Path(__file__).resolve().parents[1] / 'Brush-selector-Arduino-Gripper' / 'layer_previews'),
    )
).expanduser()
BCI_LAYER_PREVIEW_SIZE = int(os.getenv('TESSERACT_BCI_LAYER_PREVIEW_SIZE', '1200'))

MOVE_BETWEEN_SPEED = 400
STROKE_SPEED = 500

ENABLE_FACE_MESH = os.getenv('TESSERACT_ENABLE_FACE_MESH', '0').strip().lower() in {
    '1',
    'true',
    'yes',
    'on',
}
ENABLE_STARTUP_ARM_MOVE = os.getenv('TESSERACT_ENABLE_STARTUP_ARM_MOVE', '0').strip().lower() in {
    '1',
    'true',
    'yes',
    'on',
}

face_detection_thread = vision.start_face_mesh('webcam', True) if ENABLE_FACE_MESH else None
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
        self.arm = XArmAPI(self.IP_ADDRESS, is_radian=False)
        self.setup()

        # States
        self.busy = False
        self.command_lock = threading.Lock()
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
        self.load_canvas_corners()
        self.calculate_scaling_factors()
        self.calculate_canvas_bounds()
        self.canvas_reference_brush = int(CANVAS_REFERENCE_BRUSH)
        self.paint_surface_mode = self._resolved_paint_surface_mode()
        print(
            f"Paint surface mode: {self.paint_surface_mode} "
            f"(configured={PAINT_SURFACE_MODE})",
            flush=True,
        )
        print(
            f"Canvas homing reference brush: brush {self.canvas_reference_brush}. "
            "Capture canvas corners with this brush; other brushes use relative Z trims.",
            flush=True,
        )

        self.off_canvas_distance = 25  # Distance to start away from the canvas (X direction)
        self.paint_pressure_range = 5
        self.current_pressure = 0  # 0 means full pressure
        self.paint_z_offset_mm = 0.0
        self.paint_z_step_mm = PAINT_Z_NUDGE_STEP_MM
        self.paint_z_offset_limit_mm = abs(PAINT_Z_NUDGE_LIMIT_MM)
        self.paint_z_brush_count = max(1, int(PAINT_Z_BRUSH_COUNT))
        self.paint_z_offsets_path = PAINT_Z_OFFSETS_PATH
        self.paint_z_offsets_by_brush = {
            brush: 0.0 for brush in range(1, self.paint_z_brush_count + 1)
        }
        self.load_paint_z_offsets()
        self.command_video_recorder = CommandVideoRecorder.from_env()
        self.command_video_recorder.describe()
        self.osc_client = SimpleUDPClient(osc_ip, osc_port)
        print(
            f"Painting Z nudge ready: step={self.paint_z_step_mm:.1f} mm, "
            f"limit=+/-{self.paint_z_offset_limit_mm:.1f} mm, "
            f"brushes=1..{self.paint_z_brush_count}, "
            "commands=/macbook z_up BRUSH and /macbook z_down BRUSH",
            flush=True,
        )

        # SVG-related
        self.svg_strokes = []  # Will hold path strings
        self.stroke_index = 0  # Which stroke we’re on
        # Default SVG size (in case not in file):
        self.svg_min_x = 0
        self.svg_min_y = 0
        self.svg_width = 768
        self.svg_height = 1080
        self.svg_reference_space = None
        self.svg_scale_mode = 'reference'
        self._active_svg_canvas_region = (0.0, 1.0, 0.0, 1.0)
        self.default_svg_reference_path = Path(
            os.getenv(
                'TESSERACT_SVG_REFERENCE',
                str(Path(__file__).resolve().with_name('Jas-FULL-Artboard.svg')),
            )
        ).expanduser()
        self.safety_margin = 10  # to keep from going out of bounds
        self.bezier_steps = 15   # for smooth curves
        self.load_default_svg_reference()
        self.bci_state_path = BCI_PAINT_STATE_PATH
        self.bci_attention_state = 0
        self.bci_pending_emg = False
        self.bci_stop_requested = False
        self.bci_color_brush_min = max(1, int(BCI_COLOR_BRUSH_MIN))
        self.bci_color_brush_max = max(self.bci_color_brush_min, int(BCI_COLOR_BRUSH_MAX))
        self.bci_color_brush_count = self.bci_color_brush_max
        self.bci_color_cooldown_strokes = int(
            os.getenv('TESSERACT_BCI_COLOR_COOLDOWN_STROKES', '2')
        )
        self.bci_strokes_since_color_change = self.bci_color_cooldown_strokes
        self.bci_current_color_brush = 0
        self.bci_slots = {
            slot: self._empty_bci_slot(slot)
            for slot in range(3)
        }
        self.load_bci_paint_state()

        # Physical brush selection is handled by BrushSelector.
        # No drawing-space Y offset is applied for brush choice.
        self.active_brush_index = 0
        self.brush_offset_mm = 0
        
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

    def _current_wrist_orientation(self):
        result = self.arm.get_position()
        if (
            not isinstance(result, (list, tuple))
            or len(result) < 2
            or int(result[0]) != 0
            or len(result[1]) < 6
        ):
            raise RuntimeError(f"could not read current wrist orientation: {result}")
        return [float(value) for value in result[1][3:6]]

    def _set_position_keep_wrist(self, *args, **kwargs):
        kwargs.pop('roll', None)
        kwargs.pop('pitch', None)
        kwargs.pop('yaw', None)
        orientation = getattr(self, '_preserved_wrist_orientation', None)
        if orientation is None:
            orientation = self._current_wrist_orientation()
        roll, pitch, yaw = orientation
        return self.arm.set_position(*args, roll=roll, pitch=pitch, yaw=yaw, **kwargs)

    def calculate_scaling_factors(self):
        z1, x1 = self.canvas_corners[0][2], self.canvas_corners[0][0]  # top_left
        z2, x2 = self.canvas_corners[2][2], self.canvas_corners[2][0]  # bottom_left
        if abs(z2 - z1) < 1e-6:
            self.a = 0.0
            self.b = x1
            return
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

    def _resolved_paint_surface_mode(self):
        mode = PAINT_SURFACE_MODE
        if mode in {"table", "tabletop", "horizontal"}:
            return "table"
        if mode in {"easel", "vertical", "legacy"}:
            return "easel"
        xs = [float(corner[0]) for corner in self.canvas_corners]
        ys = [float(corner[1]) for corner in self.canvas_corners]
        zs = [float(corner[2]) for corner in self.canvas_corners]
        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)
        z_range = max(zs) - min(zs)
        planar_span = max(x_range, y_range, 1.0)
        if z_range < max(10.0, planar_span * 0.15):
            return "table"
        return "easel"

    def _is_table_surface(self):
        return getattr(self, "paint_surface_mode", "easel") == "table"

    @staticmethod
    def _lerp_point(a, b, t):
        return [
            float(a[i]) + (float(b[i]) - float(a[i])) * float(t)
            for i in range(3)
        ]

    def _canvas_point_from_uv(self, nx, ny):
        nx = min(max(float(nx), 0.0), 1.0)
        ny = min(max(float(ny), 0.0), 1.0)
        tl, tr, bl, br = self.canvas_corners
        top = self._lerp_point(tl, tr, nx)
        bottom = self._lerp_point(bl, br, nx)
        return self._lerp_point(top, bottom, ny)

    def _quantize_z_mm(self, value):
        return round(float(value), 1)

    def _clamp_paint_z_offset(self, value):
        limit = max(0.0, float(self.paint_z_offset_limit_mm))
        return min(max(self._quantize_z_mm(value), -limit), limit)

    def _paint_z(self, base_z):
        brush = self.active_paint_z_brush(default_to_zero=True)
        offset = self.paint_z_offsets_by_brush.get(brush, 0.0)
        self.paint_z_offset_mm = offset
        return self._quantize_z_mm(float(base_z) + offset)

    def active_paint_z_brush(self, default_to_zero=False):
        try:
            selector = globals().get("brush_selector")
            if selector is not None:
                state = selector.status()
                if bool(state.get("HOLDING_BRUSH", False)):
                    brush = int(state.get("CURRENT_BRUSH", 0))
                    if 1 <= brush <= self.paint_z_brush_count:
                        return brush
        except Exception:
            pass

        brush = int(getattr(self, "bci_current_color_brush", 0) or 0)
        if 1 <= brush <= self.paint_z_brush_count:
            return brush
        return 0 if default_to_zero else None

    def _normalize_paint_z_brush(self, brush=None):
        if brush is None or str(brush).strip().lower() in {"", "active", "current"}:
            active = self.active_paint_z_brush(default_to_zero=False)
            if active is None:
                raise ValueError(
                    "no active brush is known; specify brush 1-"
                    f"{self.paint_z_brush_count}"
                )
            return active
        normalized = int(brush)
        if normalized < 1 or normalized > self.paint_z_brush_count:
            raise ValueError(
                f"brush must be 1..{self.paint_z_brush_count}, got {normalized}"
            )
        return normalized

    def load_paint_z_offsets(self):
        if not self.paint_z_offsets_path.exists():
            print(
                f"Paint Z offsets: no saved offsets at {self.paint_z_offsets_path}; "
                "starting all brushes at +0.0 mm.",
                flush=True,
            )
            return
        try:
            data = json.loads(self.paint_z_offsets_path.read_text(encoding="utf-8"))
            offsets = data.get("brush_offsets_mm", data)
            for brush in range(1, self.paint_z_brush_count + 1):
                raw = offsets.get(str(brush), offsets.get(brush, 0.0))
                self.paint_z_offsets_by_brush[brush] = self._clamp_paint_z_offset(raw)
            print(
                f"Paint Z offsets: loaded from {self.paint_z_offsets_path}: "
                f"{self.paint_z_offsets_by_brush}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"Paint Z offsets: failed to load {self.paint_z_offsets_path}: {exc}; "
                "using +0.0 mm for all brushes.",
                flush=True,
            )

    def save_paint_z_offsets(self):
        self.paint_z_offsets_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "schema": 1,
            "brush_count": self.paint_z_brush_count,
            "brush_offsets_mm": {
                str(brush): self.paint_z_offsets_by_brush.get(brush, 0.0)
                for brush in range(1, self.paint_z_brush_count + 1)
            },
            "step_mm": self.paint_z_step_mm,
            "limit_mm": self.paint_z_offset_limit_mm,
            "updated_at_unix": time.time(),
            "note": (
                "Per-brush live painting Z trims. Positive raises the brush away "
                "from the canvas; negative lowers it toward the canvas."
            ),
        }
        self.paint_z_offsets_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"Paint Z offsets: saved to {self.paint_z_offsets_path}", flush=True)

    def report_paint_z_offset(self, label='status', brush=None):
        active = self.active_paint_z_brush(default_to_zero=True)
        if brush is None:
            print(f"\nPAINT Z OFFSETS ({label}) active_brush={active or 'none'}", flush=True)
            for brush_id in range(1, self.paint_z_brush_count + 1):
                marker = " <- active" if brush_id == active else ""
                print(
                    f"  brush {brush_id}: "
                    f"{self.paint_z_offsets_by_brush.get(brush_id, 0.0):+.1f} mm{marker}",
                    flush=True,
                )
            print(
                f"  step={self.paint_z_step_mm:.1f} mm, "
                f"limit=+/-{self.paint_z_offset_limit_mm:.1f} mm\n",
                flush=True,
            )
            payload = [
                active,
                *[
                    self.paint_z_offsets_by_brush.get(brush_id, 0.0)
                    for brush_id in range(1, self.paint_z_brush_count + 1)
                ],
            ]
            self.osc_client.send_message("/paint/z_offsets", payload)
            return self.paint_z_offsets_by_brush

        brush_id = self._normalize_paint_z_brush(brush)
        offset = self.paint_z_offsets_by_brush.get(brush_id, 0.0)
        print(
            f"PAINT Z OFFSET: brush {brush_id} {offset:+.1f} mm "
            f"({label}, step {self.paint_z_step_mm:.1f} mm)",
            flush=True,
        )
        self.osc_client.send_message(
            "/paint/z_offset",
            [brush_id, offset, 0.0, self.paint_z_step_mm, label],
        )
        return offset

    def set_paint_z_offset(self, value, label='set', brush=None):
        brush_id = self._normalize_paint_z_brush(brush)
        previous = self.paint_z_offsets_by_brush.get(brush_id, 0.0)
        offset = self._clamp_paint_z_offset(value)
        self.paint_z_offsets_by_brush[brush_id] = offset
        self.paint_z_offset_mm = offset
        delta = self._quantize_z_mm(offset - previous)
        print(
            f"PAINT Z OFFSET: brush {brush_id} {offset:+.1f} mm "
            f"({label}, delta {delta:+.1f} mm, step {self.paint_z_step_mm:.1f} mm)",
            flush=True,
        )
        self.save_paint_z_offsets()
        self.osc_client.send_message(
            "/paint/z_offset",
            [brush_id, offset, delta, self.paint_z_step_mm, label],
        )
        return offset

    def adjust_paint_z_offset(self, delta, label='nudge', brush=None):
        brush_id = self._normalize_paint_z_brush(brush)
        current = self.paint_z_offsets_by_brush.get(brush_id, 0.0)
        return self.set_paint_z_offset(current + float(delta), label=label, brush=brush_id)

    def reset_paint_z_offsets(self, brush=None):
        if brush is None or str(brush).strip().lower() == "all":
            for brush_id in range(1, self.paint_z_brush_count + 1):
                self.paint_z_offsets_by_brush[brush_id] = 0.0
            self.paint_z_offset_mm = 0.0
            print("PAINT Z OFFSET: reset all brushes to +0.0 mm", flush=True)
            self.save_paint_z_offsets()
            self.report_paint_z_offset(label="reset_all")
            return self.paint_z_offsets_by_brush
        return self.set_paint_z_offset(0.0, label="reset", brush=brush)

    def load_canvas_corners(self):
        if not CANVAS_CORNERS_PATH.exists():
            print(
                f"Canvas corners: no saved calibration at {CANVAS_CORNERS_PATH}; "
                "using defaults.",
                flush=True,
            )
            return
        try:
            data = json.loads(CANVAS_CORNERS_PATH.read_text(encoding='utf-8'))
            corners = data.get('canvas_corners', data)
            if not self._valid_canvas_corners(corners):
                raise ValueError(f"invalid canvas_corners shape: {corners}")
            self.canvas_corners = [
                [float(value) for value in corner]
                for corner in corners
            ]
            print(
                f"Canvas corners: loaded saved calibration from {CANVAS_CORNERS_PATH}: "
                f"{self.canvas_corners}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"Canvas corners: failed to load {CANVAS_CORNERS_PATH}: {exc}; "
                "using defaults.",
                flush=True,
            )

    def save_canvas_corners(self):
        data = {
            'schema': 1,
            'corner_order': ['top_left', 'top_right', 'bottom_left', 'bottom_right'],
            'canvas_corners': self.canvas_corners,
            'reference_brush': self.canvas_reference_brush,
            'updated_at_unix': time.time(),
            'note': (
                f'Robot XYZ positions in mm, saved from the Max/MSP homing patch. '
                f'Brush {self.canvas_reference_brush} is the canvas reference brush; '
                'other brush heights are adjusted relative to this homing plane '
                'with paint_z_offsets.json.'
            ),
        }
        CANVAS_CORNERS_PATH.parent.mkdir(parents=True, exist_ok=True)
        CANVAS_CORNERS_PATH.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
        print(
            f"Canvas corners: saved calibration to {CANVAS_CORNERS_PATH}: "
            f"{self.canvas_corners}",
            flush=True,
        )

    @staticmethod
    def _valid_canvas_corners(corners):
        if not isinstance(corners, list) or len(corners) != 4:
            return False
        for corner in corners:
            if not isinstance(corner, list) or len(corner) != 3:
                return False
            try:
                [float(value) for value in corner]
            except (TypeError, ValueError):
                return False
        return True

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
            margin = (1 - scale_factor) / 2
            nx = random.uniform(margin, 1 - margin)
            ny = random.uniform(margin, 1 - margin)
            if self._is_table_surface():
                random_x, random_y, random_z = self._canvas_point_from_uv(nx, ny)
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
        return (random_x, random_y, self._paint_z(random_z))

    # --------------
    # MOVE ON/OFF CANVAS
    # --------------
    def approach_canvas(self, x, y, z, roll=-180, speed=50):
        """Approach the canvas using the configured surface geometry."""
        if self._is_table_surface():
            approach_z = z + self.off_canvas_distance
            self._set_position_keep_wrist(x=x, y=y, z=approach_z,
                                          speed=MOVE_BETWEEN_SPEED, wait=True)
            self._set_position_keep_wrist(x=x, y=y, z=z,
                                          speed=speed, wait=True)
            self.osc_client.send_message("/arm/movement", "approaching")
            return

        approach_x = x - self.off_canvas_distance
        self._set_position_keep_wrist(x=approach_x, y=y, z=z,
                                      speed=MOVE_BETWEEN_SPEED, wait=True)
        # Then move onto canvas
        self._set_position_keep_wrist(x=x, y=y, z=z,
                                      speed=speed, wait=True)
        self.osc_client.send_message("/arm/movement", "approaching")

    def retreat_from_canvas(self, x, y, z, speed=50, lift=0):
        """
        Retreat from the canvas. Table mode lifts in Z; easel mode pulls away
        along -X, normal to the upright canvas plane.
        """
        print('retreating')
        if self._is_table_surface():
            retreat_z = z + max(float(lift), float(self.off_canvas_distance))
            self._set_position_keep_wrist(x=x, y=y, z=retreat_z,
                                          speed=speed, wait=True)
            self.osc_client.send_message("/arm/movement", "retreating")
            self.osc_client.send_message("/listen", 1)
            return

        retreat_x = x - max(float(lift), float(self.off_canvas_distance))
        self._set_position_keep_wrist(x=retreat_x, y=y, z=z,
                                      speed=speed, wait=True)

        self.osc_client.send_message("/arm/movement", "retreating")
        self.osc_client.send_message("/listen", 1)

    def _lift_off_plane(self, x, y, z, speed, dist=None):
        """
        Pull the tip away from the canvas. Table mode lifts in Z; easel mode
        keeps the legacy –X normal.
        """
        if dist is None:
            dist = self.STIPPLE_LIFT_MM
        if self._is_table_surface():
            new_z = z + dist
            self._set_position_keep_wrist(x=x, y=y, z=new_z,
                                          speed=speed, wait=True)
            return x, y, new_z
        new_x = x - dist           # –x is the normal to your canvas
        self._set_position_keep_wrist(x=new_x, y=y, z=z,
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
        self._set_position_keep_wrist(x=x, y=y, z=z, speed=speed, wait=True)
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
                self._set_position_keep_wrist(x=nx - self.STIPPLE_LIFT_MM,
                                              y=ny, z=nz,
                                              speed=draw_spd, wait=True)

                # tap
                self._set_position_keep_wrist(x=nx, y=ny, z=nz,
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
            self._set_position_keep_wrist(*end, speed=draw_spd, wait=True)
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
    @staticmethod
    def _parse_svg_length(value):
        if value is None:
            return None
        match = re.match(SVG_NUMBER_RE, str(value).strip())
        if not match:
            return None
        try:
            return float(match.group(0))
        except ValueError:
            return None

    def _read_svg_coordinate_space(self, root):
        view_box = root.attrib.get('viewBox') or root.attrib.get('viewbox')
        if view_box:
            nums = [float(n) for n in re.findall(SVG_NUMBER_RE, view_box)]
            if len(nums) == 4 and nums[2] > 0 and nums[3] > 0:
                self.svg_min_x, self.svg_min_y, self.svg_width, self.svg_height = nums
                return "viewBox"

        width = self._parse_svg_length(root.attrib.get('width'))
        height = self._parse_svg_length(root.attrib.get('height'))
        self.svg_min_x = 0
        self.svg_min_y = 0
        if width and width > 0:
            self.svg_width = width
        if height and height > 0:
            self.svg_height = height
        if width and width > 0 and height and height > 0:
            return "width_height"
        return None

    def _canvas_aspect_ratio(self):
        y_min, y_max, z_min, z_max = self._svg_canvas_bounds()
        canvas_width = abs(y_max - y_min)
        canvas_height = abs(z_max - z_min)
        if canvas_width <= 0 or canvas_height <= 0:
            return 1.0
        return canvas_width / canvas_height

    def _svg_canvas_bounds(self):
        margin = self.safety_margin
        return (
            self.y_min + margin,
            self.y_max - margin,
            self.z_min + margin,
            self.z_max - margin,
        )

    @staticmethod
    def _path_commands(path_data):
        cmd_pat = r'([MLHVCSQTAZmlhvcsqtaz])([^MLHVCSQTAZmlhvcsqtaz]*)'
        for cmd, params in re.findall(cmd_pat, path_data):
            yield cmd, [float(n) for n in re.findall(SVG_NUMBER_RE, params)]

    def _path_coordinate_bounds(self, path_data):
        points = []
        cur = None
        start = None
        last_ctl = None

        def add_point(x, y):
            points.append((x, y))

        for cmd, nums in self._path_commands(path_data):
            op, rel = cmd.lower(), cmd.islower()

            if op == 'm':
                for i in range(0, len(nums) - 1, 2):
                    x, y = nums[i], nums[i + 1]
                    if rel and cur:
                        x += cur[0]
                        y += cur[1]
                    cur = (x, y)
                    if i == 0:
                        start = cur
                    add_point(x, y)
                last_ctl = None

            elif op == 'l' and cur:
                for i in range(0, len(nums) - 1, 2):
                    x, y = nums[i], nums[i + 1]
                    if rel:
                        x += cur[0]
                        y += cur[1]
                    cur = (x, y)
                    add_point(x, y)
                last_ctl = None

            elif op == 'h' and cur:
                for val in nums:
                    x = cur[0] + val if rel else val
                    cur = (x, cur[1])
                    add_point(*cur)
                last_ctl = None

            elif op == 'v' and cur:
                for val in nums:
                    y = cur[1] + val if rel else val
                    cur = (cur[0], y)
                    add_point(*cur)
                last_ctl = None

            elif op in ('q', 't') and cur:
                i = 0
                while i < len(nums):
                    if op == 'q' and i + 3 < len(nums):
                        cx, cy, x, y = nums[i:i + 4]
                        i += 4
                    elif op == 't' and i + 1 < len(nums):
                        x, y = nums[i:i + 2]
                        i += 2
                        if last_ctl:
                            cx = 2 * cur[0] - last_ctl[0]
                            cy = 2 * cur[1] - last_ctl[1]
                        else:
                            cx, cy = cur
                    else:
                        break
                    if rel:
                        cx += cur[0]
                        cy += cur[1]
                        x += cur[0]
                        y += cur[1]
                    add_point(cx, cy)
                    add_point(x, y)
                    cur, last_ctl = (x, y), (cx, cy)

            elif op in ('c', 's') and cur:
                i = 0
                while i < len(nums):
                    if op == 'c' and i + 5 < len(nums):
                        c1x, c1y, c2x, c2y, x, y = nums[i:i + 6]
                        i += 6
                    elif op == 's' and i + 3 < len(nums):
                        c2x, c2y, x, y = nums[i:i + 4]
                        i += 4
                        if last_ctl:
                            c1x = 2 * cur[0] - last_ctl[0]
                            c1y = 2 * cur[1] - last_ctl[1]
                        else:
                            c1x, c1y = cur
                    else:
                        break
                    if rel:
                        c1x += cur[0]
                        c1y += cur[1]
                        c2x += cur[0]
                        c2y += cur[1]
                        x += cur[0]
                        y += cur[1]
                    add_point(c1x, c1y)
                    add_point(c2x, c2y)
                    add_point(x, y)
                    cur, last_ctl = (x, y), (c2x, c2y)

            elif op == 'a' and cur:
                i = 0
                while i + 6 < len(nums):
                    x, y = nums[i + 5], nums[i + 6]
                    i += 7
                    if rel:
                        x += cur[0]
                        y += cur[1]
                    cur = (x, y)
                    add_point(x, y)
                last_ctl = None

            elif op == 'z' and start:
                cur = start
                add_point(*cur)
                last_ctl = None

        if not points:
            return None

        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return min(xs), min(ys), max(xs), max(ys)

    def _fit_svg_coordinate_space_to_paths(self, paths):
        fitted = self._fitted_svg_coordinate_space(paths)
        if fitted is None:
            return
        self.svg_min_x, self.svg_min_y, self.svg_width, self.svg_height = fitted

    def _fitted_svg_coordinate_space(self, paths):
        bounds = [self._path_coordinate_bounds(path) for path in paths]
        bounds = [bound for bound in bounds if bound]
        if not bounds:
            return None

        min_x = min(bound[0] for bound in bounds)
        min_y = min(bound[1] for bound in bounds)
        max_x = max(bound[2] for bound in bounds)
        max_y = max(bound[3] for bound in bounds)
        width = max_x - min_x
        height = max_y - min_y
        if width <= 0 and height <= 0:
            return None

        target_aspect = self._canvas_aspect_ratio()
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        if width <= 0:
            width = height * target_aspect
            min_x = center_x - width / 2
            max_x = center_x + width / 2
        if height <= 0:
            height = width / target_aspect
            min_y = center_y - height / 2
            max_y = center_y + height / 2

        source_aspect = width / height
        if source_aspect > target_aspect:
            height = width / target_aspect
            min_y = center_y - height / 2
            max_y = center_y + height / 2
        else:
            width = height * target_aspect
            min_x = center_x - width / 2
            max_x = center_x + width / 2

        return min_x, min_y, max_x - min_x, max_y - min_y

    def _extract_svg_paths(self, root):
        paths = []

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
                nums = [float(n) for n in re.findall(SVG_NUMBER_RE, pts)]
                pairs = list(zip(nums[0::2], nums[1::2]))
                if not pairs:
                    return None
                path_str = f"M{pairs[0][0]},{pairs[0][1]}"
                for px, py in pairs[1:]:
                    path_str += f" L{px},{py}"
                return path_str
            except:
                return None

        def polygon_to_path(poly):
            path_str = polyline_to_path(poly)
            if path_str:
                path_str += " Z"
            return path_str

        namespaces = {'svg': 'http://www.w3.org/2000/svg'}

        def find_all(el, tag):
            found = el.findall(f".//svg:{tag}", namespaces)
            if not found:
                found = el.findall(f".//{tag}")
            return found

        for p in find_all(root, 'path'):
            d = p.attrib.get('d')
            if d:
                paths.append(d)

        for r in find_all(root, 'rect'):
            rp = rect_to_path(r)
            if rp:
                paths.append(rp)

        for c in find_all(root, 'circle'):
            cp = circle_to_path(c)
            if cp:
                paths.append(cp)

        for e in find_all(root, 'ellipse'):
            ep = ellipse_to_path(e)
            if ep:
                paths.append(ep)

        for l in find_all(root, 'line'):
            lp = line_to_path(l)
            if lp:
                paths.append(lp)

        for pl in find_all(root, 'polyline'):
            plp = polyline_to_path(pl)
            if plp:
                paths.append(plp)

        for pg in find_all(root, 'polygon'):
            pgp = polygon_to_path(pg)
            if pgp:
                paths.append(pgp)

        return paths

    def set_svg_reference(self, svg_file_path):
        if not os.path.exists(svg_file_path):
            print(f"SVG reference file not found: {svg_file_path}")
            return False
        tree = ET.parse(svg_file_path)
        paths = self._extract_svg_paths(tree.getroot())
        fitted = self._fitted_svg_coordinate_space(paths)
        if fitted is None:
            print(f"SVG reference has no drawable bounds: {svg_file_path}")
            return False
        self.svg_reference_space = {
            "source": svg_file_path,
            "min_x": fitted[0],
            "min_y": fitted[1],
            "width": fitted[2],
            "height": fitted[3],
        }
        print(
            f"SVG reference scale set from {svg_file_path}: "
            f"origin=({fitted[0]}, {fitted[1]}) size={fitted[2]} x {fitted[3]}",
            flush=True,
        )
        return True

    def load_default_svg_reference(self):
        if not self.default_svg_reference_path:
            return
        if not self.default_svg_reference_path.exists():
            print(
                f"Default SVG reference not found: {self.default_svg_reference_path}; "
                "reference scale mode will fall back to document/path bounds.",
                flush=True,
            )
            return
        self.set_svg_reference(str(self.default_svg_reference_path))

    def clear_svg_reference(self):
        self.svg_reference_space = None
        print(f"SVG reference scale cleared; active scale mode is {self.svg_scale_mode}.", flush=True)

    def set_svg_scale_mode(self, mode):
        normalized = str(mode).strip().lower().replace('-', '_')
        aliases = {
            'document': 'document',
            'viewbox': 'document',
            'view_box': 'document',
            'reference': 'reference',
            'ref': 'reference',
            'dark_grey': 'reference',
            'dark_gray': 'reference',
            'path': 'path_bounds',
            'paths': 'path_bounds',
            'path_bounds': 'path_bounds',
            'fit_paths': 'path_bounds',
            'fit': 'path_bounds',
        }
        if normalized not in aliases:
            raise ValueError(f"unknown SVG scale mode {mode}; use reference, document, or path_bounds")
        self.svg_scale_mode = aliases[normalized]
        print(f"SVG scale mode set to {self.svg_scale_mode}", flush=True)

    @staticmethod
    def _svg_local_name(tag):
        return str(tag).rsplit('}', 1)[-1]

    @staticmethod
    def _svg_style_value(element, key):
        style = element.attrib.get('style', '')
        for part in style.split(';'):
            if ':' not in part:
                continue
            name, value = part.split(':', 1)
            if name.strip().lower() == key:
                return value.strip().lower()
        return None

    def _svg_element_is_hidden(self, element):
        display = (
            self._svg_style_value(element, 'display')
            or element.attrib.get('display', '').strip().lower()
        )
        visibility = (
            self._svg_style_value(element, 'visibility')
            or element.attrib.get('visibility', '').strip().lower()
        )
        return display == 'none' or visibility == 'hidden'

    def _svg_layer_name(self, element):
        for key in (
            'id',
            'label',
            '{http://www.inkscape.org/namespaces/inkscape}label',
            'inkscape:label',
        ):
            value = element.attrib.get(key)
            if value:
                return str(value).strip()
        return None

    def _svg_layer_matches(self, current_layer, requested_layer):
        if requested_layer is None:
            return True
        if current_layer is None:
            return False
        return str(current_layer).strip().lower() == str(requested_layer).strip().lower()

    def _parse_svg_drawables(self, svg_file_path, layer_name=None, include_hidden=False):
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
                nums = [float(n) for n in re.findall(SVG_NUMBER_RE, pts)]
                pairs = list(zip(nums[0::2], nums[1::2]))
                if not pairs:
                    return None
                path_str = f"M{pairs[0][0]},{pairs[0][1]}"
                for px, py in pairs[1:]:
                    path_str += f" L{px},{py}"
                return path_str
            except:
                return None

        def polygon_to_path(poly):
            path_str = polyline_to_path(poly)
            if path_str:
                path_str += " Z"
            return path_str

        paths = []
        if not os.path.exists(svg_file_path):
            print(f"SVG file not found: {svg_file_path}")
            return None, paths

        tree = ET.parse(svg_file_path)
        root = tree.getroot()
        coordinate_space_source = self._read_svg_coordinate_space(root)

        def append_path(path):
            if path:
                paths.append(path)

        def walk(element, current_layer=None, hidden=False):
            element_hidden = hidden or self._svg_element_is_hidden(element)
            tag = self._svg_local_name(element.tag)
            next_layer = current_layer
            if tag == 'g':
                group_layer = self._svg_layer_name(element)
                if group_layer:
                    next_layer = group_layer

            eligible = (
                (include_hidden or not element_hidden)
                and self._svg_layer_matches(next_layer, layer_name)
            )

            if eligible:
                if tag == 'path':
                    append_path(element.attrib.get('d'))
                elif tag == 'rect':
                    append_path(rect_to_path(element))
                elif tag == 'circle':
                    append_path(circle_to_path(element))
                elif tag == 'ellipse':
                    append_path(ellipse_to_path(element))
                elif tag == 'line':
                    append_path(line_to_path(element))
                elif tag == 'polyline':
                    append_path(polyline_to_path(element))
                elif tag == 'polygon':
                    append_path(polygon_to_path(element))

            for child in list(element):
                walk(child, next_layer, element_hidden)

        walk(root)

        if self.svg_scale_mode == 'reference' and self.svg_reference_space is not None:
            self.svg_min_x = self.svg_reference_space["min_x"]
            self.svg_min_y = self.svg_reference_space["min_y"]
            self.svg_width = self.svg_reference_space["width"]
            self.svg_height = self.svg_reference_space["height"]
            coordinate_space_source = f"reference:{self.svg_reference_space['source']}"
        elif self.svg_scale_mode == 'path_bounds':
            self._fit_svg_coordinate_space_to_paths(paths)
            coordinate_space_source = "path_bounds_forced"
        elif coordinate_space_source is None:
            self._fit_svg_coordinate_space_to_paths(paths)
            coordinate_space_source = "path_bounds_fallback"

        print(
            f"Parsed {len(paths)} drawable paths from {svg_file_path}. "
            f"layer={layer_name or 'visible'} include_hidden={int(include_hidden)}. "
            f"SVG coordinate space source={coordinate_space_source}: "
            f"({self.svg_min_x}, {self.svg_min_y}) "
            f"{self.svg_width} x {self.svg_height}."
        )
        return coordinate_space_source, paths

    def parse_svg_file(self, svg_file_path):
        """
        Parse visible SVG drawables. Hidden Illustrator/Inkscape layers are skipped.
        """
        _source, paths = self._parse_svg_drawables(svg_file_path)
        return paths

    def parse_svg_layer_file(self, svg_file_path, layer_name):
        """
        Parse drawables from one named SVG layer, even if that layer is hidden.
        """
        _source, paths = self._parse_svg_drawables(
            svg_file_path,
            layer_name=layer_name,
            include_hidden=True,
        )
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

    def _empty_bci_slot(self, slot):
        slot = int(slot)
        return {
            "slot": slot,
            "label": f"svg{slot + 1}",
            "svg_path": None,
            "strokes": [],
            "stroke_index": 0,
            "coordinate_space": None,
            "canvas_region": self._bci_canvas_region_for_slot(slot),
            "finished": False,
        }

    def _bci_canvas_region_for_slot(self, slot):
        return [0.0, 1.0, 0.0, 1.0]

    def _capture_svg_coordinate_space(self):
        return {
            "min_x": float(self.svg_min_x),
            "min_y": float(self.svg_min_y),
            "width": float(self.svg_width),
            "height": float(self.svg_height),
        }

    def _apply_svg_coordinate_space(self, space):
        if not space:
            return
        self.svg_min_x = float(space["min_x"])
        self.svg_min_y = float(space["min_y"])
        self.svg_width = float(space["width"])
        self.svg_height = float(space["height"])

    def _resolve_svg_file_path(self, svg_file_path):
        raw = str(svg_file_path).strip()
        candidate = Path(raw).expanduser()
        if candidate.is_absolute() and candidate.exists():
            return candidate
        search_paths = [
            candidate,
            Path.cwd() / candidate,
            Path(__file__).resolve().parent / candidate,
            self.bci_state_path.parent / candidate,
        ]
        for path in search_paths:
            if path.exists():
                return path.resolve()
        return candidate

    @staticmethod
    def _normalize_bci_attention(raw_attention):
        if isinstance(raw_attention, str):
            normalized = raw_attention.strip().lower().replace("-", "_")
            aliases = {
                "not_focused": 0,
                "unfocused": 0,
                "low": 0,
                "0": 0,
                "half": 1,
                "half_focused": 1,
                "medium": 1,
                "mid": 1,
                "1": 1,
                "focused": 2,
                "high": 2,
                "2": 2,
            }
            if normalized in aliases:
                return aliases[normalized]
        attention = int(raw_attention)
        if attention < 0 or attention > 2:
            raise ValueError(f"attention must be 0..2, got {raw_attention}")
        return attention

    @staticmethod
    def _normalize_bci_svg_slot(raw_slot):
        if isinstance(raw_slot, str):
            normalized = raw_slot.strip().lower().replace("-", "_")
            aliases = {
                "0": 0,
                "1": 0,
                "svg1": 0,
                "svg_1": 0,
                "not_focused": 0,
                "unfocused": 0,
                "2": 1,
                "svg2": 1,
                "svg_2": 1,
                "half": 1,
                "half_focused": 1,
                "3": 2,
                "svg3": 2,
                "svg_3": 2,
                "focused": 2,
            }
            if normalized in aliases:
                return aliases[normalized]
        slot = int(raw_slot)
        if slot == 0:
            return slot
        if slot in (1, 2, 3):
            return slot - 1
        raise ValueError(f"BCI SVG slot must be svg1..svg3, got {raw_slot}")

    def _bci_slot_summary(self, slot):
        slot_data = self.bci_slots[int(slot)]
        total = len(slot_data["strokes"])
        index = int(slot_data["stroke_index"])
        return {
            "slot": int(slot),
            "svg_number": int(slot) + 1,
            "attention_state": int(slot),
            "svg_path": slot_data["svg_path"],
            "svg_layer": slot_data.get("svg_layer"),
            "stroke_index": index,
            "total_strokes": total,
            "finished": bool(total > 0 and index >= total),
        }

    def print_bci_counters(self, label="status"):
        print(f"\nBCI SVG COUNTERS ({label})", flush=True)
        for slot in range(3):
            summary = self._bci_slot_summary(slot)
            path = summary["svg_path"] or "not loaded"
            finished = " finished" if summary["finished"] else ""
            selected = " <- selected" if slot == self.bci_attention_state else ""
            print(
                f"  SVG {summary['svg_number']}: "
                f"{summary['stroke_index']}/{summary['total_strokes']} strokes"
                f"{finished}{selected} | {path}",
                flush=True,
            )
        print(
            f"  EMG pending={int(self.bci_pending_emg)} | "
            f"color cooldown={self.bci_strokes_since_color_change}/"
            f"{self.bci_color_cooldown_strokes} strokes | "
            f"current brush={self.bci_current_color_brush}\n",
            flush=True,
        )

    def bci_status_data(self):
        return {
            "schema": 1,
            "attention_state": int(self.bci_attention_state),
            "attention_svg_number": int(self.bci_attention_state) + 1,
            "pending_emg": bool(self.bci_pending_emg),
            "strokes_since_color_change": int(self.bci_strokes_since_color_change),
            "color_cooldown_strokes": int(self.bci_color_cooldown_strokes),
            "current_color_brush": int(self.bci_current_color_brush),
            "color_brush_count": int(self.bci_color_brush_count),
            "slots": {
                str(slot): self._bci_slot_summary(slot)
                for slot in range(3)
            },
        }

    def save_bci_paint_state(self):
        data = self.bci_status_data()
        data["updated_at_unix"] = time.time()
        self.bci_state_path.parent.mkdir(parents=True, exist_ok=True)
        self.bci_state_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    def load_bci_paint_state(self):
        if not self.bci_state_path.exists():
            print(
                f"BCI paint state: no saved state at {self.bci_state_path}; starting fresh.",
                flush=True,
            )
            return
        try:
            data = json.loads(self.bci_state_path.read_text(encoding="utf-8"))
            self.bci_attention_state = self._normalize_bci_attention(
                data.get("attention_state", 0)
            )
            self.bci_pending_emg = bool(data.get("pending_emg", False))
            self.bci_strokes_since_color_change = int(
                data.get(
                    "strokes_since_color_change",
                    self.bci_color_cooldown_strokes,
                )
            )
            self.bci_current_color_brush = int(data.get("current_color_brush", 0))
            slots = data.get("slots", {})
            previous_space = self._capture_svg_coordinate_space()
            try:
                for key, slot_data in slots.items():
                    slot = int(key)
                    if slot < 0 or slot > 2:
                        continue
                    svg_path = slot_data.get("svg_path")
                    if svg_path:
                        svg_layer = slot_data.get("svg_layer")
                        if svg_layer:
                            self.load_bci_svg_layer(
                                f"svg{slot + 1}",
                                svg_path,
                                svg_layer,
                                stroke_index=int(slot_data.get("stroke_index", 0)),
                                persist=False,
                            )
                        else:
                            self.load_bci_svg(
                                f"svg{slot + 1}",
                                svg_path,
                                stroke_index=int(slot_data.get("stroke_index", 0)),
                                persist=False,
                            )
            finally:
                self._apply_svg_coordinate_space(previous_space)
            print(
                f"BCI paint state: loaded saved counters from {self.bci_state_path}",
                flush=True,
            )
            self.print_bci_counters("loaded saved state")
        except Exception as exc:
            print(
                f"BCI paint state: failed to load {self.bci_state_path}: {exc}; starting fresh.",
                flush=True,
            )

    def load_bci_svg(self, slot, svg_file_path, stroke_index=0, persist=True):
        slot = self._normalize_bci_svg_slot(slot)
        resolved = self._resolve_svg_file_path(svg_file_path)
        previous_space = self._capture_svg_coordinate_space()
        try:
            strokes = self.parse_svg_file(str(resolved))
            coordinate_space = self._capture_svg_coordinate_space()
        finally:
            self._apply_svg_coordinate_space(previous_space)
        if not strokes:
            raise ValueError(f"BCI SVG {slot + 1} has no drawable strokes: {resolved}")
        clamped_index = max(0, min(int(stroke_index), len(strokes)))
        self.bci_slots[slot] = {
            "slot": slot,
            "label": f"svg{slot + 1}",
            "svg_path": str(resolved),
            "svg_layer": None,
            "strokes": strokes,
            "stroke_index": clamped_index,
            "coordinate_space": coordinate_space,
            "canvas_region": self._bci_canvas_region_for_slot(slot),
            "finished": clamped_index >= len(strokes),
        }
        print(
            f"BCI SVG {slot + 1}: loaded {len(strokes)} strokes from {resolved}; "
            f"counter={clamped_index}",
            flush=True,
        )
        if persist:
            self.save_bci_paint_state()
        self.osc_client.send_message("/bci/load", [slot + 1, len(strokes), str(resolved)])
        self.print_bci_counters(f"loaded SVG {slot + 1}")
        return self._bci_slot_summary(slot)

    def load_bci_svg_layer(self, slot, svg_file_path, layer_name, stroke_index=0, persist=True):
        slot = self._normalize_bci_svg_slot(slot)
        resolved = self._resolve_svg_file_path(svg_file_path)
        previous_space = self._capture_svg_coordinate_space()
        previous_scale_mode = self.svg_scale_mode
        try:
            self.svg_scale_mode = 'document'
            strokes = self.parse_svg_layer_file(str(resolved), layer_name)
            coordinate_space = self._capture_svg_coordinate_space()
        finally:
            self.svg_scale_mode = previous_scale_mode
            self._apply_svg_coordinate_space(previous_space)
        if not strokes:
            raise ValueError(
                f"BCI SVG {slot + 1} layer {layer_name!r} has no drawable strokes: {resolved}"
            )
        clamped_index = max(0, min(int(stroke_index), len(strokes)))
        self.bci_slots[slot] = {
            "slot": slot,
            "label": f"svg{slot + 1}",
            "svg_path": str(resolved),
            "svg_layer": str(layer_name),
            "strokes": strokes,
            "stroke_index": clamped_index,
            "coordinate_space": coordinate_space,
            "canvas_region": self._bci_canvas_region_for_slot(slot),
            "finished": clamped_index >= len(strokes),
        }
        print(
            f"BCI SVG {slot + 1}: loaded {len(strokes)} strokes from "
            f"{resolved} layer={layer_name}; counter={clamped_index}",
            flush=True,
        )
        if persist:
            self.save_bci_paint_state()
        self.osc_client.send_message(
            "/bci/load_layer",
            [slot + 1, len(strokes), str(resolved), str(layer_name)],
        )
        self.print_bci_counters(f"loaded SVG {slot + 1} layer {layer_name}")
        return self._bci_slot_summary(slot)

    def load_bci_svg_layers(self, svg_file_path, layer_names, persist=True):
        if len(layer_names) != 3:
            raise ValueError("load_layers requires exactly three layer names")
        summaries = []
        for slot, layer_name in enumerate(layer_names):
            summaries.append(
                self.load_bci_svg_layer(
                    f"svg{slot + 1}",
                    svg_file_path,
                    layer_name,
                    stroke_index=0,
                    persist=False,
                )
            )
        if persist:
            self.save_bci_paint_state()
        preview_path = self.write_bci_layer_preview()
        self.osc_client.send_message(
            "/bci/load_layers",
            [str(svg_file_path), str(preview_path)] + [str(name) for name in layer_names],
        )
        return {
            "slots": summaries,
            "preview_path": str(preview_path),
        }

    def _sample_svg_path_points(self, path_data, steps=12):
        points = []
        cur = None
        start = None
        last_ctl = None

        def add(x, y):
            points.append((float(x), float(y)))

        def cubic(p0, p1, p2, p3, t):
            mt = 1.0 - t
            return (
                mt ** 3 * p0[0] + 3 * mt ** 2 * t * p1[0] + 3 * mt * t ** 2 * p2[0] + t ** 3 * p3[0],
                mt ** 3 * p0[1] + 3 * mt ** 2 * t * p1[1] + 3 * mt * t ** 2 * p2[1] + t ** 3 * p3[1],
            )

        def quad(p0, p1, p2, t):
            mt = 1.0 - t
            return (
                mt ** 2 * p0[0] + 2 * mt * t * p1[0] + t ** 2 * p2[0],
                mt ** 2 * p0[1] + 2 * mt * t * p1[1] + t ** 2 * p2[1],
            )

        for cmd, nums in self._path_commands(path_data):
            op, rel = cmd.lower(), cmd.islower()
            if op == 'm':
                for i in range(0, len(nums) - 1, 2):
                    x, y = nums[i], nums[i + 1]
                    if rel and cur:
                        x += cur[0]
                        y += cur[1]
                    cur = (x, y)
                    if i == 0:
                        start = cur
                    add(x, y)
                last_ctl = None
            elif op == 'l' and cur:
                for i in range(0, len(nums) - 1, 2):
                    x, y = nums[i], nums[i + 1]
                    if rel:
                        x += cur[0]
                        y += cur[1]
                    cur = (x, y)
                    add(x, y)
                last_ctl = None
            elif op == 'h' and cur:
                for val in nums:
                    cur = (cur[0] + val if rel else val, cur[1])
                    add(*cur)
                last_ctl = None
            elif op == 'v' and cur:
                for val in nums:
                    cur = (cur[0], cur[1] + val if rel else val)
                    add(*cur)
                last_ctl = None
            elif op in ('c', 's') and cur:
                i = 0
                while i < len(nums):
                    if op == 'c' and i + 5 < len(nums):
                        c1x, c1y, c2x, c2y, x, y = nums[i:i + 6]
                        i += 6
                    elif op == 's' and i + 3 < len(nums):
                        c2x, c2y, x, y = nums[i:i + 4]
                        i += 4
                        if last_ctl:
                            c1x = 2 * cur[0] - last_ctl[0]
                            c1y = 2 * cur[1] - last_ctl[1]
                        else:
                            c1x, c1y = cur
                    else:
                        break
                    if rel:
                        c1x += cur[0]
                        c1y += cur[1]
                        c2x += cur[0]
                        c2y += cur[1]
                        x += cur[0]
                        y += cur[1]
                    p0 = cur
                    p1 = (c1x, c1y)
                    p2 = (c2x, c2y)
                    p3 = (x, y)
                    for step in range(1, steps + 1):
                        add(*cubic(p0, p1, p2, p3, step / steps))
                    cur, last_ctl = p3, p2
            elif op in ('q', 't') and cur:
                i = 0
                while i < len(nums):
                    if op == 'q' and i + 3 < len(nums):
                        cx, cy, x, y = nums[i:i + 4]
                        i += 4
                    elif op == 't' and i + 1 < len(nums):
                        x, y = nums[i:i + 2]
                        i += 2
                        if last_ctl:
                            cx = 2 * cur[0] - last_ctl[0]
                            cy = 2 * cur[1] - last_ctl[1]
                        else:
                            cx, cy = cur
                    else:
                        break
                    if rel:
                        cx += cur[0]
                        cy += cur[1]
                        x += cur[0]
                        y += cur[1]
                    p0 = cur
                    p1 = (cx, cy)
                    p2 = (x, y)
                    for step in range(1, steps + 1):
                        add(*quad(p0, p1, p2, step / steps))
                    cur, last_ctl = p2, p1
            elif op == 'a' and cur:
                i = 0
                while i + 6 < len(nums):
                    x, y = nums[i + 5], nums[i + 6]
                    i += 7
                    if rel:
                        x += cur[0]
                        y += cur[1]
                    cur = (x, y)
                    add(x, y)
                last_ctl = None
            elif op == 'z' and start:
                cur = start
                add(*cur)
                last_ctl = None
        return points

    @staticmethod
    def _png_chunk(chunk_type, data):
        return (
            struct.pack(">I", len(data))
            + chunk_type
            + data
            + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        )

    def _write_rgb_png(self, path, width, height, pixels):
        rows = []
        for y in range(height):
            start = y * width * 3
            rows.append(b'\x00' + bytes(pixels[start:start + width * 3]))
        raw = b''.join(rows)
        data = b'\x89PNG\r\n\x1a\n'
        data += self._png_chunk(b'IHDR', struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        data += self._png_chunk(b'IDAT', zlib.compress(raw, 9))
        data += self._png_chunk(b'IEND', b'')
        path.write_bytes(data)

    def write_bci_layer_preview(self, output_path=None):
        preview_space = None
        for slot in range(3):
            preview_space = self.bci_slots[slot].get("coordinate_space")
            if preview_space:
                break
        max_side = max(300, int(BCI_LAYER_PREVIEW_SIZE))
        if preview_space and preview_space.get("width") and preview_space.get("height"):
            aspect = float(preview_space["width"]) / float(preview_space["height"])
            if aspect >= 1.0:
                width = max_side
                height = max(1, int(round(max_side / aspect)))
            else:
                height = max_side
                width = max(1, int(round(max_side * aspect)))
        else:
            width = max_side
            height = max_side
        pixels = bytearray([255] * width * height * 3)
        colors = [(235, 45, 45), (35, 175, 70), (50, 90, 235)]

        def draw_point(px, py, color):
            if 0 <= px < width and 0 <= py < height:
                idx = (py * width + px) * 3
                pixels[idx:idx + 3] = bytes(color)

        def draw_line(a, b, color):
            x0, y0 = a
            x1, y1 = b
            dx = abs(x1 - x0)
            dy = -abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx + dy
            while True:
                draw_point(x0, y0, color)
                draw_point(x0 + 1, y0, color)
                draw_point(x0, y0 + 1, color)
                if x0 == x1 and y0 == y1:
                    break
                e2 = 2 * err
                if e2 >= dy:
                    err += dy
                    x0 += sx
                if e2 <= dx:
                    err += dx
                    y0 += sy

        def map_point(x, y, space):
            nx = (x - space["min_x"]) / space["width"] if space["width"] else 0.0
            ny = (y - space["min_y"]) / space["height"] if space["height"] else 0.0
            px = int(round(nx * (width - 1)))
            py = int(round(ny * (height - 1)))
            return px, py

        for slot in range(3):
            slot_data = self.bci_slots[slot]
            space = slot_data.get("coordinate_space")
            if not space:
                continue
            color = colors[slot]
            for stroke in slot_data.get("strokes", []):
                samples = self._sample_svg_path_points(stroke)
                if len(samples) < 2:
                    continue
                mapped = [map_point(x, y, space) for x, y in samples]
                for start, end in zip(mapped, mapped[1:]):
                    draw_line(start, end, color)

        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            source = Path(self.bci_slots[0].get("svg_path") or "layers").stem
            output_path = BCI_LAYER_PREVIEW_DIR / f"{timestamp}_{source}_layers_preview.png"
        output_path = Path(output_path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_rgb_png(output_path, width, height, pixels)
        print(f"BCI layer preview written: {output_path}", flush=True)
        return output_path

    def set_bci_attention(self, raw_attention):
        attention = self._normalize_bci_attention(raw_attention)
        previous = int(self.bci_attention_state)
        self.bci_attention_state = attention
        self.save_bci_paint_state()
        print(
            f"BCI attention state set to {attention}; selecting SVG {attention + 1}",
            flush=True,
        )
        self.osc_client.send_message("/bci/attention", [attention, attention + 1])
        return {"previous": previous, "current": attention, "changed": previous != attention}

    def trigger_bci_emg(self, value=1):
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"false", "off", "no"}:
                active = False
            elif normalized in {"true", "on", "yes"}:
                active = True
            else:
                active = abs(float(normalized)) > 0.0
        else:
            active = abs(float(value)) > 0.0
        if active:
            self.bci_pending_emg = True
            print(
                "BCI EMG trigger received; color change is pending at the next stroke boundary.",
                flush=True,
            )
        else:
            self.bci_pending_emg = False
            print("BCI EMG trigger cleared.", flush=True)
        self.save_bci_paint_state()
        self.osc_client.send_message(
            "/bci/emg",
            [
                int(self.bci_pending_emg),
                int(self.bci_strokes_since_color_change),
                int(self.bci_color_cooldown_strokes),
            ],
        )
        return self.bci_pending_emg

    def reset_bci_paint(self):
        for slot in range(3):
            self.bci_slots[slot]["stroke_index"] = 0
            self.bci_slots[slot]["finished"] = False
        self.bci_pending_emg = False
        self.bci_stop_requested = False
        self.bci_strokes_since_color_change = self.bci_color_cooldown_strokes
        self.save_bci_paint_state()
        print("BCI paint state reset: all SVG counters are 0.", flush=True)
        self.osc_client.send_message("/bci/reset", [1])
        self.print_bci_counters("reset")
        return self.bci_status_data()

    def set_bci_svg_counter(self, slot, index):
        slot = self._normalize_bci_svg_slot(slot)
        slot_data = self.bci_slots[slot]
        total = len(slot_data["strokes"])
        clamped_index = max(0, min(int(index), total))
        slot_data["stroke_index"] = clamped_index
        slot_data["finished"] = total > 0 and clamped_index >= total
        self.save_bci_paint_state()
        print(
            f"BCI SVG {slot + 1}: stroke counter set to {clamped_index}/{total}",
            flush=True,
        )
        self.osc_client.send_message("/bci/goto", [slot + 1, clamped_index, total])
        self.print_bci_counters(f"counter set SVG {slot + 1}")
        return self._bci_slot_summary(slot)

    def draw_bci_next_stroke(self, speed=None):
        slot = int(self.bci_attention_state)
        slot_data = self.bci_slots[slot]
        total = len(slot_data["strokes"])
        index = int(slot_data["stroke_index"])
        if total == 0:
            raise ValueError(f"BCI SVG {slot + 1} is not loaded")
        if index >= total:
            slot_data["finished"] = True
            self.save_bci_paint_state()
            print(f"BCI SVG {slot + 1} is finished; no stroke drawn.", flush=True)
            self.osc_client.send_message("/bci/finished", [slot + 1, index, total])
            self.print_bci_counters(f"SVG {slot + 1} finished")
            return {
                "drawn": False,
                "slot": slot,
                "svg_number": slot + 1,
                "stroke_index": index,
                "total_strokes": total,
            }

        previous_space = self._capture_svg_coordinate_space()
        previous_region = self._active_svg_canvas_region
        self._apply_svg_coordinate_space(slot_data["coordinate_space"])
        self._active_svg_canvas_region = tuple(
            float(value)
            for value in slot_data.get(
                "canvas_region",
                self._bci_canvas_region_for_slot(slot),
            )
        )
        print(
            f"BCI paint: attention={self.bci_attention_state} SVG {slot + 1} "
            f"stroke {index + 1}/{total} region={self._active_svg_canvas_region}",
            flush=True,
        )
        try:
            self.draw_svg_stroke(slot_data["strokes"][index], speed=speed)
        finally:
            self._apply_svg_coordinate_space(previous_space)
            self._active_svg_canvas_region = previous_region

        slot_data["stroke_index"] = index + 1
        slot_data["finished"] = slot_data["stroke_index"] >= total
        self.bci_strokes_since_color_change += 1
        global muse_strokes_since_brush_change
        muse_strokes_since_brush_change += 1
        self.save_bci_paint_state()
        payload = [
            int(self.bci_attention_state),
            slot + 1,
            slot_data["stroke_index"],
            total,
        ]
        print(f"OSC send: /bci/stroke_done {payload}", flush=True)
        self.osc_client.send_message("/bci/stroke_done", payload)
        self.print_bci_counters(f"after SVG {slot + 1} stroke {slot_data['stroke_index']}")
        return {
            "drawn": True,
            "slot": slot,
            "svg_number": slot + 1,
            "stroke_index": slot_data["stroke_index"],
            "total_strokes": total,
        }

    def apply_bci_pending_color_change(self, brush_selector_instance):
        if not self.bci_pending_emg:
            return None
        remaining = self.bci_color_cooldown_strokes - self.bci_strokes_since_color_change
        if remaining > 0:
            print(
                f"BCI color change pending; waiting {remaining} more stroke(s) for cooldown.",
                flush=True,
            )
            self.osc_client.send_message(
                "/bci/color_wait",
                [
                    int(remaining),
                    int(self.bci_strokes_since_color_change),
                    int(self.bci_color_cooldown_strokes),
                ],
            )
            return None

        if brush_selector_instance is None:
            raise RuntimeError("BCI color change requested but BrushSelector is not initialized")

        selector_state = brush_selector_instance.status()
        current = self.bci_current_color_brush
        if selector_state.get("HOLDING_BRUSH"):
            current = int(selector_state.get("CURRENT_BRUSH", current))
        if current < self.bci_color_brush_min or current > self.bci_color_brush_max:
            current = 0
        if current == 0 or current >= self.bci_color_brush_max:
            next_brush = self.bci_color_brush_min
        else:
            next_brush = current + 1
        print(
            f"BCI color change: EMG cycling brush {current or 'none'} -> {next_brush}",
            flush=True,
        )
        state = brush_selector_instance.ready(next_brush)
        self.bci_current_color_brush = int(state.get("CURRENT_BRUSH", next_brush))
        self.bci_pending_emg = False
        self.bci_strokes_since_color_change = 0
        self.save_bci_paint_state()
        self.osc_client.send_message(
            "/bci/color",
            [self.bci_current_color_brush, int(self.bci_strokes_since_color_change)],
        )
        return state

    def draw_bci_stroke_for_event(self, reason, speed=None):
        try:
            result = self.draw_bci_next_stroke(speed=speed)
        except Exception as exc:
            print(f"BCI event stroke skipped for {reason}: {exc}", flush=True)
            self.osc_client.send_message("/bci/event_stroke_error", [str(reason), str(exc)])
            return {"drawn": False, "reason": str(reason), "error": str(exc)}
        result["reason"] = str(reason)
        print(f"BCI event stroke for {reason}: {result}", flush=True)
        self.osc_client.send_message(
            "/bci/event_stroke",
            [
                str(reason),
                int(result.get("drawn", False)),
                int(result.get("svg_number", 0)),
                int(result.get("stroke_index", 0)),
                int(result.get("total_strokes", 0)),
            ],
        )
        return result

    def send_bci_status(self):
        status = self.bci_status_data()
        self.osc_client.send_message(
            "/bci/status",
            [
                status["attention_state"],
                status["attention_svg_number"],
                int(status["pending_emg"]),
                status["strokes_since_color_change"],
                status["color_cooldown_strokes"],
                status["current_color_brush"],
            ],
        )
        for slot in range(3):
            summary = status["slots"][str(slot)]
            self.osc_client.send_message(
                "/bci/svg_status",
                [
                    summary["svg_number"],
                    summary["attention_state"],
                    summary["stroke_index"],
                    summary["total_strokes"],
                    int(summary["finished"]),
                    summary["svg_path"] or "",
                ],
            )
        print(f"BCI status: {status}", flush=True)
        self.print_bci_counters("status")
        return status

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
        previous_preserved_wrist = getattr(self, '_preserved_wrist_orientation', None)
        self._preserved_wrist_orientation = self._current_wrist_orientation()
        print(
            f"SVG drawing: preserving wrist orientation "
            f"{self._preserved_wrist_orientation}",
            flush=True,
        )
        if speed is None:
            base_speed = self._speed_mm_s()
        else:
            base_speed = speed
        # ---------- helpers ----------
        cmd_pat = r'([MLHVCSQTAZmlhvcsqtaz])([^MLHVCSQTAZmlhvcsqtaz]*)'
        num_pat = SVG_NUMBER_RE

        def parse_path(d):
            for cmd, params in re.findall(cmd_pat, d):
                yield cmd, [float(n) for n in re.findall(num_pat, params)]

        def clamp(x, y, z):
            if self._is_table_surface():
                return x, y, self._paint_z(z)
            y_min, y_max, z_min, z_max = self._svg_canvas_bounds()

            y_lower = y_min
            y_upper = y_max

            y = min(max(y, y_lower), y_upper)

            z = min(max(z, z_min), z_max)
            # Keep X tied to the calibrated canvas plane; the live trim changes
            # only the painting height so Max can compensate while a stroke runs.
            return self.x_from_z(z) + 10, y, self._paint_z(z)
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
                    rx, ry, rz = clamp(rx, ry, rz)

                    if not pen_down:                     # first sub-path
                        self.approach_canvas(rx, ry, rz, speed=draw_spd)
                        pen_down  = True
                    else:                                # subsequent “M” inside <path>
                        x_lift = 5
                        z_lift = 0
                        if self._is_table_surface():
                            self._set_position_keep_wrist(z=last_rb[2] + z_lift,
                                                          speed=draw_spd, wait=True)
                            self._set_position_keep_wrist(x=rx, y=ry, z=rz + z_lift,
                                                          speed=draw_spd, wait=True)
                            self._set_position_keep_wrist(z=rz,
                                                          speed=draw_spd, wait=True)
                        else:
                            self._set_position_keep_wrist(x=last_rb[0] - x_lift,
                                                          y=last_rb[1], z=last_rb[2],
                                                          speed=draw_spd, wait=True)
                            self._set_position_keep_wrist(x=rx - x_lift, y=ry, z=rz,
                                                          speed=draw_spd, wait=True)
                            self._set_position_keep_wrist(x=rx, y=ry, z=rz,
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
                    rx_target, ry_target, rz_target = clamp(rx_target, ry_target, rz_target) # Clamp after jitter

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
                            self._set_position_keep_wrist(x=segment_start_rb[0], y=segment_start_rb[1], z=segment_start_rb[2],
                                                          speed=draw_spd, wait=True) # move to XY (still lifted)
                            self._set_position_keep_wrist(x=segment_start_rb[0], y=segment_start_rb[1], z=segment_end_rb[2], # Lower to drawing Z
                                                          speed=draw_spd / 1.5, wait=True)

                        # Normal continuous line segment
                        self._set_position_keep_wrist(x=segment_end_rb[0], y=segment_end_rb[1], z=segment_end_rb[2],
                                                      speed=draw_spd, wait=False)
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
                        rx_target, ry_target, rz_target = clamp(rx_target, ry_target, rz_target)
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
                            self._set_position_keep_wrist(x=segment_end_rb[0] - self.STIPPLE_LIFT_MM,
                                                          y=segment_end_rb[1], z=segment_end_rb[2],
                                                          speed=draw_spd, wait=True)
                            # Tap down
                            self._set_position_keep_wrist(x=segment_end_rb[0], y=segment_end_rb[1], z=segment_end_rb[2],
                                                          speed=draw_spd / 1.5, wait=True)
                            # Lift up along –X
                            last_rb = self.retract_pen(segment_end_rb[0], segment_end_rb[1],
                                                       segment_end_rb[2], draw_spd)
                            setattr(self, '_stipple_was_active', True)
                        else:
                            # Continuous drawing for curve segment
                            if k_pt == 0 and getattr(self, '_stipple_was_active', False): # Transition from stipple to continuous
                                # segment_start_rb is already lifted. Move to its XY, then lower to segment_end_rb's Z.
                                self._set_position_keep_wrist(x=segment_start_rb[0], y=segment_start_rb[1], z=segment_start_rb[2], # XY of start (lifted)
                                                              speed=draw_spd, wait=True)
                                self._set_position_keep_wrist(x=segment_start_rb[0], y=segment_start_rb[1], z=segment_end_rb[2], # Lower to target Z
                                                              speed=draw_spd / 1.5, wait=True)

                            self._set_position_keep_wrist(x=segment_end_rb[0], y=segment_end_rb[1], z=segment_end_rb[2],
                                                          speed=draw_spd, wait=False)
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
                        rx_target, ry_target, rz_target = clamp(rx_target, ry_target, rz_target)
                        segment_end_rb = (rx_target, ry_target, rz_target)

                        if stipple_level > self.STIPPLE_THRESHOLD:
                            if k_pt == 0 and pen_down and not getattr(self, '_stipple_was_active', False):
                                segment_start_rb = self.retract_pen(segment_start_rb[0],
                                                                    segment_start_rb[1],
                                                                    segment_start_rb[2],
                                                                    draw_spd)

                                self._set_position_keep_wrist(x=segment_end_rb[0] - self.STIPPLE_LIFT_MM,
                                                              y=segment_end_rb[1], z=segment_end_rb[2],
                                                              speed=draw_spd, wait=True)
                                self._set_position_keep_wrist(x=segment_end_rb[0], y=segment_end_rb[1], z=segment_end_rb[2],
                                                              speed=draw_spd / 1.5, wait=True)
                                last_rb = self.retract_pen(segment_end_rb[0], segment_end_rb[1],
                                                           segment_end_rb[2], draw_spd)
                                setattr(self, '_stipple_was_active', True)
                        else:
                            if k_pt == 0 and getattr(self, '_stipple_was_active', False):
                                self._set_position_keep_wrist(x=segment_start_rb[0], y=segment_start_rb[1], z=segment_start_rb[2],
                                                              speed=draw_spd, wait=True)
                                self._set_position_keep_wrist(x=segment_start_rb[0], y=segment_start_rb[1], z=segment_end_rb[2],
                                                              speed=draw_spd / 1.5, wait=True)

                            self._set_position_keep_wrist(x=segment_end_rb[0], y=segment_end_rb[1], z=segment_end_rb[2],
                                                          speed=draw_spd, wait=False)
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
                    rx_target, ry_target, rz_target = clamp(rx, ry, rz) # rx,ry,rz were from self.map_svg_to_robot(x,y)

                    segment_end_rb = (rx_target, ry_target, rz_target)

                    if stipple_level > self.STIPPLE_THRESHOLD:
                        # Ensure pen is lifted before starting stipple segment
                        if pen_down and not getattr(self, '_stipple_was_active', False):
                             last_rb = self.retract_pen(last_rb[0], last_rb[1], last_rb[2], draw_spd)

                        last_rb = self._stipple_segment_runner(last_rb, segment_end_rb, draw_spd, stipple_level)
                    else:
                        # If previous segment was stippling, need to move to start of continuous line and lower pen
                        if getattr(self, '_stipple_was_active', False):
                            self._set_position_keep_wrist(x=last_rb[0], y=last_rb[1], z=last_rb[2], # XY of current (lifted), Z of current (lifted)
                                                          speed=draw_spd, wait=True)
                            self._set_position_keep_wrist(x=last_rb[0], y=last_rb[1], z=segment_end_rb[2], # Lower to target Z for drawing
                                                          speed=draw_spd / 1.5, wait=True)

                        self._set_position_keep_wrist(x=segment_end_rb[0], y=segment_end_rb[1], z=segment_end_rb[2],
                                                      speed=draw_spd, wait=True) # Arcs are usually single moves
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
                        self._set_position_keep_wrist(x=last_rb[0], y=last_rb[1], z=last_rb[2], # XY of current (lifted)
                                                      speed=draw_spd, wait=True)
                        self._set_position_keep_wrist(x=last_rb[0], y=last_rb[1], z=segment_end_rb[2], # Lower to target Z
                                                      speed=draw_spd / 1.5, wait=True)

                    self._set_position_keep_wrist(x=segment_end_rb[0], y=segment_end_rb[1], z=segment_end_rb[2],
                                                  speed=draw_spd, wait=True) # wait=True for Z
                    last_rb = segment_end_rb
                    setattr(self, '_stipple_was_active', False)

                cur = start_svg # SVG current point resets to start of sub-path
                last_ctl = None

            # ---- unrecognised commands ignored ----

        # --------------- PEN UP & RETREAT ---------------
        # If the last operation was stippling, last_rb is already off the canvas.
        # retreat_from_canvas applies the correct surface-normal retreat for table/easel mode.
        if pen_down and last_rb:
            final_z_for_retreat = last_rb[2]
            if getattr(self, '_stipple_was_active', False):
                # retract_pen keeps Z constant, so no adjustment needed
                final_z_for_retreat = last_rb[2]

            self.retreat_from_canvas(last_rb[0], last_rb[1], final_z_for_retreat, speed=self._speed_mm_s())
            # Clear the stippling state for the next independent stroke
            if hasattr(self, '_stipple_was_active'): # Check attribute on self
                delattr(self, '_stipple_was_active') # Delete attribute from self
        if previous_preserved_wrist is None:
            delattr(self, '_preserved_wrist_orientation')
        else:
            self._preserved_wrist_orientation = previous_preserved_wrist


    def map_svg_to_robot(self, svg_x, svg_y):
        """
        Convert an SVG user-space coordinate into (robot_x, robot_y, robot_z).
        The SVG coordinate space is fitted to the drawable path bounds during
        load so the painted extents use the calibrated canvas area.

        Here we flip the vertical axis so that svg_y=0 corresponds
        to z_max (top) and svg_y=svg_height corresponds to z_min (bottom).
        """
        nx = (svg_x - self.svg_min_x) / self.svg_width   # 0..1 (left to right)
        ny = (svg_y - self.svg_min_y) / self.svg_height  # 0..1 (top to bottom)
        region = getattr(self, "_active_svg_canvas_region", (0.0, 1.0, 0.0, 1.0))
        region_x0, region_x1, region_y0, region_y1 = [float(value) for value in region]
        nx = region_x0 + nx * (region_x1 - region_x0)
        ny = region_y0 + ny * (region_y1 - region_y0)
        if self._is_table_surface():
            robot_x, robot_y, robot_z = self._canvas_point_from_uv(nx, ny)
            return (robot_x, robot_y, robot_z)

        y_min, y_max, z_min, z_max = self._svg_canvas_bounds()

         # # FLIP the vertical axis for z:
         # robot_z = self.z_max - ny * (self.z_max - self.z_min)
         # robot_y = self.y_min + nx * (self.y_max - self.y_min)

         # flipped
         # robot_z = self.z_min + ny * (self.z_max - self.z_min)
        robot_z = z_max - ny * (z_max - z_min)

        robot_y = y_max - nx * (y_max - y_min)

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
            self._set_position_keep_wrist(*p_start_rb, speed=speed/1.5, wait=True)
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
        self._set_position_keep_wrist(*p_start_rb, speed=speed/1.5, wait=True)
        cur_x, cur_y, cur_z = self._lift_off_plane(*p_start_rb, speed=speed)

        # remaining dots
        for i in range(1, n_dot):
            t  = i / (n_dot - 1)
            nx = p_start_rb[0] + t*dx
            ny = p_start_rb[1] + t*dy
            nz = p_start_rb[2] + t*dz

            self._set_position_keep_wrist(x=nx - self.STIPPLE_LIFT_MM, y=ny, z=nz,
                                          speed=speed, wait=True)
            self._set_position_keep_wrist(x=nx, y=ny, z=nz,
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
brush_selector = None

ARMTRACK_SPEED_MIN = 10.0
ARMTRACK_SPEED_MAX = 300.0
ARMTRACK_OPEN_GRIPPER_POS = 850
ARMTRACK_CLOSE_GRIPPER_POS = 0


def _armtrack_as_int_code(result):
    if isinstance(result, (tuple, list)) and result:
        result = result[0]
    try:
        return int(result)
    except (TypeError, ValueError):
        return -1


def _armtrack_ack(command):
    print(f"OSC send: /armtrack/ack {[str(command), 0]}", flush=True)
    controller.osc_client.send_message("/armtrack/ack", [str(command), 0])


def _armtrack_error(command, error):
    print(f"OSC send: /armtrack/error {[str(command), str(error)]}", flush=True)
    controller.osc_client.send_message("/armtrack/error", [str(command), str(error)])


def _armtrack_clamp_speed(raw_speed):
    speed = float(raw_speed)
    return max(ARMTRACK_SPEED_MIN, min(ARMTRACK_SPEED_MAX, speed))


def _armtrack_wait_flag(raw_wait):
    if isinstance(raw_wait, str):
        return raw_wait.strip().lower() in {"1", "true", "yes", "on"}
    return bool(int(raw_wait))


def _armtrack_open_gripper(arm):
    if hasattr(arm, 'open_lite6_gripper'):
        return _armtrack_as_int_code(arm.open_lite6_gripper())
    if hasattr(arm, 'set_gripper_position'):
        return _armtrack_as_int_code(
            arm.set_gripper_position(ARMTRACK_OPEN_GRIPPER_POS, wait=True, auto_enable=True)
        )
    if hasattr(arm, 'robotiq_open'):
        return _armtrack_as_int_code(arm.robotiq_open(wait=True))
    raise RuntimeError('No supported gripper-open API found on this arm profile.')


def _armtrack_close_gripper(arm):
    if hasattr(arm, 'close_lite6_gripper'):
        return _armtrack_as_int_code(arm.close_lite6_gripper())
    if hasattr(arm, 'set_gripper_position'):
        return _armtrack_as_int_code(
            arm.set_gripper_position(ARMTRACK_CLOSE_GRIPPER_POS, wait=True, auto_enable=True)
        )
    if hasattr(arm, 'robotiq_close'):
        return _armtrack_as_int_code(arm.robotiq_close(wait=True))
    raise RuntimeError('No supported gripper-close API found on this arm profile.')


def _handle_armtrack_command(message):
    if len(message) < 2:
        _armtrack_error('armtrack', 'missing_sub_command')
        return

    sub_cmd = str(message[1])

    if sub_cmd in {'move_to_pose', 'move_linear'}:
        if not controller.allowed_to_play:
            _armtrack_error(sub_cmd, 'panic_mode_enabled')
            return
        if len(message) < 10:
            _armtrack_error(sub_cmd, 'expected 8 payload fields: x y z roll pitch yaw speed wait')
            return

        try:
            x = float(message[2])
            y = float(message[3])
            z = float(message[4])
            roll = float(message[5])
            pitch = float(message[6])
            yaw = float(message[7])
            speed = _armtrack_clamp_speed(message[8])
            wait = _armtrack_wait_flag(message[9])
        except (TypeError, ValueError) as exc:
            _armtrack_error(sub_cmd, f'invalid_payload:{exc}')
            return

        try:
            controller.setup()
            code = _armtrack_as_int_code(
                controller.arm.set_position(
                    x=x,
                    y=y,
                    z=z,
                    roll=roll,
                    pitch=pitch,
                    yaw=yaw,
                    speed=speed,
                    wait=wait,
                )
            )
        except Exception as exc:
            _armtrack_error(sub_cmd, f'exception:{exc}')
            return

        if code == 0:
            _armtrack_ack(sub_cmd)
        else:
            _armtrack_error(sub_cmd, code)
        return

    if sub_cmd == 'open_gripper':
        if not controller.allowed_to_play:
            _armtrack_error(sub_cmd, 'panic_mode_enabled')
            return
        try:
            controller.setup()
            code = _armtrack_open_gripper(controller.arm)
        except Exception as exc:
            _armtrack_error(sub_cmd, f'exception:{exc}')
            return
        if code == 0:
            _armtrack_ack(sub_cmd)
        else:
            _armtrack_error(sub_cmd, code)
        return

    if sub_cmd == 'close_gripper':
        if not controller.allowed_to_play:
            _armtrack_error(sub_cmd, 'panic_mode_enabled')
            return
        try:
            controller.setup()
            code = _armtrack_close_gripper(controller.arm)
        except Exception as exc:
            _armtrack_error(sub_cmd, f'exception:{exc}')
            return
        if code == 0:
            _armtrack_ack(sub_cmd)
        else:
            _armtrack_error(sub_cmd, code)
        return

    if sub_cmd == 'get_position':
        try:
            controller.setup()
            result = controller.arm.get_position()
        except Exception as exc:
            _armtrack_error(sub_cmd, f'exception:{exc}')
            return

        if not isinstance(result, (tuple, list)) or len(result) < 2:
            _armtrack_error(sub_cmd, 'invalid_get_position_response')
            return

        code = _armtrack_as_int_code(result[0])
        position = result[1]
        if code != 0:
            _armtrack_error(sub_cmd, code)
            return
        if not isinstance(position, (tuple, list)) or len(position) < 6:
            _armtrack_error(sub_cmd, 'position_payload_too_short')
            return

        payload = [float(position[i]) for i in range(6)]
        controller.osc_client.send_message('/armtrack/current_pose', payload)
        _armtrack_ack(sub_cmd)
        return

    _armtrack_error(sub_cmd, 'unknown_sub_command')


def _optional_float_arg(message, index, default):
    if len(message) > index:
        return float(message[index])
    return float(default)


def _consume_paint_z_brush_arg(message, index):
    if len(message) <= index:
        return None, index
    raw = message[index]
    token = str(raw).strip().lower()
    if token in {"brush", "b"}:
        if len(message) <= index + 1:
            raise ValueError("missing brush number")
        return controller._normalize_paint_z_brush(message[index + 1]), index + 2
    if token in {"current", "active"}:
        return token, index + 1
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None, index
    if value.is_integer():
        brush = int(value)
        if 1 <= brush <= controller.paint_z_brush_count:
            return brush, index + 1
    return None, index


def _paint_z_step_arg(message, index):
    return abs(_optional_float_arg(message, index, controller.paint_z_step_mm))


def _handle_paint_z_action(sub_cmd, message, arg_index, brush=None):
    if sub_cmd in {'up', 'z_up', 'z-up', 'zup'}:
        parsed_brush, next_index = _consume_paint_z_brush_arg(message, arg_index)
        if parsed_brush is not None:
            brush = parsed_brush
            arg_index = next_index
        step = _paint_z_step_arg(message, arg_index)
        controller.adjust_paint_z_offset(step, label='z_up', brush=brush)
    elif sub_cmd in {'down', 'z_down', 'z-down', 'zdown'}:
        parsed_brush, next_index = _consume_paint_z_brush_arg(message, arg_index)
        if parsed_brush is not None:
            brush = parsed_brush
            arg_index = next_index
        step = _paint_z_step_arg(message, arg_index)
        controller.adjust_paint_z_offset(-step, label='z_down', brush=brush)
    elif sub_cmd in {'reset', 'zero', 'clear'}:
        parsed_brush, next_index = _consume_paint_z_brush_arg(message, arg_index)
        if parsed_brush is not None:
            brush = parsed_brush
        elif len(message) > arg_index and str(message[arg_index]).strip().lower() == "all":
            brush = "all"
        controller.reset_paint_z_offsets(brush=brush)
    elif sub_cmd in {'set', 'offset'}:
        if brush is None and len(message) - arg_index >= 2:
            parsed_brush, next_index = _consume_paint_z_brush_arg(message, arg_index)
            if parsed_brush is not None:
                brush = parsed_brush
                arg_index = next_index
        controller.set_paint_z_offset(float(message[arg_index]), label='set', brush=brush)
    elif sub_cmd in {'status', 'get'}:
        parsed_brush, _next_index = _consume_paint_z_brush_arg(message, arg_index)
        if parsed_brush is not None:
            brush = parsed_brush
        controller.report_paint_z_offset(label='status', brush=brush)
    else:
        raise ValueError(f"unknown paint Z sub-command: {sub_cmd}")


def _handle_paint_z_command(message):
    cmd = str(message[0]).strip().lower()
    try:
        if cmd in {'z_up', 'z-up', 'zup'}:
            brush, next_index = _consume_paint_z_brush_arg(message, 1)
            step = _paint_z_step_arg(message, next_index)
            controller.adjust_paint_z_offset(step, label='z_up', brush=brush)
            return

        if cmd in {'z_down', 'z-down', 'zdown'}:
            brush, next_index = _consume_paint_z_brush_arg(message, 1)
            step = _paint_z_step_arg(message, next_index)
            controller.adjust_paint_z_offset(-step, label='z_down', brush=brush)
            return

        if len(message) < 2:
            controller.report_paint_z_offset(label='status')
            return

        brush, next_index = _consume_paint_z_brush_arg(message, 1)
        if brush is not None:
            if len(message) <= next_index:
                controller.report_paint_z_offset(label='status', brush=brush)
                return
            sub_cmd = str(message[next_index]).strip().lower()
            _handle_paint_z_action(sub_cmd, message, next_index + 1, brush=brush)
        else:
            sub_cmd = str(message[1]).strip().lower()
            _handle_paint_z_action(sub_cmd, message, 2, brush=None)
    except (IndexError, TypeError, ValueError) as exc:
        print(f"Paint Z command failed: {message} error={exc}", flush=True)
        controller.osc_client.send_message("/paint/z_error", [cmd, str(exc)])


def _join_message_args(message, start_index):
    if len(message) <= start_index:
        raise ValueError("missing argument")
    return " ".join(str(part) for part in message[start_index:])


def _is_live_while_busy_command(message):
    if not message:
        return False
    cmd = str(message[0]).strip().lower()
    if cmd == "panic" or cmd in PAINT_Z_COMMANDS:
        return True
    if cmd == "bci" and len(message) > 1:
        sub_cmd = str(message[1]).strip().lower()
        return sub_cmd in BCI_LIVE_SUBCOMMANDS
    return False


def _handle_bci_command(message):
    global brush_selector
    if len(message) < 2:
        print("OSC BCI error: expected sub-command", flush=True)
        controller.osc_client.send_message("/bci/error", ["payload", "missing_sub_command"])
        return

    sub_cmd = str(message[1]).strip().lower()
    try:
        if sub_cmd == "load":
            slot = message[2]
            svg_file = _join_message_args(message, 3)
            summary = controller.load_bci_svg(slot, svg_file)
            print(f"OSC BCI done: load -> {summary}", flush=True)

        elif sub_cmd == "load_layer":
            if len(message) < 5:
                raise ValueError("usage: bci load_layer SLOT SVG_FILE LAYER_NAME")
            slot = message[2]
            layer_name = str(message[-1])
            svg_file = " ".join(str(part) for part in message[3:-1])
            summary = controller.load_bci_svg_layer(slot, svg_file, layer_name)
            preview_path = controller.write_bci_layer_preview()
            print(
                f"OSC BCI done: load_layer -> {summary}; preview={preview_path}",
                flush=True,
            )
            controller.osc_client.send_message("/bci/preview", [str(preview_path)])

        elif sub_cmd == "load_layers":
            if len(message) < 3:
                raise ValueError("usage: bci load_layers SVG_FILE [LAYER1 LAYER2 LAYER3]")
            if len(message) == 3:
                svg_file = str(message[2])
                layer_names = ["Layer_3", "Layer_5", "Layer_6"]
            elif len(message) >= 6:
                svg_file = " ".join(str(part) for part in message[2:-3])
                layer_names = [str(part) for part in message[-3:]]
            else:
                raise ValueError("load_layers needs either one SVG file or SVG file plus three layer names")
            result = controller.load_bci_svg_layers(svg_file, layer_names)
            print(f"OSC BCI done: load_layers -> {result}", flush=True)
            controller.osc_client.send_message("/bci/preview", [result["preview_path"]])
            controller.send_bci_status()

        elif sub_cmd in {"preview", "preview_layers"}:
            preview_path = controller.write_bci_layer_preview()
            print(f"OSC BCI done: preview -> {preview_path}", flush=True)
            controller.osc_client.send_message("/bci/preview", [str(preview_path)])

        elif sub_cmd in {"attention", "focus", "state"}:
            attention = controller.set_bci_attention(message[2])
            print(f"OSC BCI done: attention -> {attention}", flush=True)
            if attention.get("changed"):
                controller.draw_bci_stroke_for_event(
                    f"attention_change_{attention['previous']}_to_{attention['current']}"
                )
            controller.send_bci_status()

        elif sub_cmd in {"emg", "movement"}:
            value = message[2] if len(message) > 2 else 1
            pending = controller.trigger_bci_emg(value)
            print(f"OSC BCI done: emg pending -> {pending}", flush=True)

        elif sub_cmd in {"next", "paint", "stroke"}:
            speed = float(message[2]) if len(message) > 2 else None
            result = controller.draw_bci_next_stroke(speed=speed)
            if result.get("drawn"):
                controller.apply_bci_pending_color_change(brush_selector)
            controller.send_bci_status()

        elif sub_cmd == "run":
            count = int(message[2]) if len(message) > 2 else 1
            count = max(1, min(count, 100))
            speed = float(message[3]) if len(message) > 3 else None
            controller.bci_stop_requested = False
            drawn_count = 0
            for _ in range(count):
                if controller.bci_stop_requested or not controller.allowed_to_play:
                    break
                result = controller.draw_bci_next_stroke(speed=speed)
                if not result.get("drawn"):
                    break
                drawn_count += 1
                controller.apply_bci_pending_color_change(brush_selector)
            print(f"OSC BCI done: run drew {drawn_count} stroke(s)", flush=True)
            controller.osc_client.send_message("/bci/run_done", [drawn_count, count])
            controller.send_bci_status()

        elif sub_cmd == "stop":
            controller.bci_stop_requested = True
            print("OSC BCI stop requested; current stroke will finish first.", flush=True)
            controller.osc_client.send_message("/bci/stop", [1])

        elif sub_cmd == "status":
            controller.send_bci_status()

        elif sub_cmd in {"reset", "clear"}:
            controller.reset_bci_paint()
            controller.send_bci_status()

        elif sub_cmd in {"goto", "counter", "set_counter"}:
            summary = controller.set_bci_svg_counter(message[2], int(message[3]))
            print(f"OSC BCI done: counter -> {summary}", flush=True)

        elif sub_cmd in {"cooldown", "color_cooldown", "colour_cooldown"}:
            controller.bci_color_cooldown_strokes = max(0, int(message[2]))
            controller.save_bci_paint_state()
            print(
                f"BCI color cooldown set to {controller.bci_color_cooldown_strokes} stroke(s)",
                flush=True,
            )
            controller.send_bci_status()

        elif sub_cmd in {"current_brush", "current_color", "current_colour"}:
            controller.bci_current_color_brush = max(0, int(message[2]))
            controller.save_bci_paint_state()
            print(
                f"BCI current color brush set to {controller.bci_current_color_brush}",
                flush=True,
            )
            controller.send_bci_status()

        else:
            print(f"OSC BCI error: unknown sub-command {sub_cmd}", flush=True)
            controller.osc_client.send_message("/bci/error", [sub_cmd, "unknown_sub_command"])

    except Exception as exc:
        print(f"OSC BCI command failed: args={message} error={exc}", flush=True)
        controller.osc_client.send_message("/bci/error", [sub_cmd, str(exc)])


def _draw_bci_stroke_if_brush_changed(before_state, after_state, reason):
    before_holding = bool(before_state.get("HOLDING_BRUSH", False))
    after_holding = bool(after_state.get("HOLDING_BRUSH", False))
    before_brush = int(before_state.get("CURRENT_BRUSH", 0))
    after_brush = int(after_state.get("CURRENT_BRUSH", 0))
    if not after_holding or after_brush <= 0:
        return None
    if before_holding and before_brush == after_brush:
        return None
    return controller.draw_bci_stroke_for_event(f"{reason}_brush_{after_brush}")


def interactive(address, *args):
    global controller
    if not controller:
        print("Error: Controller not initialized.")
        return

    max_msg = list(args)
    cmd = str(max_msg[0]).strip().lower() if max_msg else ""
    live_command = _is_live_while_busy_command(max_msg)

    if not controller.command_lock.acquire(blocking=False):
        if live_command:
            print(f"OSC recv during busy, allowing live command: address={address} args={max_msg}", flush=True)
            _interactive_impl(address, *args)
            return

        print(f"OSC busy: ignoring command while another command is running: {max_msg}", flush=True)
        controller.osc_client.send_message("/busy", [1, cmd])
        controller.osc_client.send_message("/listen", 0)
        return

    controller.busy = True
    try:
        if not live_command:
            controller.osc_client.send_message("/listen", 0)
            controller.command_video_recorder.start(max_msg)
        _interactive_impl(address, *args)
    finally:
        if not live_command:
            controller.command_video_recorder.stop()
        controller.busy = False
        controller.command_lock.release()
        if controller.allowed_to_play:
            controller.osc_client.send_message("/listen", 1)
        else:
            controller.osc_client.send_message("/listen", 0)


def macbook_path_interactive(address, *args):
    path_parts = [
        part.strip()
        for part in str(address).split("/")
        if part.strip()
    ]
    if path_parts and path_parts[0].lower() == "macbook":
        path_parts = path_parts[1:]
    if not path_parts:
        interactive(address, *args)
        return

    converted_args = path_parts + list(args)
    print(
        f"OSC recv path-style: address={address} converted_args={converted_args}",
        flush=True,
    )
    interactive(address, *converted_args)


def unhandled_osc(address, *args):
    print(f"OSC unhandled: address={address} args={list(args)}", flush=True)


def _muse_paint_next_stroke(_value):
    start_attention = int(controller.bci_attention_state)
    attempted = []
    result = None
    for offset in range(3):
        slot = (start_attention + offset) % 3
        slot_data = controller.bci_slots[slot]
        attempted.append(slot + 1)
        if int(slot_data.get("stroke_index", 0)) >= len(slot_data.get("strokes", [])):
            continue
        if slot != int(controller.bci_attention_state):
            print(
                f"Muse OSC head_down_paint: selected SVG {start_attention + 1} is finished; "
                f"switching to SVG {slot + 1}",
                flush=True,
            )
            controller.set_bci_attention(slot)
        result = controller.draw_bci_next_stroke()
        break

    if result is None:
        print(
            f"Muse OSC head_down_paint: no unfinished BCI SVG slots found; attempted={attempted}",
            flush=True,
        )
        result = {
            "drawn": False,
            "reason": "all_slots_finished",
            "slot": start_attention,
            "svg_number": start_attention + 1,
            "stroke_index": 0,
            "total_strokes": 0,
        }

    if result.get("drawn"):
        color_state = controller.apply_bci_pending_color_change(brush_selector)
        if color_state is not None:
            global muse_strokes_since_brush_change
            muse_strokes_since_brush_change = 0
    controller.send_bci_status()
    return result


def _muse_brush_change_allowed(action_name, *, log=False):
    if muse_strokes_since_brush_change >= MUSE_BRUSH_CHANGE_STROKE_LOCKOUT:
        return True
    remaining = MUSE_BRUSH_CHANGE_STROKE_LOCKOUT - muse_strokes_since_brush_change
    if log:
        print(
            f"Muse OSC {action_name} ignored until {remaining} more stroke command(s) run",
            flush=True,
        )
        controller.osc_client.send_message(
            "/brush/ignored",
            [action_name, "stroke_lockout", int(remaining)],
        )
    return False


def _muse_next_color(_value):
    global muse_strokes_since_brush_change
    if brush_selector is None:
        raise RuntimeError("BrushSelector is not initialized")
    controller.trigger_bci_emg(_value)
    state = controller.apply_bci_pending_color_change(brush_selector)
    if state is not None:
        muse_strokes_since_brush_change = 0
        controller.osc_client.send_message("/brush/done", ["muse_next_color", state["CURRENT_BRUSH"]])
        return state
    return {
        "changed": False,
        "pending": bool(controller.bci_pending_emg),
        "strokes_since_color_change": int(controller.bci_strokes_since_color_change),
        "required_strokes": int(controller.bci_color_cooldown_strokes),
    }


def _handle_muse_action(action_name, value, action_func, *, use_threshold=True, use_cooldown=True):
    global last_muse_action_times
    if use_threshold and value < MUSE_HEAD_GESTURE_THRESHOLD:
        return

    now = time.monotonic()
    previous = last_muse_action_times.get(action_name, 0.0)
    if use_cooldown and now - previous < MUSE_HEAD_GESTURE_COOLDOWN_SEC:
        print(
            f"Muse OSC {action_name} ignored during cooldown: value={value:.3f}",
            flush=True,
        )
        return

    if not controller.command_lock.acquire(blocking=False):
        print(
            f"Muse OSC {action_name} ignored while robot is busy: value={value:.3f}",
            flush=True,
        )
        return

    if use_cooldown:
        last_muse_action_times[action_name] = now
    controller.busy = True
    try:
        print(f"Muse OSC {action_name} running: value={value:.3f}", flush=True)
        controller.osc_client.send_message("/listen", 0)
        result = action_func(value)
        print(f"Muse OSC {action_name} done: {result}", flush=True)
    except Exception as exc:
        print(f"Muse OSC {action_name} failed: {exc}", flush=True)
        controller.osc_client.send_message("/bci/error", [action_name, str(exc)])
    finally:
        controller.busy = False
        controller.command_lock.release()
        controller.osc_client.send_message("/listen", 1 if controller.allowed_to_play else 0)


def muse_osc_handler(address, *args):
    global controller
    global brush_selector
    values = list(args)
    print(f"Muse OSC recv: address={address} args={values}", flush=True)
    if not controller or not values:
        return

    normalized_address = str(address).strip().lower()
    attention_source = normalized_address in {
        "/muse/elements/experimental/concentration",
        "/muse/concentration",
        "/muse/classifier/focused",
    }
    relaxed_source = normalized_address in {
        "/muse/elements/experimental/mellow",
        "/muse/mellow",
        "/muse/classifier/relaxed",
    }
    muse_actions = {
        "/muse/classifier/head/up": ("head_up_paint", _muse_paint_next_stroke, True, False),
        "/muse/elements/head/up": ("head_up_paint", _muse_paint_next_stroke, True, False),
        "/muse/head/up": ("head_up_paint", _muse_paint_next_stroke, True, False),
        "/muse/classifier/head/down": ("head_down_paint", _muse_paint_next_stroke, True, False),
        "/muse/elements/head/down": ("head_down_paint", _muse_paint_next_stroke, True, False),
        "/muse/head/down": ("head_down_paint", _muse_paint_next_stroke, True, False),
        "/muse/classifier/head/left": ("head_left_paint", _muse_paint_next_stroke, True, False),
        "/muse/elements/head/left": ("head_left_paint", _muse_paint_next_stroke, True, False),
        "/muse/head/left": ("head_left_paint", _muse_paint_next_stroke, True, False),
        "/muse/classifier/head/right": ("head_right_paint", _muse_paint_next_stroke, True, False),
        "/muse/elements/head/right": ("head_right_paint", _muse_paint_next_stroke, True, False),
        "/muse/head/right": ("head_right_paint", _muse_paint_next_stroke, True, False),
        "/muse/classifier/jaw_clench": ("jaw_clench_next_color", _muse_next_color, True, True),
        "/muse/elements/jaw_clench": ("jaw_clench_next_color", _muse_next_color, True, True),
        "/muse/jaw_clench": ("jaw_clench_next_color", _muse_next_color, True, True),
        "/muse/classifier/blink": ("blink_next_color", _muse_next_color, True, True),
        "/muse/elements/blink": ("blink_next_color", _muse_next_color, True, True),
        "/muse/blink": ("blink_next_color", _muse_next_color, True, True),
    }
    if normalized_address in muse_actions:
        try:
            value = max(0.0, min(1.0, float(values[0])))
        except (TypeError, ValueError):
            print(f"Muse OSC gesture ignored; non-numeric value: {values[0]}", flush=True)
            return
        action_name, action_func, use_threshold, use_cooldown = muse_actions[normalized_address]
        _handle_muse_action(
            action_name,
            value,
            action_func,
            use_threshold=use_threshold,
            use_cooldown=use_cooldown,
        )
        return

    if not attention_source and not relaxed_source:
        return

    try:
        value = max(0.0, min(1.0, float(values[0])))
    except (TypeError, ValueError):
        print(f"Muse OSC attention ignored; non-numeric value: {values[0]}", flush=True)
        return

    attention_value = 1.0 - value if relaxed_source else value
    if attention_value < MUSE_ATTENTION_LOW:
        attention = 0
    elif attention_value < MUSE_ATTENTION_HIGH:
        attention = 1
    else:
        attention = 2

    if attention == int(controller.bci_attention_state):
        return

    controller.set_bci_attention(attention)
    controller.send_bci_status()
    print(
        f"Muse OSC mapped {normalized_address} value={value:.3f} to BCI attention {attention}",
        flush=True,
    )


def _interactive_impl(address, *args):
    """
    Example OSC handler. We’ll add a “svg” sub-command for loading and stepping
    through the SVG, stroke by stroke.
    """
    global controller # Ensure we're using the global controller initialized in main
    global brush_selector
    if not controller:
        print("Error: Controller not initialized.")
        return

    max_msg = list(args)
    cmd = str(max_msg[0]).strip().lower() if max_msg else ""

    print(f"OSC recv: address={address} args={max_msg}", flush=True)

    if cmd in PAINT_Z_COMMANDS:
        _handle_paint_z_command(max_msg)
        return

    if cmd == 'homing':
        # example code
        if max_msg[1] in {'draw_border', 'border'}:
            controller.draw_canvas_border()
        elif max_msg[1] == 'capture':
            if len(max_msg) < 3:
                print("Homing capture error: expected corner name", flush=True)
                controller.osc_client.send_message("/homing/error", ["capture", "missing_corner"])
                return

            corner_type = str(max_msg[2])
            corner_index = CANVAS_CORNER_INDEX.get(corner_type)
            if corner_index is None:
                print(f"Homing capture error: unknown corner type {corner_type}", flush=True)
                controller.osc_client.send_message("/homing/error", [corner_type, "unknown_corner"])
                return

            controller.setup()
            position_result = controller.arm.get_position()
            if (
                not isinstance(position_result, (list, tuple))
                or len(position_result) < 2
                or int(position_result[0]) != 0
                or len(position_result[1]) < 3
            ):
                print(
                    f"Homing capture failed for {corner_type}: could not read current "
                    f"arm position: {position_result}",
                    flush=True,
                )
                controller.osc_client.send_message(
                    "/homing/error",
                    [corner_type, "get_position_failed"],
                )
                return

            current_pose = position_result[1]
            x_val, y_val, z_val = [float(value) for value in current_pose[:3]]
            controller.canvas_corners[corner_index] = [x_val, y_val, z_val]
            controller.calculate_scaling_factors()
            controller.calculate_canvas_bounds()
            controller.save_canvas_corners()
            print(
                f"Canvas corners: captured current robot XYZ as {corner_type} "
                f"using brush {controller.canvas_reference_brush} as reference "
                f"and persistently saved: {[x_val, y_val, z_val]}",
                flush=True,
            )
            print(
                f"Canvas corners: active calibration is now {controller.canvas_corners}; "
                "brush-specific paint Z offsets are relative to this calibration.",
                flush=True,
            )
            controller.osc_client.send_message("/homing/done", [corner_type, x_val, y_val, z_val])
        elif max_msg[1] == 'move':
            if len(max_msg) < 3:
                print("Homing move error: expected corner name", flush=True)
                controller.osc_client.send_message("/homing/error", ["move", "missing_corner"])
                return

            corner_type = str(max_msg[2])
            corner_index = CANVAS_CORNER_INDEX.get(corner_type)
            if corner_index is None:
                print(f"Homing move error: unknown corner type {corner_type}", flush=True)
                controller.osc_client.send_message("/homing/error", [corner_type, "unknown_corner"])
                return

            target = [float(value) for value in controller.canvas_corners[corner_index]]
            controller.setup()
            position_result = controller.arm.get_position()
            if (
                not isinstance(position_result, (list, tuple))
                or len(position_result) < 2
                or int(position_result[0]) != 0
                or len(position_result[1]) < 6
            ):
                print(
                    f"Homing move failed for {corner_type}: could not read current "
                    f"arm orientation: {position_result}",
                    flush=True,
                )
                controller.osc_client.send_message(
                    "/homing/error",
                    [corner_type, "get_position_failed"],
                )
                return

            current_pose = position_result[1]
            current_roll, current_pitch, current_yaw = current_pose[3:6]
            print(
                f"Homing move: moving to saved {corner_type} XYZ={target} "
                "while preserving current wrist orientation",
                flush=True,
            )
            move_result = controller.arm.set_position(
                x=target[0], y=target[1], z=target[2],
                roll=current_roll, pitch=current_pitch, yaw=current_yaw,
                speed=100, wait=True
            )
            move_code = _armtrack_as_int_code(move_result)
            if move_code != 0:
                print(
                    f"Homing move failed for {corner_type}; set_position code={move_code}",
                    flush=True,
                )
                controller.osc_client.send_message("/homing/error", [corner_type, move_code])
                return

            print(
                f"Homing move done: reached saved {corner_type} XYZ={target}",
                flush=True,
            )
            controller.osc_client.send_message(
                "/homing/done",
                ["move", corner_type, target[0], target[1], target[2]],
            )
        else:
            # set corners
            corner_type = max_msg[1]  # top_left, top_right, etc.
            x_val = float(max_msg[2])
            y_val = float(max_msg[3])
            z_val = float(max_msg[4])
            previous_canvas_corners = [corner[:] for corner in controller.canvas_corners]
            if corner_type == 'top_left':
                controller.canvas_corners[0] = [x_val, y_val, z_val]
            elif corner_type == 'top_right':
                controller.canvas_corners[1] = [x_val, y_val, z_val]
            elif corner_type == 'bottom_left':
                controller.canvas_corners[2] = [x_val, y_val, z_val]
            elif corner_type == 'bottom_right':
                controller.canvas_corners[3] = [x_val, y_val, z_val]
            else:
                print(f"Homing error: unknown corner type {corner_type}", flush=True)
                controller.osc_client.send_message("/homing/error", [str(corner_type), "unknown_corner"])
                return

            move_requested = len(max_msg) > 5 and int(max_msg[5]) == 1
            if move_requested:
                controller.setup()
                position_result = controller.arm.get_position()
                if (
                    not isinstance(position_result, (list, tuple))
                    or len(position_result) < 2
                    or int(position_result[0]) != 0
                    or len(position_result[1]) < 6
                ):
                    print(
                        f"Homing move skipped: could not read current arm orientation: "
                        f"{position_result}",
                        flush=True,
                    )
                    controller.canvas_corners = previous_canvas_corners
                    controller.osc_client.send_message(
                        "/homing/error",
                        [str(corner_type), "get_position_failed"],
                    )
                    return
                current_pose = position_result[1]
                current_roll, current_pitch, current_yaw = current_pose[3:6]
                move_result = controller.arm.set_position(
                    x=x_val, y=y_val, z=z_val,
                    roll=current_roll, pitch=current_pitch, yaw=current_yaw,
                    speed=100, wait=True
                )
                move_code = _armtrack_as_int_code(move_result)
                if move_code != 0:
                    controller.canvas_corners = previous_canvas_corners
                    controller.calculate_scaling_factors()
                    controller.calculate_canvas_bounds()
                    print(
                        f"Homing move failed for {corner_type}; set_position code={move_code}. "
                        "Canvas calibration was not saved.",
                        flush=True,
                    )
                    controller.osc_client.send_message(
                        "/homing/error",
                        [str(corner_type), move_code],
                    )
                    return
            controller.calculate_scaling_factors()
            controller.calculate_canvas_bounds()
            controller.save_canvas_corners()
            print(
                f"Canvas corners: active calibration updated and persistently saved "
                f"after {corner_type}: {controller.canvas_corners}",
                flush=True,
            )
            controller.osc_client.send_message("/homing/done", [str(corner_type), x_val, y_val, z_val])

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

    elif cmd == 'bci':
        _handle_bci_command(max_msg)

    # ----------------
    # NEW: SVG COMMAND
    # ----------------
    elif cmd == 'svg':
        sub_cmd = max_msg[1]
        if sub_cmd == 'load':
            # E.g. /macbook svg load my_drawing.svg
            svg_file = max_msg[2]
            controller.load_svg_strokes(svg_file)
        elif sub_cmd in {'reference', 'set_reference'}:
            svg_file = max_msg[2]
            if controller.set_svg_reference(svg_file):
                controller.set_svg_scale_mode('reference')
                controller.osc_client.send_message("/svg/reference", [
                    svg_file,
                    controller.svg_reference_space["min_x"],
                    controller.svg_reference_space["min_y"],
                    controller.svg_reference_space["width"],
                    controller.svg_reference_space["height"],
                ])
            else:
                controller.osc_client.send_message("/svg/error", ["reference", svg_file])
        elif sub_cmd in {'clear_reference', 'reference_clear'}:
            controller.clear_svg_reference()
            controller.osc_client.send_message("/svg/reference", ["cleared"])
        elif sub_cmd == 'reference_status':
            if controller.svg_reference_space is None:
                controller.osc_client.send_message("/svg/reference", ["none"])
            else:
                ref = controller.svg_reference_space
                controller.osc_client.send_message("/svg/reference", [
                    ref["source"], ref["min_x"], ref["min_y"], ref["width"], ref["height"]
                ])
        elif sub_cmd in {'scale_mode', 'mode'}:
            try:
                controller.set_svg_scale_mode(max_msg[2])
                controller.osc_client.send_message("/svg/scale_mode", [controller.svg_scale_mode])
            except (IndexError, ValueError) as exc:
                print(f"SVG scale mode error: {exc}", flush=True)
                controller.osc_client.send_message("/svg/error", ["scale_mode", str(exc)])
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
        if len(max_msg) < 2:
            print("OSC brush error: expected sub-command", flush=True)
            return

        sub_cmd = str(max_msg[1])
        try:
            print(f"OSC brush command: {sub_cmd} payload={max_msg[2:]}", flush=True)
            if sub_cmd in {'pickup', 'pick_up', 'pick'}:
                print("OSC send: /listen 0", flush=True)
                controller.osc_client.send_message("/listen", 0)
                before_state = brush_selector.status()
                state = brush_selector.pickup(int(max_msg[2]))
                _draw_bci_stroke_if_brush_changed(before_state, state, "pickup")
                print(f"OSC brush done: pickup -> {state}", flush=True)
                print(f"OSC send: /brush/done {['pickup', state['CURRENT_BRUSH']]}", flush=True)
                controller.osc_client.send_message("/brush/done", ["pickup", state["CURRENT_BRUSH"]])
                print("OSC send: /listen 1", flush=True)
                controller.osc_client.send_message("/listen", 1)
            elif sub_cmd in {'putback', 'put_back', 'drop'}:
                print("OSC send: /listen 0", flush=True)
                controller.osc_client.send_message("/listen", 0)
                holder = int(max_msg[2]) if len(max_msg) > 2 else None
                state = brush_selector.putback(holder)
                print(f"OSC brush done: putback -> {state}", flush=True)
                print(f"OSC send: /brush/done {['putback', state['CURRENT_BRUSH']]}", flush=True)
                controller.osc_client.send_message("/brush/done", ["putback", state["CURRENT_BRUSH"]])
                print("OSC send: /listen 1", flush=True)
                controller.osc_client.send_message("/listen", 1)
            elif sub_cmd == 'swap':
                print("OSC send: /listen 0", flush=True)
                controller.osc_client.send_message("/listen", 0)
                before_state = brush_selector.status()
                state = brush_selector.swap(int(max_msg[2]))
                _draw_bci_stroke_if_brush_changed(before_state, state, "swap")
                print(f"OSC brush done: swap -> {state}", flush=True)
                print(f"OSC send: /brush/done {['swap', state['CURRENT_BRUSH']]}", flush=True)
                controller.osc_client.send_message("/brush/done", ["swap", state["CURRENT_BRUSH"]])
                print("OSC send: /listen 1", flush=True)
                controller.osc_client.send_message("/listen", 1)
            elif sub_cmd in {'open', 'release'}:
                print("OSC send: /listen 0", flush=True)
                controller.osc_client.send_message("/listen", 0)
                holder = int(max_msg[2]) if len(max_msg) > 2 else None
                state = brush_selector.open_hand(holder)
                print(f"OSC brush done: open -> {state}", flush=True)
                print(f"OSC send: /brush/done {['open', state['CURRENT_BRUSH']]}", flush=True)
                controller.osc_client.send_message("/brush/done", ["open", state["CURRENT_BRUSH"]])
                print("OSC send: /listen 1", flush=True)
                controller.osc_client.send_message("/listen", 1)
            elif sub_cmd in {'open_ready', 'ready_open', 'recover_open'}:
                print("OSC send: /listen 0", flush=True)
                controller.osc_client.send_message("/listen", 0)
                holder = int(max_msg[2]) if len(max_msg) > 2 else None
                state = brush_selector.open_ready(holder)
                print(f"OSC brush done: open_ready -> {state}", flush=True)
                print(
                    f"OSC send: /brush/done {['open_ready', state['PREVIEW_BRUSH']]}",
                    flush=True,
                )
                controller.osc_client.send_message(
                    "/brush/done",
                    ["open_ready", state["PREVIEW_BRUSH"]],
                )
                print("OSC send: /listen 1", flush=True)
                controller.osc_client.send_message("/listen", 1)
            elif sub_cmd in {'pickup_location', 'move_pickup', 'align_pickup', 'preview_pickup'}:
                print("OSC send: /listen 0", flush=True)
                controller.osc_client.send_message("/listen", 0)
                state = brush_selector.move_to_pickup_location(int(max_msg[2]))
                print(f"OSC brush done: pickup_location -> {state}", flush=True)
                print(
                    f"OSC send: /brush/done {['pickup_location', state['PREVIEW_BRUSH']]}",
                    flush=True,
                )
                controller.osc_client.send_message(
                    "/brush/done",
                    ["pickup_location", state["PREVIEW_BRUSH"]],
                )
                print("OSC send: /listen 1", flush=True)
                controller.osc_client.send_message("/listen", 1)
            elif sub_cmd in {'capture', 'capture_pickup', 'record_pickup', 'set_pickup'}:
                if len(max_msg) < 3:
                    print("OSC brush capture_pickup error: expected brush number", flush=True)
                    controller.osc_client.send_message(
                        "/brush/error",
                        ["capture_pickup", "missing_brush_number"],
                    )
                    return
                print("OSC send: /listen 0", flush=True)
                controller.osc_client.send_message("/listen", 0)
                state = brush_selector.record_pickup(int(max_msg[2]), derive_related=True)
                print(f"OSC brush done: capture_pickup -> {state}", flush=True)
                print(
                    f"OSC send: /brush/done {['capture_pickup', state['PREVIEW_BRUSH']]}",
                    flush=True,
                )
                controller.osc_client.send_message(
                    "/brush/done",
                    ["capture_pickup", state["PREVIEW_BRUSH"]],
                )
                print("OSC send: /listen 1", flush=True)
                controller.osc_client.send_message("/listen", 1)
            elif sub_cmd in {'record_ready', 'set_ready'}:
                print("OSC send: /listen 0", flush=True)
                controller.osc_client.send_message("/listen", 0)
                state = brush_selector.record_ready(int(max_msg[2]))
                print(f"OSC brush done: record_ready -> {state}", flush=True)
                print(f"OSC send: /brush/done {['record_ready', state['CURRENT_BRUSH']]}", flush=True)
                controller.osc_client.send_message("/brush/done", ["record_ready", state["CURRENT_BRUSH"]])
                print("OSC send: /listen 1", flush=True)
                controller.osc_client.send_message("/listen", 1)
            elif sub_cmd == 'ready':
                print("OSC send: /listen 0", flush=True)
                controller.osc_client.send_message("/listen", 0)
                before_state = brush_selector.status()
                state = brush_selector.ready(int(max_msg[2]))
                _draw_bci_stroke_if_brush_changed(before_state, state, "ready")
                print(f"OSC brush done: ready -> {state}", flush=True)
                print(f"OSC send: /brush/done {['ready', state['CURRENT_BRUSH']]}", flush=True)
                controller.osc_client.send_message("/brush/done", ["ready", state["CURRENT_BRUSH"]])
                print("OSC send: /listen 1", flush=True)
                controller.osc_client.send_message("/listen", 1)
            elif sub_cmd == 'move_next':
                print("OSC send: /listen 0", flush=True)
                controller.osc_client.send_message("/listen", 0)
                state = brush_selector.move_next_location()
                print(f"OSC brush done: move_next -> {state}", flush=True)
                print(
                    f"OSC send: /brush/done {['move_next', state['PREVIEW_BRUSH']]}",
                    flush=True,
                )
                controller.osc_client.send_message(
                    "/brush/done",
                    ["move_next", state["PREVIEW_BRUSH"]],
                )
                print("OSC send: /listen 1", flush=True)
                controller.osc_client.send_message("/listen", 1)
            elif sub_cmd == 'next_left':
                print("OSC send: /listen 0", flush=True)
                controller.osc_client.send_message("/listen", 0)
                before_state = brush_selector.status()
                state = brush_selector.move_next_left()
                _draw_bci_stroke_if_brush_changed(before_state, state, "next_left")
                print(f"OSC brush done: next_left -> {state}", flush=True)
                print(
                    f"OSC send: /brush/done {['next_left', state['CURRENT_BRUSH']]}",
                    flush=True,
                )
                controller.osc_client.send_message(
                    "/brush/done",
                    ["next_left", state["CURRENT_BRUSH"]],
                )
                print("OSC send: /listen 1", flush=True)
                controller.osc_client.send_message("/listen", 1)
            elif sub_cmd == 'next_right':
                print("OSC send: /listen 0", flush=True)
                controller.osc_client.send_message("/listen", 0)
                before_state = brush_selector.status()
                state = brush_selector.move_next_right()
                _draw_bci_stroke_if_brush_changed(before_state, state, "next_right")
                print(f"OSC brush done: next_right -> {state}", flush=True)
                print(
                    f"OSC send: /brush/done {['next_right', state['CURRENT_BRUSH']]}",
                    flush=True,
                )
                controller.osc_client.send_message(
                    "/brush/done",
                    ["next_right", state["CURRENT_BRUSH"]],
                )
                print("OSC send: /listen 1", flush=True)
                controller.osc_client.send_message("/listen", 1)
            elif sub_cmd == 'select_current':
                print("OSC send: /listen 0", flush=True)
                controller.osc_client.send_message("/listen", 0)
                before_state = brush_selector.status()
                state = brush_selector.select_current()
                _draw_bci_stroke_if_brush_changed(before_state, state, "select_current")
                print(f"OSC brush done: select_current -> {state}", flush=True)
                print(
                    f"OSC send: /brush/done {['select_current', state['CURRENT_BRUSH']]}",
                    flush=True,
                )
                controller.osc_client.send_message(
                    "/brush/done",
                    ["select_current", state["CURRENT_BRUSH"]],
                )
                print("OSC send: /listen 1", flush=True)
                controller.osc_client.send_message("/listen", 1)
            elif sub_cmd in {'more_paint', 'dip', 'refresh_paint'}:
                print("OSC send: /listen 0", flush=True)
                controller.osc_client.send_message("/listen", 0)
                state = brush_selector.more_paint()
                print(f"OSC brush done: more_paint -> {state}", flush=True)
                print(
                    f"OSC send: /brush/done {['more_paint', state['CURRENT_BRUSH']]}",
                    flush=True,
                )
                controller.osc_client.send_message(
                    "/brush/done",
                    ["more_paint", state["CURRENT_BRUSH"]],
                )
                print("OSC send: /listen 1", flush=True)
                controller.osc_client.send_message("/listen", 1)
            elif sub_cmd == 'status':
                state = brush_selector.status()
                payload = [int(state["HOLDING_BRUSH"]), int(state["CURRENT_BRUSH"])]
                preview_payload = [int(state.get("PREVIEW_BRUSH", 0))]
                print(f"OSC brush status: {state}", flush=True)
                print(f"OSC send: /brush/state {payload}", flush=True)
                controller.osc_client.send_message(
                    "/brush/state",
                    payload,
                )
                print(f"OSC send: /brush/preview {preview_payload}", flush=True)
                controller.osc_client.send_message("/brush/preview", preview_payload)
                config = brush_selector.motion_config()
                motion_payload = [config["arm_speed"], config["arm_acc"]]
                print(f"OSC send: /brush/motion {motion_payload}", flush=True)
                controller.osc_client.send_message("/brush/motion", motion_payload)
                grip_config = brush_selector.gripper_strength_config()
                grip_payload = [
                    grip_config["close_angle"],
                    grip_config["min_close_angle"],
                    grip_config["max_close_angle"],
                    grip_config["step_degrees"],
                    int(grip_config.get("applied_live", False)),
                ]
                print(f"OSC send: /brush/gripper_strength {grip_payload}", flush=True)
                controller.osc_client.send_message("/brush/gripper_strength", grip_payload)
            elif sub_cmd in {'gripper_stronger', 'grip_stronger', 'stronger'}:
                step = float(max_msg[2]) if len(max_msg) > 2 else 2.0
                config = brush_selector.adjust_gripper_strength(abs(step))
                payload = [
                    config["close_angle"],
                    config["min_close_angle"],
                    config["max_close_angle"],
                    config["step_degrees"],
                    int(config.get("applied_live", False)),
                ]
                print(f"OSC brush gripper stronger: {config}", flush=True)
                print(f"OSC send: /brush/gripper_strength {payload}", flush=True)
                controller.osc_client.send_message("/brush/gripper_strength", payload)
            elif sub_cmd in {'gripper_softer', 'grip_softer', 'softer'}:
                step = float(max_msg[2]) if len(max_msg) > 2 else 2.0
                config = brush_selector.adjust_gripper_strength(-abs(step))
                payload = [
                    config["close_angle"],
                    config["min_close_angle"],
                    config["max_close_angle"],
                    config["step_degrees"],
                    int(config.get("applied_live", False)),
                ]
                print(f"OSC brush gripper softer: {config}", flush=True)
                print(f"OSC send: /brush/gripper_strength {payload}", flush=True)
                controller.osc_client.send_message("/brush/gripper_strength", payload)
            elif sub_cmd in {'gripper_strength', 'grip_strength', 'gripper', 'grip'}:
                action = str(max_msg[2]).strip().lower() if len(max_msg) > 2 else 'status'
                if action in {'status', 'get'}:
                    config = brush_selector.gripper_strength_config()
                elif action in {'stronger', 'up', 'increase'}:
                    step = float(max_msg[3]) if len(max_msg) > 3 else 2.0
                    config = brush_selector.adjust_gripper_strength(abs(step))
                elif action in {'softer', 'down', 'decrease'}:
                    step = float(max_msg[3]) if len(max_msg) > 3 else 2.0
                    config = brush_selector.adjust_gripper_strength(-abs(step))
                elif action in {'set', 'angle'}:
                    config = brush_selector.set_gripper_strength(float(max_msg[3]))
                else:
                    config = brush_selector.set_gripper_strength(float(max_msg[2]))
                payload = [
                    config["close_angle"],
                    config["min_close_angle"],
                    config["max_close_angle"],
                    config["step_degrees"],
                    int(config.get("applied_live", False)),
                ]
                print(f"OSC brush gripper strength: {config}", flush=True)
                print(f"OSC send: /brush/gripper_strength {payload}", flush=True)
                controller.osc_client.send_message("/brush/gripper_strength", payload)
            elif sub_cmd in {'speed', 'arm_speed'}:
                config = brush_selector.set_motion_config(arm_speed=float(max_msg[2]))
                payload = [config["arm_speed"], config["arm_acc"]]
                print(f"OSC brush speed set: {config}", flush=True)
                print(f"OSC send: /brush/motion {payload}", flush=True)
                controller.osc_client.send_message("/brush/motion", payload)
            elif sub_cmd in {'acc', 'arm_acc', 'accel', 'acceleration'}:
                config = brush_selector.set_motion_config(arm_acc=float(max_msg[2]))
                payload = [config["arm_speed"], config["arm_acc"]]
                print(f"OSC brush acceleration set: {config}", flush=True)
                print(f"OSC send: /brush/motion {payload}", flush=True)
                controller.osc_client.send_message("/brush/motion", payload)
            elif sub_cmd in {'motion', 'motion_config'}:
                config = brush_selector.set_motion_config(
                    arm_speed=float(max_msg[2]),
                    arm_acc=float(max_msg[3]),
                )
                payload = [config["arm_speed"], config["arm_acc"]]
                print(f"OSC brush motion set: {config}", flush=True)
                print(f"OSC send: /brush/motion {payload}", flush=True)
                controller.osc_client.send_message("/brush/motion", payload)
            else:
                print(f"OSC brush error: unknown sub-command {sub_cmd}", flush=True)
                print(f"OSC send: /brush/error {[sub_cmd, 'unknown_sub_command']}", flush=True)
                controller.osc_client.send_message("/brush/error", [sub_cmd, "unknown_sub_command"])
        except (IndexError, ValueError) as exc:
            print(f"OSC brush payload error: args={max_msg} error={exc}", flush=True)
            print(f"OSC send: /brush/error {['payload', str(exc)]}", flush=True)
            controller.osc_client.send_message("/brush/error", ["payload", str(exc)])
            print("OSC send: /listen 1", flush=True)
            controller.osc_client.send_message("/listen", 1)
        except BrushSelectorError as exc:
            print(f"OSC brush command failed: {exc}", flush=True)
            print(f"OSC send: /brush/error {[sub_cmd, str(exc)]}", flush=True)
            controller.osc_client.send_message("/brush/error", [sub_cmd, str(exc)])
            print("OSC send: /listen 1", flush=True)
            controller.osc_client.send_message("/listen", 1)
            
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



    elif cmd == 'armtrack':
        _handle_armtrack_command(max_msg)

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
muse_osc_server_instance = None
last_muse_action_times = {}
muse_strokes_since_brush_change = MUSE_BRUSH_CHANGE_STROKE_LOCKOUT

def on_exit(controller_instance, server_instance, muse_server_instance=None):
    print("Exiting program...")
    if controller_instance:
        try:
            controller_instance.command_video_recorder.stop()
        except Exception as exc:
            print(f"Command video recorder stop failed during shutdown: {exc}", flush=True)
        print("Disconnecting XArm controller...")
        controller_instance.disconnect()
    if server_instance:
        print("Closing OSC server...")
        server_instance.server_close()
    if muse_server_instance:
        print("Closing Muse OSC server...")
        muse_server_instance.server_close()
    print("Cleanup finished.")

def main():
    global controller # Declare controller as global to modify it
    global brush_selector
    global osc_server_instance
    global muse_osc_server_instance

    # Instantiate controller
    controller = XArmController(ROBOT_IP)
    brush_selector = BrushSelector(controller.arm, controller.osc_client)
    if ENABLE_STARTUP_ARM_MOVE:
        controller.arm.set_servo_angle(servo_id=5, angle=90, wait=True) # Optional setup move
    else:
        print("Startup arm move disabled; arm will stay at its current position.", flush=True)
    controller.osc_client.send_message("/listen", 1)

    # Setup OSC server
    osc_dispatcher = dispatcher.Dispatcher()
    osc_dispatcher.map("/macbook", interactive) # interactive will use the global controller
    osc_dispatcher.map("/macbook/*", macbook_path_interactive)
    osc_dispatcher.set_default_handler(unhandled_osc)

    osc_server_instance = osc_server.ThreadingOSCUDPServer((LINUX_IP, MAXMSP_FROM_PORT), osc_dispatcher)
    print("Serving on {}".format(osc_server_instance.server_address))

    if MUSE_OSC_ENABLED:
        muse_dispatcher = dispatcher.Dispatcher()
        muse_dispatcher.map("/muse/*", muse_osc_handler)
        muse_dispatcher.set_default_handler(muse_osc_handler)
        try:
            muse_osc_server_instance = osc_server.ThreadingOSCUDPServer(
                (MUSE_OSC_BIND_IP, MUSE_OSC_BIND_PORT),
                muse_dispatcher,
            )
            muse_thread = threading.Thread(
                target=muse_osc_server_instance.serve_forever,
                name="muse-osc-server",
                daemon=True,
            )
            muse_thread.start()
            print(
                "Muse OSC listening on "
                f"{muse_osc_server_instance.server_address}; "
                "send Muse to this machine on UDP port "
                f"{MUSE_OSC_BIND_PORT}.",
                flush=True,
            )
        except OSError as exc:
            print(
                f"Muse OSC listener disabled; failed to bind "
                f"{MUSE_OSC_BIND_IP}:{MUSE_OSC_BIND_PORT}: {exc}",
                flush=True,
            )

    # Register exit handler
    # Pass controller and server to on_exit using a lambda or functools.partial
    # because atexit doesn't pass arguments to registered functions directly.
    atexit.register(lambda: on_exit(controller, osc_server_instance, muse_osc_server_instance))

    try:
        osc_server_instance.serve_forever()
    except KeyboardInterrupt:
        print("Keyboard interrupt received, shutting down.")
    finally:
        # Ensure cleanup is called if serve_forever() exits unexpectedly
        on_exit(controller, osc_server_instance, muse_osc_server_instance)

if __name__ == '__main__':
    main()
