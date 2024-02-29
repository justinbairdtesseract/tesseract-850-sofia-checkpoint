import cv2
import numpy as np
from scipy.optimize import minimize

class ImageToRoboticCommands:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = None
        self.edges = None
        self.features = []  # Unified list of dots, lines, and arcs

    def load_and_preprocess_image(self):
        self.image = cv2.imread(self.image_path)
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        self.edges = cv2.Canny(blurred_image, threshold1=50, threshold2=150)

    def detect_lines(self):
        lines = cv2.HoughLinesP(self.edges, 1, np.pi / 180, threshold=50, minLineLength=10, maxLineGap=250)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            self.features.append({'type': 'line', 'start': (x1, y1), 'end': (x2, y2)})


    ### ARC DETECTION
    def segment_edges(self):
        contours, _ = cv2.findContours(self.edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [cv2.approxPolyDP(contour, epsilon=1.0, closed=False) for contour in contours]

    def fit_arc_to_points(self, points):
        def calculate_residuals(params):
            cx, cy, r = params
            residuals = np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2) - r
            return residuals

        def objective_function(params):
            return np.sum(calculate_residuals(params) ** 2)

        x_m = np.mean(points[:, 0])
        y_m = np.mean(points[:, 1])
        initial_guess = [x_m, y_m, np.std(points[:, 0]) + np.std(points[:, 1])]  # Improved initial radius guess

        result = minimize(objective_function, initial_guess, method='L-BFGS-B')

        if result.success:
            fitted_params = result.x
            return fitted_params
        else:
            return None

    def detect_arcs(self):
        segments = self.segment_edges()
        for segment in segments:
            if len(segment) < 5:  # Simple filter for potential arc segments
                continue
            points = np.squeeze(segment, axis=1)  # Remove redundant dimension
            arc_params = self.fit_arc_to_points(points)
            if arc_params is not None:
                self.arcs.append(arc_params)


    ### NEED TO ADD DOTS

    def randomize_features(self):
        random.shuffle(self.features)

    def process_image(self):
        self.load_and_preprocess_image()
        self.detect_lines()
        self.detect_arcs()
        # self.detect_dots() 
        self.randomize_features()
