# Canvas corner coordinates [x, y, z]
canvas_corners = [
    [546, 205, 835],  # top_left
    [546, -333, 835], # top_right
    [508, 205, 88],   # bottom_left
    [508, -333, 88]   # bottom_right
]

def calculate_scaling_factors():
    z1, x1 = canvas_corners[0][2], canvas_corners[0][0]  # Top left corner
    z2, x2 = canvas_corners[2][2], canvas_corners[2][0]  # Bottom left corner
    a = (x2 - x1) / (z2 - z1)
    b = x1 - a * z1
    return a, b

def x_from_z(z, a, b):
    return a * z + b

def fill_canvas(canvas_corners, line_spacing=10):
    a, b = calculate_scaling_factors()
    
    # Calculate the horizontal range (y-axis) of the canvas
    y_min = min(canvas_corners[0][1], canvas_corners[1][1])
    y_max = max(canvas_corners[0][1], canvas_corners[1][1])

    # Calculate the vertical range (z-axis) of the canvas
    z_min = min(canvas_corners[0][2], canvas_corners[2][2])
    z_max = max(canvas_corners[0][2], canvas_corners[2][2])

    # Determine the direction of filling based on the tilt of the canvas
    start_z, end_z = z_max, z_min

    # Start filling from the top of the canvas to the bottom
    current_y = y_min
    while current_y <= y_max:
        start = [x_from_z(start_z, a, b), current_y, start_z]
        end = [x_from_z(end_z, a, b), current_y, end_z]

        # Print the simulated movement
        print(f"Simulating line from {start} to {end}")

        # Move to the next position based on the line spacing
        current_y += line_spacing

        # Swap start and end z positions to create a back-and-forth fill pattern
        start_z, end_z = end_z, start_z

# Now, call fill_canvas to simulate filling the canvas
fill_canvas(canvas_corners, line_spacing=10)
