#!/usr/bin/env python3
from legacy.svg_painter import SVGPainter
import time
import argparse

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='SVG Painter Test Script')
    parser.add_argument('--ip', type=str, default='192.168.0.203', help='xArm IP address')
    parser.add_argument('--svg', type=str, default='simple.svg', help='SVG file to paint')
    parser.add_argument('--speed', type=int, default=15, help='Drawing speed (1-100)')
    parser.add_argument('--calibrate', action='store_false', dest='skip_calibration', 
                        help='Run canvas calibration (use --no-calibrate to skip)')
    parser.set_defaults(skip_calibration=True)
    args = parser.parse_args()
    
    # Create painter instance
    print(f"Connecting to xArm at {args.ip}...")
    painter = SVGPainter(args.ip)
    
    try:
        # Test connection with a simple movement
        print("Testing robot connection...")
        result = painter.arm.set_position(x=400, y=0, z=400, roll=-180, pitch=-90, yaw=0, speed=20, wait=True)
        if result != 0:
            print(f"Warning: Initial movement test returned code {result}")
        else:
            print("Robot connection test successful")
        time.sleep(1)
        
        # Skip calibration by default
        if not args.skip_calibration:
            painter.calibrate_canvas()
        
        # Verify canvas boundaries
        print("\nCanvas boundaries:")
        print(f"Y range: {painter.y_min:.2f} to {painter.y_max:.2f}")
        print(f"Z range: {painter.z_min:.2f} to {painter.z_max:.2f}")
        
        # Start painting
        print(f"\n=== Starting to paint {args.svg} ===")
        print(f"Using drawing speed: {args.speed}")
        success = painter.paint_svg(args.svg, speed=args.speed)
        
        if success:
            print("\n=== Painting completed successfully ===")
        else:
            print("\n=== Painting encountered issues ===")
        
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
    except Exception as e:
        print(f"Error during execution: {e}")


if __name__ == "__main__":
    main()