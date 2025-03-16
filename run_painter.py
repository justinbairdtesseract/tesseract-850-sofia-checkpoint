#!/usr/bin/env python3
"""
Test script for the SVG Painter to validate its functionality
"""

import os
import time
from svg_painter import SVGPainter

def main():
    # Initialize the SVG Painter with the robot's IP address
    painter = SVGPainter(ip_address='192.168.0.203')
    
    try:
        print("\n===== SVG Painter Test Script =====\n")
        
        # Test basic movement
        print("Testing basic robot movement...")
        result = painter.arm.set_position(x=400, y=0, z=400, roll=-180, pitch=-90, yaw=0, speed=50, wait=True)
        print(f"Movement test result: {result}")
        time.sleep(2)
        
        # Test the drawing capabilities
        print("\n----- Drawing Tests -----")
        print("Which test would you like to run?")
        print("1. Square test")
        print("2. Circle test")
        print("3. Zigzag test")
        print("4. Calibration pattern")
        print("5. Draw SVG file")
        print("6. Run all tests")
        
        choice = input("Enter choice (1-6): ")
        
        if choice == '1' or choice == '6':
            print("\nRunning square test...")
            painter.test_drawing(test_type="square", size=100, speed=20)
            time.sleep(1)
            
        if choice == '2' or choice == '6':
            print("\nRunning circle test...")
            painter.test_drawing(test_type="circle", size=100, speed=20)
            time.sleep(1)
            
        if choice == '3' or choice == '6':
            print("\nRunning zigzag test...")
            painter.test_drawing(test_type="zigzag", size=100, speed=20)
            time.sleep(1)
            
        if choice == '4' or choice == '6':
            print("\nRunning calibration pattern...")
            painter.test_drawing(test_type="calibration", size=100, speed=20)
            time.sleep(1)
            
        if choice == '5' or choice == '6':
            svg_file = input("\nEnter SVG file path (or press Enter for default 'test.svg'): ")
            if not svg_file:
                svg_file = "3.svg"
                
            if not os.path.exists(svg_file):
                print(f"Error: File '{svg_file}' not found.")
            else:
                print(f"\nPainting SVG file: {svg_file}")
                painter.paint_svg(svg_file, speed=20)
                
        print("\nTests completed!")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"\nError during test: {e}")
    finally:
        print("\nDisconnecting from robot...")
        painter.disconnect()
        print("Done.")

if __name__ == "__main__":
    main()