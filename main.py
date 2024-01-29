from xarm.wrapper import XArmAPI

class XArmController:
    def __init__(self):
        self.IP_ADDRESS = 'add IP'
        self.arm = XArmAPI(self.IP_ADDRESS)
        self.setup()

    def setup(self):
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)  # Position control mode
        self.arm.set_state(state=0)  # Set arm to 'ready' state

    def move_to_position(self, x, y, z, speed=100):
        self.arm.set_position(x=x, y=y, z=z, speed=speed, wait=True)

    def move_home(self):
        self.arm.set_position(x=0, y=0, z=0, speed=100, wait=True)

    def disconnect(self):
        self.arm.disconnect()

def main():
    controller = XArmController('your_xarm_ip')
    
    try:
        # Test movements
        controller.move_to_position(200, 0, 200)
        controller.move_home()
    finally:
        # Ensure the arm is disconnected properly
        controller.disconnect()



if __name__ == '__main__':
    main()