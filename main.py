from micArray import MicArray
from robotics import Robot

if __name__ == '__main__':
    robot = Robot(updateTime=0.03, disable=False)
    mic = MicArray(robot=robot)
    mic.run()
