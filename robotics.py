import threading
import time

import numpy as np
import serial
import constants

# todo: add interpolation, to "upsample" the estimated angles
class Robot:
    def __init__(self, serial_port="COM6", updateTime=0.25, disable=False):
        self.disable = disable
        if disable:
            return
        try:
            self.ser = serial.Serial(serial_port, 115200)
        except Exception as _:
            print(f"Could not open serial port {serial_port}. Robot disabled.")
            self.disable = True
            return

        self.updateTime = updateTime

        # Thread safe storage for the most recent command
        self._lock = threading.Lock()
        self._latest_command = None
        self._stop_event = threading.Event()

        # Send calibration command
        self._calibrate()

        # Start periodic sender thread
        self._sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
        self._sender_thread.start()


    def _calibrate(self):
        print("Calibrating robot...")
        self.sendString("-1", wait=15)

    def update(self, theta, phi):
        if self.disable:
            return

        theta, phi = float(theta), float(phi)

        phiSteps = self._getPhiSteps(phi)
        thetaSteps = self._getThetaSteps(theta)

        command = f"{int(phiSteps)}, {thetaSteps}\n"

        print(command)

        # Store the latest command
        with self._lock:
            self._latest_command = command


    def _sender_loop(self):
        # ckground thread initialization
        while not self._stop_event.is_set():
            with self._lock:
                command = self._latest_command

            if command is not None:
                self.sendString(command)

            time.sleep(self.updateTime)


    def sendString(self, command="-1", wait=0):
        self.ser.write(command.encode())
        if wait > 0:
            time.sleep(wait)


    def _close(self):
        if self.disable:
            return

        # Stop the background thread and close serial connection
        self._stop_event.set()
        if self._sender_thread.is_alive():
            self._sender_thread.join(timeout=1.0)
        self.ser.close()


    def __del__(self):
        try:
            self._close()
        except:
            pass

    def _getThetaSteps(self, theta):  # convert theta to the "steps" position used by the robot (the polar angle)
        theta = theta % (2 * np.pi)

        thetaSteps = theta * 2 * 1333 / np.pi

        thetaSteps = 1333 - int(thetaSteps) % 1333

        # this is the calibration to adjust for the partially not working step motor were using for the polar angle. Somehow it works :)
        thetaSteps = (np.sqrt((0.0008068*thetaSteps**2-0.0753*thetaSteps+2)*(0.000766*thetaSteps**2))+0.000766*thetaSteps**2) / 2

        return thetaSteps

    def _getPhiSteps(self, phi):  # same for phi (the xy angle (azimuthal))
        phi = 2*np.pi - phi % (2 * np.pi)

        phiSteps = phi * constants.NUM_STEPS_PHI_ANGLE / (2 * np.pi)

        return int(phiSteps)