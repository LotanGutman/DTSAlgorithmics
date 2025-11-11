import threading
import time

import numpy as np
import serial
import constants

# todo: add interpolation, to "upsample" the estimated angles
class Robot:
    def __init__(self, serial_port="COM6", updateTime=0.25):
        self.ser = serial.Serial(serial_port, 115200)

        self.updateTime = updateTime

        # Thread safe storage for the most recent command
        self._lock = threading.Lock()
        self._latest_command = None
        self._stop_event = threading.Event()

        # Send calibration command
        self.ser.write("-1".encode())
        time.sleep(15)  # wait for calibration to complete

        # Start periodic sender thread
        self._sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
        self._sender_thread.start()


    def update(self, theta, phi):
        theta, phi = float(theta), float(phi)

        phi = 2*np.pi - phi % (2 * np.pi)  # between 0 and 2pi # the xy angle. The coordinate system defined needs improvment.

        phiSteps = phi * constants.NUM_STEPS_PHI_ANGLE / (2 * np.pi)

        command = f"{int(phiSteps)}, 1\n"

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


    def close(self):
        # Stop the background thread and close serial connection
        self._stop_event.set()
        if self._sender_thread.is_alive():
            self._sender_thread.join(timeout=1.0)
        self.ser.close()


    def __del__(self):
        try:
            self.close()
        except:
            pass
