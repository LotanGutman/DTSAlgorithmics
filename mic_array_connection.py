import numpy as np
import keyboard
import pyaudio
import matplotlib.pyplot as plt

import signal_processing
import constants
from sun_pygame import SunDisplay3D
from robotics import Robot


class MicArray:
    def __init__(self, rate=constants.SAMPLE_RATE, chunk_size=constants.CHUNK_SIZE, robot: Robot = None):
        self.pyaudio_instance = None
        self.stream = None
        self.channels = None
        self.sample_rate = rate
        self.chunk_size = chunk_size if chunk_size else None
        self.plot_lines = None
        self.fig = None
        self.ax = None
        self.robot = robot if robot else Robot()

    def _select_mic_device_index(self):
        print("-" * 120)
        print("Available miniDSP non-speaker audio devices:")
        print("-" * 120)

        max_channels = 0
        max_channels_device_index = None
        max_name = ''

        for i in range(self.pyaudio_instance.get_device_count()):
            dev = self.pyaudio_instance.get_device_info_by_index(i)
            name = dev['name'].encode('utf-8')
            if 'miniDSP' in dev['name'] and "Speakers" not in dev['name']:
                input_channels = dev['maxInputChannels']
                print(f'index={i}, name={name}, in_channels={input_channels}')
                if input_channels > max_channels:
                    max_channels = input_channels
                    max_name = name
                    max_channels_device_index = i

        if max_channels_device_index is None:
            raise Exception('Cannot find input device')

        self.channels = max_channels
        self.chunk_size = self.chunk_size if self.chunk_size else self.sample_rate // self.channels

        print("-" * 120)
        print("Selected device:")
        print("-" * 120)
        print(f'Audio device: name={max_name}, channels={max_channels}')

        return max_channels_device_index

    def _setup_plot(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()

        times = [i - constants.CORRELATION_WINDOW for i in range(constants.CORRELATION_RANGE)]
        y0 = [0] * constants.CORRELATION_RANGE

        self.ax.clear()
        self.plot_lines = [self.ax.plot(times, y0)[0] for _ in range(self.channels - 1)]
        for p in self.plot_lines:
            p.set_marker('o')

    def _process_audio_data(self, data):
        samples = np.frombuffer(data, dtype=np.int16)
        samples = samples.astype(np.float32)
        num_samples = samples.shape[0] // self.channels
        samples_by_channel = samples.reshape((num_samples, self.channels))
        samples_by_channel = samples_by_channel[:, :self.channels].copy()
        samples_by_channel = samples_by_channel[:, :7]

        for i in range(samples_by_channel.shape[1]):
            samples_by_channel[:, i] = signal_processing.center(samples_by_channel[:, i])

        return samples_by_channel

    def _calculate_angles(self, samples_by_channel):
        strength, calculated_phi, strongest_tau, values, angles, powers = signal_processing.calculate_strengthes(
            samples_by_channel)

        taus = powers / constants.SAMPLE_RATE
        calculated_theta = signal_processing.find_theta(taus, calculated_phi)
        calculated_theta = np.pi / 2 - calculated_theta

        return calculated_phi, calculated_theta

    def _update_visualization(self, samples_by_channel, calculated_phi, calculated_theta, sun3d):
        sun3d.update(calculated_phi, calculated_theta)
        self.robot.update(calculated_theta, calculated_phi)

        mic2_signal = samples_by_channel[:, 0]
        mic5_signal = samples_by_channel[:, 6]
        r = signal_processing.corr(mic2_signal, mic5_signal)

        self.plot_lines[0].set_ydata(r)
        self.ax.set_title(f"correlation for mics 0&6, σ= {signal_processing.sigma(r)}")

        self.ax.relim()
        self.ax.autoscale_view()
        plt.draw()
        plt.pause(0.1)

    def run(self):
        self.pyaudio_instance = pyaudio.PyAudio()
        device_index = self._select_mic_device_index()

        self.stream = self.pyaudio_instance.open(
            format=constants.SAMPLE_FORMAT,
            channels=self.channels,
            rate=self.sample_rate,
            frames_per_buffer=self.chunk_size,
            input=True,
            input_device_index=device_index,
        )

        print("-" * 120)
        print("Selected and automatic settings:")
        print("-" * 120)
        print("Chunk size:", self.chunk_size)
        print("Channels:", self.channels)
        print("-" * 120)
        print("Starting measurements...")
        print("-" * 120)

        self._setup_plot()
        sun3d = SunDisplay3D()

        try:
            while True:
                data = self.stream.read(self.chunk_size)
                samples_by_channel = self._process_audio_data(data)
                calculated_phi, calculated_theta = self._calculate_angles(samples_by_channel)
                self._update_visualization(samples_by_channel, calculated_phi, calculated_theta, sun3d)

                if keyboard.is_pressed('w'):
                    break
        finally:
            print('Finished')
            self.stream.stop_stream()
            self.stream.close()
            self.pyaudio_instance.terminate()

        plt.pause(100)


def audio_capture():
    robot = Robot()
    mic = MicArray(robot=robot)
    mic.run()


if __name__ == '__main__':
    audio_capture()
