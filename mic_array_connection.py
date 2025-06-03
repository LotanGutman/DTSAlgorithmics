import time

import numpy as np
import keyboard
import pyaudio
import matplotlib.pyplot as plt

import signal_processing
import constants
from sun_pygame import SunDisplay, SunDisplay3D

t = []
plt.ion()
fig, ax = plt.subplots()

# sun = SunDisplay(width=600, height=600, title="My Data Sun")
sun3D = SunDisplay3D()

calculated_theta = 0


class MicArray:
    def __init__(self, rate=constants.fs, chunk_size=constants.chunk_size):
        self.pyaudio_instance = None
        self.stream = None
        self.channels = None
        self.sample_rate = rate
        self.chunk_size = chunk_size if chunk_size else None
        self.frames = []

    def _select_mic_device_index(self):
        print(
            "--------------------------------------------- available miniDSP non speaker audio devices: ---------------------------------------------")

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
            raise Exception('can not find input device')
        self.channels = max_channels
        self.chunk_size = self.chunk_size if self.chunk_size else self.sample_rate // self.channels
        print(
            "------------------------------------------------ selected device: ----------------------------------------------------------")
        print(f'Automatically selected Audio device: name={max_name}, channels={max_channels}')

        return max_channels_device_index

    def run(self):
        global plot_lines, calculated_theta

        self.frames = []
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

        print(
            "------------------------------------------ selected and automatic settings ----------------------------------------------------------")
        print("chunk size: ", self.chunk_size)
        print("channels: ", self.channels)

        print(
            "------------------------------------------------ starting measurments ----------------------------------------------------------")

        times = [i - constants.nnnn for i in
                 range(constants.NNNN)]  # np.linspace(0, self.chunk_size / self.sample_rate, self.chunk_size)
        y0 = [0] * constants.NNNN  # self.chunk_size
        ax.clear()
        plot_lines = [ax.plot(times, y0)[0] for _ in range(self.channels - 1)]
        for p in plot_lines:
            p.set_marker('o')
        # ax.set_ylim([0, 1])

        while True:
            # get data
            data = self.stream.read(self.chunk_size)
            samples = np.frombuffer(data, dtype=np.int16)
            samples = samples.astype(np.float32)
            num_samples = samples.shape[0] // self.channels
            samples_by_channel = samples.reshape((num_samples,
                                                  self.channels))
            samples_by_channel = samples_by_channel[:,
                                 :self.channels].copy()  # removing the trash 8th channel

            # filter data
            for i in range(samples_by_channel.shape[1]):
                samples_by_channel[:, i] = signal_processing.center(samples_by_channel[:, i])

            # calculate tau and so on
            strength, calculated_phi, strongest_tau, values, angles, powers = signal_processing.calculate_strengthes(
                samples_by_channel)

            # again - srot
            taus = powers / constants.fs
            calculated_theta = signal_processing.find_theta(taus, calculated_phi)

            if signal_processing.thetas.__len__() > 9:
                t0 = time.time()
                signal_processing.thetas = signal_processing.deque(signal_processing.fast_iterative_circular_denoise(list(signal_processing.thetas)), constants.window_len)
                signal_processing.phis = signal_processing.deque(signal_processing.fast_iterative_circular_denoise(list(signal_processing.phis)), constants.window_len)
                print(time.time() - t0)

            sun3D.update(calculated_phi, np.pi/2 - calculated_theta)

            # plot stuff:
            mic2_signal = samples_by_channel[:, 0]
            mic5_signal = samples_by_channel[:, 6]
            r = signal_processing.corr(mic2_signal, mic5_signal)
            plot_lines[0].set_ydata(r)
            ax.set_title(f"correlation for mics 0&6, σ= {signal_processing.sigma(r)}")

            """ # plot signal lines
            for i in range(self.channels - 1):
                plot_lines[i].set_ydata(samples_by_channel[:, i])
            """
            ax.relim()  # Recalculate limits
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.1)  # Small pause to allow GUI event loop to update

            if keyboard.is_pressed('w'):
                break

        print('Finished 324432')

        self.stream.stop_stream()
        self.stream.close()
        # Terminate the PortAudio interface
        self.pyaudio_instance.terminate()

        plt.pause(100)


def audio_capture():
    mic = MicArray()
    mic.run()


if __name__ == '__main__':
    audio_capture()
