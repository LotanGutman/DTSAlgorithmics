from numpy import pi, sin, cos, linspace, array
from pyaudio import paInt16

nnnn = 15 # 15
NNNN = 2*nnnn + 1

p = 0.75  # level of "trust" in the last angle calculated vs the previously calculated angle # hi shrot # todo: actally implement LPF for TI angle

c = 343

fs = 48000

chunk_size = 10000

window_len = 50

alpha = 0.2



r = 0.04 # [m] = 4.5 [cm]
num_circular_mics = 6
angles = linspace(0, 2*pi, num_circular_mics, endpoint=False)

# nuttal window BPF stuff
center_freq = 2500  # Hz
bandwidth = 3000    # Hz
low_cutoff = center_freq - bandwidth / 2  # 1000 Hz
high_cutoff = center_freq + bandwidth / 2  # 4000 Hz
numtaps = 1010  # Number of filter coefficients (higher → sharper filter)


to_show_propability = False


mics = [[0, 0]]
for angle in angles:
    mics.append([r*cos(angle), r*sin(angle)])

mics = array(mics)

SAMPLE_FORMAT = paInt16  # pyaudio.paInt16
