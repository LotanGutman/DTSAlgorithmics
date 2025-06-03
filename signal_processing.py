from collections import deque

import numpy as np
from numpy.fft import fftfreq
from scipy.fft import fft

import constants
from scipy.optimize import minimize_scalar
from scipy.signal import fftconvolve
from scipy.signal.windows import gaussian

thetas = deque([0.0, ], constants.window_len)
phis = deque([0.0, ], constants.window_len)


def circular_avg(num1, num2):
    return num2

    return (num1 + num2) / 2

    num1 = num1 % (2 * np.pi)
    num2 = num2 % (2 * np.pi)

    delta = (num2 - num1) % (2 * np.pi)
    if delta > np.pi:
        delta -= 2 * np.pi

    # Adjust num2 to be close to num1
    num2_adjusted = num1 + delta

    # Compute average
    avg = (num1 + num2_adjusted) / 2

    # Normalize to [0, 2π)
    return avg % (2 * np.pi)


def sigma(correlationArray):
    no_dc = center(correlationArray)
    var = np.sum(no_dc ** 2) / (len(no_dc) - 1)
    return np.sqrt(var)


def center(signal):
    s = signal - np.mean(signal)
    return s

def coherence_sum(signal):  # will iterate through all indices; if x[i] < 0, it will add it to x[i +- 3] and zero x[i]
    # return signal

    for i in range(len(signal)):
        if signal[i] < 0:
            signal[(i + 3) % 6] = (signal[(i + 3) % 6] - signal[i]) / 2
            signal[i] = 0
    return signal


def rms(signal):
    return np.sqrt(np.mean(signal ** 2))


def corr(signal1, signal2, n=constants.nnnn):
    signal1 = np.asarray(signal1)
    signal2 = np.asarray(signal2)

    length = len(signal1)
    if length != len(signal2):
        raise ValueError("signal1 and signal2 must be the same length")

    correlation = np.zeros(2 * n + 1)

    for lag in range(-n, n + 1):
        shifted = np.roll(signal2, lag)  # circular shift
        correlation[lag + n] = np.dot(signal1, shifted)  # fast inner product

    correlation = correlation / (rms(signal1) * rms(signal2) * len(signal1))

    return correlation


def calculate_delay_fourier(signal1, signal2, interpolate=True, fs=constants.fs):
    s1 = np.asarray(signal1)
    s2 = np.asarray(signal2)

    if s1.ndim != 1 or s2.ndim != 1:
        raise ValueError("Signals must be 1-dimensional arrays.")

    L1, L2 = len(s1), len(s2)

    if L1 == 0 or L2 == 0:
        return np.nan

    N = max(L1, L2)

    s1_padded = np.pad(s1, (0, N - L1), mode='constant', constant_values=0)
    s2_padded = np.pad(s2, (0, N - L2), mode='constant', constant_values=0)

    S1_fft = np.fft.fft(s1_padded)
    S2_fft = np.fft.fft(s2_padded)

    G = S1_fft * np.conj(S2_fft)

    abs_G = np.abs(G)
    if np.all(abs_G < 1e-12):
        return np.nan

    G_phat = G / (abs_G + 1e-10)

    cc_complex = np.fft.ifft(G_phat)
    cc = np.real(cc_complex)

    cc_shifted = np.fft.fftshift(cc)

    if N % 2 == 0:
        lags = np.arange(-N // 2, N // 2)
    else:
        lags = np.arange(-(N - 1) // 2, (N - 1) // 2 + 1)

    if not np.any(np.isfinite(cc_shifted)):
        return np.nan

    max_index_shifted = np.argmax(cc_shifted)

    integer_delay = lags[max_index_shifted]

    delta = 0.0
    if interpolate:
        if N >= 3 and 1 <= max_index_shifted < N - 1:
            y_minus = cc_shifted[max_index_shifted - 1]
            y_0 = cc_shifted[max_index_shifted]
            y_plus = cc_shifted[max_index_shifted + 1]

            interpolation_denominator = y_minus - (2 * y_0) + y_plus

            if interpolation_denominator != 0:
                delta = 0.5 * (y_minus - y_plus) / interpolation_denominator

    delay_samples = float(integer_delay) + delta

    if fs is not None:
        if not isinstance(fs, (int, float)) or fs <= 0:
            raise ValueError("Sampling frequency (fs) must be a positive number.")
        return delay_samples
    else:
        return delay_samples

def calculate_delay(signal1, signal2, interpolate=True):
    cc = corr(signal1, signal2)

    if not interpolate:
        return np.argmax(cc) - constants.nnnn

    max_index = np.argmax(cc)

    # Interpolation around the peak for sub-sample accuracy
    if 1 <= max_index < len(cc) - 1:
        y_minus = cc[max_index - 1]
        y_0 = cc[max_index]
        y_plus = cc[max_index + 1]

        denom = 2 * (2 * y_0 - y_plus - y_minus)
        if denom != 0:
            delta = 0.5 * (y_minus - y_plus) / denom
        else:
            delta = 0.0
    else:
        delta = 0.0

    delay = (max_index - constants.nnnn) + delta
    return delay


def calculate_strengthes(signals):  # lotan's method
    global phis

    signal0 = np.array(signals[:, 0])
    signals = np.array([signals[:, i] for i in range(1, 7)])  # 1,2,6
    powers = np.zeros(signals.shape[0])

    for i in range(len(signals)):
        signal = signals[i]
        powers[i] = calculate_delay_fourier(signal0, signal) # calculate_delay(signal0, signal)

    values, angles, strongest_angle, strongest_val = get_TI_angle(powers)

    last_phi = phis[-1]
    strongest_angle = strongest_angle + np.round((last_phi - strongest_angle) / (2 * np.pi)) * (2 * np.pi)
    strongest_angle = constants.alpha * strongest_angle + (1 - constants.alpha) * last_phi
    phis.append(strongest_angle)

    return powers, strongest_angle, strongest_val, values, angles, powers

def get_TI_angle(values, M=3):
    angles = constants.angles

    # values = values - np.min(values)

    angles = np.unwrap(angles)  # remove anomalies in angle (not mandatory)

    X = np.ones((len(angles), 1 + 2 * M))
    for n in range(1, M + 1):
        X[:, 2 * n - 1] = np.cos(n * angles)
        X[:, 2 * n] = np.sin(n * angles)

    # Solve LS
    c, _, _, _ = np.linalg.lstsq(X, values, rcond=None)

    # Reconstructed function
    def f_hat(theta):
        result = c[0] * 0.5  # a0/2 term
        for n in range(1, M + 1):
            result += c[2 * n - 1] * np.cos(n * theta) + c[2 * n] * np.sin(n * theta)
        return result

    angles = np.linspace(0, 2 * np.pi, 100, endpoint=False)

    res = minimize_scalar(
        lambda theta: -f_hat(theta),
        bounds=(0, 2 * np.pi),
        method='bounded',
        options={'xatol': 1e-8}
    )
    strongest_angle = res.x
    strongest_value = f_hat(strongest_angle)
    strongest_angle = strongest_angle % (2 * np.pi)

    if constants.to_show_propability:
        values = [f_hat(angle) for angle in angles]
    else:
        values = None

    return values, angles, strongest_angle, strongest_value

def find_theta(taus, phi):
    global thetas
    delta = taus*constants.c
    ang = np.array([0, 1.04719755, 2.0943951, 3.14159265, 4.1887902, 5.23598776])
    th = np.array([])
    for i in range(len(ang)):
        z = np.cos(ang[i] - phi)
        if abs(z) > 0.1:
            t = np.arccos(delta[i] / (0.045 * z))
            th = np.append(th, t)

    count = 0
    sum = 0
    for i in th:
        if not np.isnan(i):
            count += 1
            sum += i

    theta = sum/count

    last_theta = thetas[-1]
    theta = theta + np.round((last_theta - theta) / (2 * np.pi)) * (2 * np.pi)
    theta = constants.alpha * theta + (1 - constants.alpha) * last_theta
    thetas.append(theta)

    return theta



def fast_iterative_circular_denoise(y, iterations=20, noise_threshold_percentile=0.15):
    """Optimized version of iterative circular Gaussian denoising."""
    y = np.asarray(y)
    if len(y) < 2:
        return y

    x, y_sin = np.cos(y), np.sin(y)

    # Cache FFT computations
    def get_noise_cutoff(signal):
        fft_mag = np.abs(fft(signal - signal.mean()))
        pos_freqs = fft_mag[(fftfreq(len(y)) > 0) & (fftfreq(len(y)) <= 0.5)]
        return np.percentile(pos_freqs, noise_threshold_percentile * 100) if len(pos_freqs) else 0

    # Compute initial noise cutoff
    noise_cutoff = max(get_noise_cutoff(x), get_noise_cutoff(y_sin))
    if noise_cutoff <= 0:
        return y

    # Design the filter kernel once (outside iterations)
    sigma = len(y) / (2 * np.pi * noise_cutoff)
    kernel_size = min(50, max(3, len(y) // 4))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = gaussian(kernel_size, sigma)
    kernel /= kernel.sum()
    pad_size = kernel_size // 2

    # Pre-allocate buffers
    buffer_size = len(y) + 2 * pad_size
    x_buffer = np.empty(buffer_size)
    y_buffer = np.empty(buffer_size)

    for _ in range(iterations):
        # Circular padding using buffer views
        x_buffer[:pad_size] = x[-pad_size:]
        x_buffer[pad_size:-pad_size] = x
        x_buffer[-pad_size:] = x[:pad_size]

        y_buffer[:pad_size] = y_sin[-pad_size:]
        y_buffer[pad_size:-pad_size] = y_sin
        y_buffer[-pad_size:] = y_sin[:pad_size]

        # FFT-based convolution (faster for larger arrays)
        x = fftconvolve(x_buffer, kernel, mode='valid')
        y_sin = fftconvolve(y_buffer, kernel, mode='valid')

    return np.arctan2(y_sin, x)

def pca_denoise(sig, k=1):
    sig = sig.T
    mu = sig.mean(axis=1, keepdims=True)
    xc = sig - mu

    R = xc @ xc.T / xc.shape[1]
    eigvals, U = np.linalg.eigh(R)      # ascending order
    U_k = U[:, -k:]                     # last k columns

    P = U_k @ U_k.T
    denoised = (P @ xc) + mu
    return denoised.T



