from collections import deque

import numpy as np
from scipy.optimize import minimize_scalar

import constants

thetas = deque([0.0], constants.SMOOTHING_WINDOW_LEN)
phis = deque([0.0], constants.SMOOTHING_WINDOW_LEN)


def center(signal):
    return signal - np.mean(signal)


def rms(signal):
    return np.sqrt(np.mean(signal ** 2))


def sigma(correlation_array):
    no_dc = center(correlation_array)
    var = np.sum(no_dc ** 2) / (len(no_dc) - 1)
    return np.sqrt(var)


def corr(signal1, signal2, n=constants.CORRELATION_WINDOW):
    signal1 = np.asarray(signal1)
    signal2 = np.asarray(signal2)

    length = len(signal1)
    if length != len(signal2):
        raise ValueError("signal1 and signal2 must be the same length")

    correlation = np.zeros(2 * n + 1)

    for lag in range(-n, n + 1):
        shifted = np.roll(signal2, lag)
        correlation[lag + n] = np.dot(signal1, shifted)

    correlation = correlation / (rms(signal1) * rms(signal2) * len(signal1))

    return correlation


def calculate_delay(signal1, signal2, interpolate=True):
    cc = corr(signal1, signal2)

    if not interpolate:
        return np.argmax(cc) - constants.CORRELATION_WINDOW

    max_index = np.argmax(cc)

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

    delay = (max_index - constants.CORRELATION_WINDOW) + delta
    return delay


def calculate_delay_fourier(signal1, signal2, interpolate=True, fs=constants.SAMPLE_RATE):
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


def get_TI_angle(values, M=3):
    angles = constants.MIC_ANGLES

    angles = np.unwrap(angles)

    X = np.ones((len(angles), 1 + 2 * M))
    for n in range(1, M + 1):
        X[:, 2 * n - 1] = np.cos(n * angles)
        X[:, 2 * n] = np.sin(n * angles)

    c, _, _, _ = np.linalg.lstsq(X, values, rcond=None)

    def f_hat(theta):
        result = c[0] * 0.5
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

    if constants.SHOW_PROBABILITY:
        values = [f_hat(angle) for angle in angles]
    else:
        values = None

    return values, angles, strongest_angle, strongest_value


def calculate_strengthes(signals):  # lotan's method
    global phis

    signal0 = np.array(signals[:, 0])
    signals = np.array([signals[:, i] for i in range(1, 7)])
    powers = np.zeros(signals.shape[0])

    for i in range(len(signals)):
        signal = signals[i]
        powers[i] = calculate_delay_fourier(signal0, signal)

    values, angles, strongest_angle, strongest_val = get_TI_angle(powers)

    last_phi = phis[-1]
    strongest_angle = strongest_angle + np.round((last_phi - strongest_angle) / (2 * np.pi)) * (2 * np.pi)
    strongest_angle = constants.SMOOTHING_ALPHA * strongest_angle + (1 - constants.SMOOTHING_ALPHA) * last_phi
    phis.append(strongest_angle)

    return powers, strongest_angle, strongest_val, values, angles, powers


def find_theta(taus, phi):
    global thetas
    delta = taus * constants.SPEED_OF_SOUND
    ang = np.array([0, 1.04719755, 2.0943951, 3.14159265, 4.1887902, 5.23598776])
    th = np.array([])

    for i in range(len(ang)):
        z = np.cos(ang[i] - phi)
        if abs(z) > 0.1:
            t = np.arccos(delta[i] / (constants.MIC_ARRAY_RADIUS * z))
            th = np.append(th, t)

    count = 0
    sum_val = 0
    for i in th:
        if not np.isnan(i):
            count += 1
            sum_val += i

    try:
        theta = sum_val / count
    except Exception:
        theta = thetas[-1]

    last_theta = thetas[-1]
    theta = theta + np.round((last_theta - theta) / (2 * np.pi)) * (2 * np.pi)
    theta = constants.SMOOTHING_ALPHA * theta + (1 - constants.SMOOTHING_ALPHA) * last_theta
    thetas.append(theta)

    return theta


def polynomial_lpf(data, window_size=10, poly_order=5):
    filtered = np.zeros_like(data, dtype=float)
    half_window = window_size // 2

    for i in range(len(data)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(data), i + half_window + 1)

        window_data = data[start_idx:end_idx]
        window_indices = np.arange(start_idx, end_idx)

        coeffs = np.polyfit(window_indices, window_data, poly_order)

        filtered[i] = np.polyval(coeffs, i)

    return filtered


def pca_denoise(sig, k=1):
    sig = sig.T
    mu = sig.mean(axis=1, keepdims=True)
    xc = sig - mu

    R = xc @ xc.T / xc.shape[1]
    eigvals, U = np.linalg.eigh(R)
    U_k = U[:, -k:]

    P = U_k @ U_k.T
    denoised = (P @ xc) + mu
    return denoised.T
