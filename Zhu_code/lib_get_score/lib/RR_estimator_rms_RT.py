import numpy as np
from scipy.signal import butter, filtfilt

def RR_estimator_rms_RT(signal, r_peaks, samp_freq):
    QRS_INTERVAL = [-80, 80]  # duration of the QRS complex to extract
    HEARTBEAT_WINDOW = 16  # number of heartbeats in the moving window to estimate respiratory rate
    FFT_LENGTH = 16  # length of power spectrum
    FREQ_RANGE = [0.03, 0.3]  # default frequency range to extract respiratory rates
    AVERAGING_WINDOW = 16  # define window for extracting median RR

    lower_RR = 3
    upper_RR = 40

    if signal.size == 0:
        raise ValueError('ECG data not provided')
    else:
        ecg = signal

    # Filter R-peaks based on QRS interval
    r_peaks = np.asarray(r_peaks)
    r_peaks = r_peaks[(r_peaks > abs(round(QRS_INTERVAL[0] * (samp_freq / 1000)))) &
                      (r_peaks < len(ecg) - round(QRS_INTERVAL[1] * (samp_freq / 1000)))]

    # Extract QRS complexes
    qrses = extract_qrs_complexes(r_peaks, ecg, samp_freq, QRS_INTERVAL)

    # Estimate respiratory rate and RMS
    resprate, rms = rr_estimator(lower_RR, upper_RR, qrses, r_peaks, samp_freq, FFT_LENGTH, HEARTBEAT_WINDOW)

    return rms, r_peaks

def extract_qrs_complexes(rpeaks_corr, ecg, FS, QRS_INTERVAL):
    F1 = 8 / FS  # lower bandpass filter frequency
    F2 = 20 / FS  # upper bandpass filter frequency
    low = F1 * 2
    high = F2 * 2

    # Design bandpass filter
    b, a = butter(2, [low, high], btype='bandpass')
    data = filtfilt(b, a, ecg)

    # Initialize beats matrix
    beatsduration = range(int(round(QRS_INTERVAL[0] * (256 / 1000))), int(round(QRS_INTERVAL[1] * (256 / 1000))))
    beats = np.zeros((len(beatsduration), len(rpeaks_corr)))

    # Extract QRS complexes
    for i in range(len(rpeaks_corr)):
        dur = range(rpeaks_corr[i] + int(round(QRS_INTERVAL[0] * (256 / 1000))),
                    rpeaks_corr[i] + int(round(QRS_INTERVAL[1] * (256 / 1000))))
        beats[:, i] = data[dur]

    return beats

def rr_estimator(low_RR, upper_RR, qrses, rpeaks_corr, FS, FFT_LENGTH, HEARTBEAT_WINDOW):
    resp_range = [low_RR, upper_RR]
    rms = np.zeros(len(qrses[0]))

    # Calculate RMS
    for i in range(len(rms)):
        rms[i] = np.sqrt(np.mean(qrses[:, i] ** 2))

    # Frequency vector
    freq_vector = np.arange(0.5 / (np.floor(FFT_LENGTH / 2) + 1), 0.5, 0.5 / (np.floor(FFT_LENGTH / 2) + 1))

    # R-R intervals
    rrint = np.diff(rpeaks_corr)
    resprate = []

    return resprate, rms