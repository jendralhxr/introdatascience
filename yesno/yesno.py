import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram

sr, y = wavfile.read("live_yesno.wav")

# Convert stereo to mono and normalize to [-1, 1] if needed
if y.ndim > 1:
    y = y.mean(axis=1)
y = y / np.max(np.abs(y))

time = np.linspace(0, len(y) / sr, num=len(y))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# 1. Raw Waveform
ax1.plot(time, y, linewidth=0.6)
ax1.set_title("Raw Waveform")
ax1.set_ylabel("Amplitude")
ax1.grid(True, alpha=0.3)

# 2. Spectrogram
f, t, Sxx = spectrogram(y, fs=sr, nperseg=512, noverlap=384)
ax2.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading="gouraud", cmap="nipy_spectral")
ax2.set_title("Spectrogram")
ax2.set_ylabel("Frequency (Hz)")
ax2.set_xlabel("Time (s)")
ax2.set_ylim(0, 5000)

plt.tight_layout()
plt.show()

#--

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 1. Load audio and resample to 10 kHz (standard for formant analysis up to 5 kHz)
y, sr = librosa.load("live_yesno.wav", sr=10000)

# Pre-emphasis filter boosts higher frequencies so F2 isn't drowned out by F1
y_pre = librosa.effects.preemphasis(y, coef=0.97)

# 2. Wideband STFT (~6.4 ms window -> win_length=64 at 10 kHz) to highlight formants
n_fft = 256
win_length = 64
hop_length = 32

S = np.abs(librosa.stft(y_pre, n_fft=n_fft, win_length=win_length, hop_length=hop_length))
S_db = librosa.amplitude_to_db(S, ref=np.max)

# 3. Extract Formant Frequencies via LPC
def get_formants(signal, sr, frame_length=256, hop_length=32, order=12):
    frames = librosa.util.frame(signal, frame_length=frame_length, hop_length=hop_length)
    times = librosa.frames_to_time(np.arange(frames.shape[1]), sr=sr, hop_length=hop_length)
    f1_list, f2_list = [], []

    for frame in frames.T:
        windowed = frame * np.hamming(len(frame))
        if np.sum(windowed**2) < 1e-4:
            f1_list.append(np.nan)
            f2_list.append(np.nan)
            continue

        a = librosa.lpc(windowed, order=order)
        roots = np.roots(a)
        roots = [r for r in roots if np.imag(r) >= 0.01]

        angs = np.arctan2(np.imag(roots), np.real(roots))
        freqs = angs * (sr / (2 * np.pi))
        bws = -0.5 * (sr / (2 * np.pi)) * np.log(np.abs(roots))

        valid_formants = sorted([f for f, bw in zip(freqs, bws) if 90 < f < 3500 and bw < 400])
        f1_list.append(valid_formants[0] if len(valid_formants) > 0 else np.nan)
        f2_list.append(valid_formants[1] if len(valid_formants) > 1 else np.nan)

    return times, np.array(f1_list), np.array(f2_list)

times, f1, f2 = get_formants(y_pre, sr, frame_length=256, hop_length=hop_length, order=12)

# 4. Plot Zoomed-In 0–2000 Hz Spectrogram (Loud = White) + Tracked Formants
fig, ax = plt.subplots(figsize=(10, 3.5))
img = librosa.display.specshow(
    S_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="hz", ax=ax, cmap="gray"
)

# Overlay F1 and F2 tracks with high-contrast colors
ax.plot(times, f1, "o", color="#ff3b30", markersize=4, label="F1 (Tongue Height)")
ax.plot(times, f2, "o", color="#00e5ff", markersize=4, label="F2 (Tongue Frontness)")

ax.set_ylim(0, 2000)  # Expanded to 2,000 Hz to capture F2's up-down trajectory
ax.set_yticks(np.arange(0, 2001, 250))  # Clean 250 Hz gridlines for reading trajectories
ax.set_title("Formant Tracking (0–2,000 Hz)")
ax.set_ylabel("Frequency (Hz)")
ax.set_xlabel("Time (s)")
ax.legend(loc="upper right")
ax.grid(True, color="white", alpha=0.2)

fig.colorbar(img, ax=ax, format="%+2.0f dB")
plt.tight_layout()
plt.show()