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

# 3. Extract Pitch (F0) using pYIN on the raw (un-preemphasized) signal
f0, voiced_flag, voiced_probs = librosa.pyin(
    y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"), sr=sr, hop_length=hop_length
)
f0_times = librosa.times_like(f0, sr=sr, hop_length=hop_length)

# 4. Extract Formant Frequencies via LPC
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

# 5. Plot 0–4,000 Hz Spectrogram (Loud = White) + Pitch (F0) + Formants (F1, F2)
fig, ax = plt.subplots(figsize=(10, 4))
img = librosa.display.specshow(
    S_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="hz", ax=ax, cmap="gray"
)

# Overlay F0 (Pitch), F1, and F2 as dashed lines
ax.plot(f0_times, f0, "--", color="#00e5ff", linewidth=1.8, label="F0 (Pitch)")
ax.plot(times, f1, "--", color="#00e676", linewidth=1.8, label="F1 (Tongue Height)")
ax.plot(times, f2, "--", color="#ff3b30", linewidth=1.8, label="F2 (Tongue Frontness)")

ax.set_ylim(0, 2200)
ax.set_yticks(np.arange(0, 2201, 500))
ax.set_title("Pitch (F0) & Formant Tracking (0–2200 Hz)")
ax.set_ylabel("Frequency (Hz)")
ax.set_xlabel("Time (s)")
ax.legend(loc="upper right")
ax.grid(True, color="white", alpha=0.2)

fig.colorbar(img, ax=ax, format="%+2.0f dB")
plt.tight_layout()
plt.show()