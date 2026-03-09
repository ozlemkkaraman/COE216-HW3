import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf

# load and normalize the audio
audio_file = "speech.wav"
signal, fs = librosa.load(audio_file, sr=None)
signal = signal / np.max(np.abs(signal))  # normalize [-1,1]
signal = signal - np.mean(signal)

# frame parameters
frame_size = int(0.020 * fs)  # 20 ms
hop_size = frame_size // 2     # 50% overlap
window = np.hamming(frame_size)


# Noise Threshold Estimation
noise_duration = 2 
noise_segment = signal[:int(noise_duration*fs)]
noise_energy = np.mean([np.sum((noise_segment[i:i+frame_size]*window)**2)
                        for i in range(0, len(noise_segment)-frame_size, hop_size)])
threshold = noise_energy * 1.5 
print("Noise Threshold:", threshold)

# Main Loop: VAD + Voiced/Unvoiced
energy_values = []
zcr_values = []
classification = []  # 0: Silence, 1: Unvoiced, 2: Voiced

hangover_limit = 4
hangover_counter = 0
zcr_threshold = 0.15

for i in range(0, len(signal)-frame_size, hop_size):
    frame = signal[i:i+frame_size] * window
    energy = np.sum(frame**2)
    zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2*len(frame))
    
    energy_values.append(energy)
    zcr_values.append(zcr)
    
    is_speech_now = 1 if energy > threshold else 0
    if is_speech_now:
        is_speech_final = 1
        hangover_counter = hangover_limit
    else:
        if hangover_counter > 0:
            is_speech_final = 1
            hangover_counter -= 1
        else:
            is_speech_final = 0
    
    if is_speech_final:
        classification.append(2 if zcr < zcr_threshold else 1)
    else:
        classification.append(0)


# Speech-only reconstruction
speech_mask = np.zeros(len(signal), dtype=bool)
for idx, label in enumerate(classification):
    if label != 0:
        start = idx * hop_size
        end = min(start + frame_size, len(signal))
        speech_mask[start:end] = True

speech_signal = signal[speech_mask]
sf.write("speech_only.wav", speech_signal, fs)

# Compression Report
orig_dur = len(signal) / fs
new_dur = len(speech_signal) / fs
compression = (1 - new_dur / orig_dur) * 100
print(f"Original Duration: {orig_dur:.2f}s")
print(f"Speech-only Duration: {new_dur:.2f}s")
print(f"Compression: {compression:.2f}%")


# Visualization
time_signal = np.arange(len(signal)) / fs
time_frames = np.arange(len(energy_values)) * (hop_size/fs)

plt.figure(figsize=(15,12))

# Original Signal
plt.subplot(3,1,1)
plt.plot(time_signal, signal, color='darkblue')
plt.title("Original Audio Signal")
plt.ylabel("Amplitude")

# Energy & ZCR
plt.subplot(3,1,2)
plt.plot(time_frames, energy_values, label="Energy", color='red')
plt.plot(time_frames, zcr_values, label="ZCR", color='blue')
plt.axhline(threshold, color='green', linestyle='--', label="Noise Threshold")
plt.title("Energy and ZCR")
plt.legend()

# VAD + Voiced/Unvoiced
plt.subplot(3,1,3)
plt.plot(time_signal, signal, color='gray', alpha=0.3)
for idx, label in enumerate(classification):
    start = idx * hop_size
    end = min(start + frame_size, len(signal))
    if label == 2:
        plt.axvspan(start/fs, end/fs, color='orange', alpha=0.3)
    elif label == 1:
        plt.axvspan(start/fs, end/fs, color='green', alpha=0.3)
plt.title("Voiced (Orange) vs Unvoiced (Green)")
plt.tight_layout()

plt.show()
