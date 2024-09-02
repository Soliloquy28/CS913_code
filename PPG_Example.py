import numpy as np
import matplotlib.pyplot as plt
import os

# Read data from txt file
data = np.loadtxt('/dcs/large/u2212061/ppg_34_zero/mesa-sleep-0001.txt')

# 30s = 1024 signals
ppg_signal = data[:1025]

sampling_rate = 34.13  # Hz
duration = len(ppg_signal) / sampling_rate
time = np.linspace(0, duration, len(ppg_signal))

plt.figure(figsize=(12, 3))
plt.plot(time, ppg_signal)
plt.title(f'PPG Signal ({sampling_rate}Hz)')
plt.xlabel('Time (seconds)')
plt.ylabel('PPG Amplitude')
plt.grid(True)
plt.xlim(0, duration)
xticks = np.arange(0, duration + 1, 1)
plt.xticks(xticks)

file_path = f'./ppg_signal_plot_{sampling_rate}Hz.png'
plt.savefig(file_path, dpi=300, bbox_inches='tight')

print(f"Plot saved as {file_path}")
