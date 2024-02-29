# python3 -m sounddevice

# TEST AUDIO RECORDING

import sounddevice as sd
import wave

fs = 44100  # Sample rate
duration = 5  # seconds
filename = 'output.wav'
device = 6

print("Recording...")
myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16', device=device)
sd.wait()  # Wait until recording is finished
print("Finished recording.")

# Save as WAV file
with wave.open(filename, 'wb') as wf:
    wf.setnchannels(1)  # Mono
    wf.setsampwidth(2)  # Sample width in bytes, int16 = 2 bytes
    wf.setframerate(fs)
    wf.writeframes(myrecording.tobytes())