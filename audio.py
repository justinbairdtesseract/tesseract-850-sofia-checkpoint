import numpy as np
import sounddevice as sd
from essentia.standard import OnsetRate, OnsetDetection, RhythmExtractor2013, FrameGenerator, RMS
from essentia.standard import Key, Windowing, Spectrum, FrameGenerator, FFT, CartesianToPolar
import threading
import time

fs = 44100  
blocksize = 1024  
hopSize = 512  
duration = 3 # how often tempo is calculated in seconds
# device = '1'

onset_detection = OnsetRate()
audio_buffer = np.array([], dtype=np.float32)
buffer_lock = threading.Lock()

def simple_peak_picking(onset_detection_function, threshold, min_distance):
    peaks = []
    last_peak = -min_distance
    for i, value in enumerate(onset_detection_function):
        if value > threshold and i - last_peak >= min_distance:
            peaks.append(i)
            last_peak = i
    return peaks


def detect_onsets():
    global audio_buffer
    while True:
        time.sleep(0.1)
        buffer_lock.acquire()
        if len(audio_buffer) > 0:
            onset_detection_function = []
            for frame in FrameGenerator(audio_buffer, frameSize=blocksize, hopSize=hopSize, startFromZero=True):
                windowed_frame = Windowing(type='hann')(frame)
                fft = FFT()(windowed_frame)  # Apply FFT to get the complex spectrum
                magnitude, phase = CartesianToPolar()(fft)  # Separate magnitude and phase
                onset_detection = OnsetDetection(method="hfc")
                odf = onset_detection(magnitude, phase)  # Now providing both magnitude and phase
                onset_detection_function.append(odf)
            # Once we have the full onset detection function, apply peak picking
            peaks = simple_peak_picking(onset_detection_function, threshold=0.5, min_distance=512)  # Adjust these values as needed
            for peak in peaks:
                print(f"Onset detected at frame {peak}")
                # Do something with detected onsets here
            audio_buffer = np.array([], dtype=np.float32)  # Clear the buffer after processing
        buffer_lock.release()


def estimate_audio_features():
    global audio_buffer
    while True:
        time.sleep(duration)  # Wait for the specified duration
        buffer_lock.acquire()
        if len(audio_buffer) > 0:
            # Tempo estimation
            rhythm_extractor = RhythmExtractor2013(method="multifeature")
            results = rhythm_extractor(audio_buffer)
            tempo = results[0] 
            if tempo > 0:
                print(f"Estimated tempo: {tempo} BPM")
            else:
                print("No tempo detected.")
            
            # Volume estimation
            rms = RMS()(audio_buffer)
            print(f"Volume (RMS level): {rms}")

            # Clear the buffer after processing
            audio_buffer = np.array([], dtype=np.float32)
        else:
            print("No audio data for analysis.")
        buffer_lock.release()
        
        

def audio_callback(indata, frames, time, status):
    global audio_buffer
    if status:
        print(status)

    audio_frame = np.mean(indata, axis=1) if indata.shape[1] > 1 else np.squeeze(indata)

    buffer_lock.acquire()
    audio_buffer = np.concatenate((audio_buffer, audio_frame))
    buffer_lock.release()

audio_features_thread = threading.Thread(target=estimate_audio_features)
audio_features_thread.start()

onset_thread = threading.Thread(target=detect_onsets)
onset_thread.start()


with sd.InputStream(callback=audio_callback, samplerate=fs, channels=1, blocksize=blocksize):
    print("Streaming started... Press Ctrl+C to stop")
    try:
        while True:  # Keep the stream alive
            pass
    except KeyboardInterrupt:
        print("Streaming stopped")