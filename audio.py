import numpy as np
import sounddevice as sd
from essentia.standard import OnsetRate, RhythmExtractor2013, FrameGenerator
import threading
import time

fs = 44100  
blocksize = 1024  
hopSize = 512  
duration = 3 # how often tempo is calculated in seconds
device = 'Scarlett 4i4 USB'

onset_detection = OnsetRate()
audio_buffer = np.array([], dtype=np.float32)
buffer_lock = threading.Lock()

def estimate_tempo():
    global audio_buffer
    while True:
        time.sleep(duration)
        buffer_lock.acquire()
        if len(audio_buffer) > 0:
            rhythm_extractor = RhythmExtractor2013(method="multifeature")
            results = rhythm_extractor(audio_buffer)
            tempo = results[0] 
            if tempo > 0:
                print(f"Estimated tempo: {tempo} BPM")
            else:
                print("No tempo detected.")
            audio_buffer = np.array([], dtype=np.float32)
        else:
            print("No audio data for tempo estimation.")
        buffer_lock.release()


def detect_onsets():
    global audio_buffer
    while True:
        time.sleep(0.1) 
        buffer_lock.acquire()
        if len(audio_buffer) > 0:
            # Prepare the audio buffer for onset detection
            for frame in FrameGenerator(audio_buffer, frameSize=blocksize, hopSize=hopSize, startFromZero=True):
                windowed_frame = Windowing(type='hann')(frame)
                spectrum = Spectrum()(windowed_frame)
                onset_detection = OnsetDetection(method="hfc")
                onsets = onset_detection(spectrum)
                # Trigger your action here if onsets are detected
                if onsets:
                    print("Onset detected")
                    # Trigger your action here
            audio_buffer = np.array([], dtype=np.float32)  # Clear the buffer after processing
        buffer_lock.release()


def estimate_key():
    global audio_buffer
    while True:
        time.sleep(duration)  # Use the same duration as tempo estimation or adjust based on your needs
        buffer_lock.acquire()
        if len(audio_buffer) > 0:
            # Pre-processing steps: Windowing and Spectrum calculation
            # Accumulate spectral frames for key estimation
            spectral_frames = []
            for frame in FrameGenerator(audio_buffer, frameSize=blocksize, hopSize=hopSize, startFromZero=True):
                windowed_frame = Windowing(type='hann')(frame)
                spectrum_frame = Spectrum()(windowed_frame)
                spectral_frames.append(spectrum_frame)
            
            if spectral_frames:
                # Concatenate spectral frames to form the basis for key estimation
                spectral_data = np.vstack(spectral_frames)
                
                # Estimate key
                key_estimator = Key(profileType='edma', # Use 'edma' or 'bgate' depending on your preference
                                    usePolyphony=True,
                                    useThreeChords=True)
                key, scale, strength = key_estimator(spectral_data)
                
                if key and scale:
                    print(f"Estimated key: {key} {scale}, strength: {strength}")
                else:
                    print("Key estimation was unsuccessful.")
            else:
                print("Not enough audio data for key estimation.")
                
            audio_buffer = np.array([], dtype=np.float32)  # Clear the buffer after processing
        else:
            print("No audio data for key estimation.")
        buffer_lock.release()

def estimate_volume():
    global audio_buffer
    while True:
        time.sleep(0.1)  # Adjust based on your preference for how often to update volume info
        buffer_lock.acquire()
        if len(audio_buffer) > 0:
            rms = RMS()(audio_buffer)
            print(f"Volume (RMS level): {rms}")
            audio_buffer = np.array([], dtype=np.float32)  # Optionally clear the buffer after processing
        else:
            print("No audio data for volume estimation.")
        buffer_lock.release()


def audio_callback(indata, frames, time, status):
    global audio_buffer
    if status:
        print(status)

    audio_frame = np.mean(indata, axis=1) if indata.shape[1] > 1 else np.squeeze(indata)

    buffer_lock.acquire()
    audio_buffer = np.concatenate((audio_buffer, audio_frame))
    buffer_lock.release()


        
tempo_thread = threading.Thread(target=estimate_tempo)
tempo_thread.start()
onset_thread = threading.Thread(target=detect_onsets)
onset_thread.start()
key_thread = threading.Thread(target=estimate_key)
key_thread.start()
volume_thread = threading.Thread(target=estimate_volume)
volume_thread.start()


with sd.InputStream(callback=audio_callback, samplerate=fs, channels=1, blocksize=blocksize, device=device):
    print("Streaming started... Press Ctrl+C to stop")
    try:
        while True:  # Keep the stream alive
            pass
    except KeyboardInterrupt:
        print("Streaming stopped")