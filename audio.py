import numpy as np
import sounddevice as sd
from essentia.standard import OnsetRate, RhythmExtractor2013, FrameGenerator
import threading
import time

fs = 44100  
blocksize = 1024  
hopSize = 512  
duration = 3 # how often tempo is calculated in seconds
device = 6

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

def audio_callback(indata, frames, time, status):
    global audio_buffer
    if status:
        print(status)
    
    audio_frame = np.mean(indata, axis=1) if indata.shape[1] > 1 else np.squeeze(indata)

    buffer_lock.acquire()
    audio_buffer = np.concatenate((audio_buffer, audio_frame))
    buffer_lock.release()

    local_onsets_detected = False
    for frame in FrameGenerator(audio_frame, frameSize=blocksize, hopSize=hopSize, startFromZero=True):
        onsets, rate = onset_detection(frame)
        if len(onsets) > 0:
            print("Onset detected")
            local_onsets_detected = True

        
tempo_thread = threading.Thread(target=estimate_tempo)
tempo_thread.start()


with sd.InputStream(callback=audio_callback, samplerate=fs, channels=1, blocksize=blocksize, device=device):
    print("Streaming started... Press Ctrl+C to stop")
    try:
        while True:  # Keep the stream alive
            pass
    except KeyboardInterrupt:
        print("Streaming stopped")