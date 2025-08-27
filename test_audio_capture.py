#!/usr/bin/env python3
"""
Test audio capture directly to debug why speech isn't being detected.
"""

import pyaudio
import numpy as np
import time
import threading
from queue import Queue

# Audio parameters matching the config
DEVICE_ID = 16
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024
VAD_THRESHOLD = 0.05  # Updated threshold

class AudioDebugger:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.audio_buffer = Queue()
        self.is_recording = False
        self.capture_thread = None
        self.process_thread = None
        
    def start_capture(self):
        """Start audio capture."""
        print("🎤 Starting audio capture...")
        
        # Open stream
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=DEVICE_ID,
            frames_per_buffer=CHUNK_SIZE
        )
        
        self.is_recording = True
        
        # Start threads
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        
        self.capture_thread.start()
        self.process_thread.start()
        
        print("✅ Audio capture started")
    
    def _capture_loop(self):
        """Continuously capture audio."""
        print("📡 Capture thread started")
        chunk_count = 0
        
        while self.is_recording:
            try:
                # Read audio chunk
                data = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                chunk_count += 1
                
                # Convert to numpy
                audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                audio_data = audio_data / 32768.0
                
                # Put in buffer
                self.audio_buffer.put({
                    'data': audio_data,
                    'timestamp': time.time(),
                    'chunk_id': chunk_count
                })
                
                if chunk_count % 100 == 0:  # Every ~6 seconds
                    print(f"   📦 Captured {chunk_count} chunks")
                    
            except Exception as e:
                print(f"❌ Capture error: {e}")
    
    def _process_loop(self):
        """Process audio chunks."""
        print("🔄 Process thread started")
        speech_chunks = 0
        total_chunks = 0
        max_rms = 0
        
        while self.is_recording:
            try:
                # Get audio chunk (timeout after 0.1s)
                chunk = self.audio_buffer.get(timeout=0.1)
                audio_data = chunk['data']
                total_chunks += 1
                
                # Calculate RMS
                rms = np.sqrt(np.mean(audio_data**2))
                
                # Update max
                if rms > max_rms:
                    max_rms = rms
                    print(f"   📈 New max RMS: {max_rms:.4f}")
                
                # Check VAD threshold
                if rms > VAD_THRESHOLD:
                    speech_chunks += 1
                    print(f"🗣️  SPEECH DETECTED! Chunk #{chunk['chunk_id']}, RMS: {rms:.4f}")
                
                # Periodic status
                if total_chunks % 50 == 0:  # Every ~3 seconds
                    print(f"📊 Status: {total_chunks} chunks processed, {speech_chunks} with speech")
                    print(f"   Current RMS: {rms:.4f}, Max RMS: {max_rms:.4f}, Threshold: {VAD_THRESHOLD}")
                    
            except:
                pass  # Queue timeout is normal
    
    def run_test(self, duration=30):
        """Run the test for specified duration."""
        print(f"🚀 Running audio capture test for {duration} seconds")
        print(f"   Device: {DEVICE_ID}")
        print(f"   Sample rate: {SAMPLE_RATE}")
        print(f"   VAD threshold: {VAD_THRESHOLD}")
        print("=" * 60)
        
        self.start_capture()
        
        print("\n🎙️ PLEASE SPEAK NOW! Say something for testing...")
        print("-" * 60)
        
        # Wait for duration
        time.sleep(duration)
        
        # Stop
        print("\n🛑 Stopping capture...")
        self.is_recording = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=1)
        if self.process_thread:
            self.process_thread.join(timeout=1)
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        self.audio.terminate()
        print("✅ Test complete")

if __name__ == "__main__":
    debugger = AudioDebugger()
    debugger.run_test(30)