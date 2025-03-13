import streamlit as st
import os
import wave
import numpy as np
import pyaudio
import speech_recognition as sr
import time
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load advanced translation model
model_name = "facebook/nllb-200-3.3B"
try:
    translation_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    translation_tokenizer = AutoTokenizer.from_pretrained(model_name)
except Exception as e:
    st.error(f"\u274C Translation model loading error: {e}")

# Define audio parameters
SAMPLE_RATE = 44100  
CHANNELS = 1
DURATION = 80 
AUDIO_FILE = "long_audio.wav"

def record_audio(filename=AUDIO_FILE, duration=DURATION, sample_rate=SAMPLE_RATE):
    """Records audio using pyaudio and saves as WAV file."""
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=CHANNELS, 
                    rate=sample_rate, input=True, 
                    frames_per_buffer=1024)
    
    st.write("\U0001F3A4 **Recording... Speak in Malayalam.** (Max: 5 minutes)")
    frames = []
    for _ in range(0, int(sample_rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
    
    return filename

def transcribe_audio(audio_file):
    """Converts speech from Malayalam to text using Google Speech Recognition."""
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.record(source)
        
        start_time = time.time()  # Start timer
        transcript = recognizer.recognize_google(audio, language='ml-IN')
        end_time = time.time()  # End timer

        transcription_time = round(end_time - start_time, 2)  # Compute time
        return transcript, transcription_time
    
    except sr.UnknownValueError:
        return "⚠️ Could not understand the audio.", 0
    except sr.RequestError as e:
        return f"\u274C Google Speech Recognition error: {e}", 0
    except Exception as e:
        return f"\u274C Audio processing error: {e}", 0

def translate_malayalam_to_english(text):
    """Improved Malayalam-to-English translation with better preprocessing and generation."""
    if not text:
        return "⚠️ No transcription available for translation.", 0
    
    try:
        # Normalize and split long sentences
        text = text.strip().replace("\n", " ")
        sentences = text.split(". ")
        
        start_time = time.time()  # Start timer
        translated_sentences = []
        for sentence in sentences:
            inputs = translation_tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
            translated = translation_model.generate(
                **inputs, num_beams=7, length_penalty=1.2, early_stopping=True
            )
            translated_text = translation_tokenizer.decode(translated[0], skip_special_tokens=True)
            translated_sentences.append(translated_text)

        end_time = time.time()  # End timer
        translation_time = round(end_time - start_time, 2)  # Compute time

        return " ".join(translated_sentences), translation_time
    
    except Exception as e:
        return f"\u274C Translation error: {e}", 0

def main():
    st.title("\U0001F50A Speech-to-Text & Translation")
    st.write("\U0001F3A7 Speak in Malayalam, and the app will transcribe and translate it.")

    if st.button("\U0001F680 Start Recording"):
        audio_file = record_audio()
        if audio_file:
            st.write("\U0001F504 **Transcribing...**")
            transcript, transcription_time = transcribe_audio(audio_file)
            st.write("\U0001F4DC **Transcribed Text (Malayalam):**", transcript)
            st.write(f"⏳ **Transcription Time:** {transcription_time} sec")

            st.write("\U0001F504 **Translating to English...**")
            translated_text, translation_time = translate_malayalam_to_english(transcript)
            st.write("\U0001F4DC **Translated Text (English):**", translated_text)
            st.write(f"⏳ **Translation Time:** {translation_time} sec")

            st.success("\u2705 Translation completed!")
        
        # Remove the recorded file after processing
        if os.path.exists(AUDIO_FILE):
            os.remove(AUDIO_FILE)

if __name__ == "__main__":
    main()
