#!/usr/bin/env python3
import os
from gtts import gTTS  # Text-to-Speech
import speech_recognition as sr  # Speech Recognition

class SpeechProcessor:
    def __init__(self):
        self.audio_path = "/home/mustar/catkin_ws/src/cr_receptionist/assets/main_audio.mp3"
        self.recognizer = sr.Recognizer()

    def text2audio(self, text):
        """Converts given text to audio and plays it."""
        tts = gTTS(text)
        tts.save(self.audio_path)  # Save the audio file
        print(f"Audio saved at {self.audio_path}")
        os.system(f"mpg321 {self.audio_path}")  # Play the audio
        os.remove(self.audio_path)  # Remove the audio file after playing
        print("Audio file removed after playback.")

    def audio2text(self):
        """Records audio from the microphone and converts it to text."""
        result = ""
        try:
            with sr.Microphone() as source:
                print(">>> Say something!")
                audio = self.recognizer.record(source, duration=3)  # Record for 3 seconds
            
            # Recognize speech using Google's Speech Recognition API
            result = self.recognizer.recognize_google(audio).lower()
            print("SR result: " + result)
        
        except sr.UnknownValueError:
            print("SR could not understand audio.")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")

        return result

# Example usage
if __name__ == "__main__":
    speech_processor = SpeechProcessor()

    # Example: Convert text to audio and play it
    speech_processor.text2audio("Hello, how are you?")

    # Example: Convert audio input to text
    spoken_text = speech_processor.audio2text()
    print(f"You said: {spoken_text}")
