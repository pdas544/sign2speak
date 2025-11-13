import pyttsx3
import os
from datetime import datetime
from deep_translator import GoogleTranslator

class TTSHelper:
    def __init__(self, audio_dir="audio", target_lang="hindi"):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Speed of speech
        self.engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
        
        # Initialize translator
        self.translator = GoogleTranslator(source='auto', target=target_lang)
        self.target_lang = target_lang
        
        # Create audio directory if it doesn't exist
        self.audio_dir = audio_dir
        os.makedirs(self.audio_dir, exist_ok=True)
        print(f"Audio files will be saved to: {os.path.abspath(self.audio_dir)}")
        
        # Cache for translations to avoid repeated API calls
        self.translation_cache = {}

    def translate_text(self, text):
        """Translate text to target language with caching"""
       
 
        try:
            print(f"Attempting to translate: {text}")
            translation = self.translator.translate(text)
            
            # if translation and hasattr(translation, 'text'):
            #     translated_text = translation.text
            #     self.translation_cache[text] = translated_text
            #     print(f"Translation successful: {text} → {translated_text}")
            #     return translated_text
            # else:
            #     print(f"Translation failed - Invalid response for: {text}")
            return translation
                
        except Exception as e:
            print(f"Google Translation error for '{text}': {str(e)}")
            

    def save_to_file(self, word):
        """Save speech to an audio file in both English and Hindi"""
        try:
            # Generate unique filename using timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save English audio
            en_filename = f"{word}_en_{timestamp}.mp3"
            en_filepath = os.path.join(self.audio_dir, en_filename)
            self.engine.save_to_file(word, en_filepath)
            self.engine.runAndWait()
            
            # Translate and save Hindi audio
            translated = self.translate_text(word)
            hi_filename = f"{word}_hi_{timestamp}.mp3"
            hi_filepath = os.path.join(self.audio_dir, hi_filename)
            self.engine.save_to_file(translated, hi_filepath)
            self.engine.runAndWait()
            
            print(f"Audio saved: {en_filepath} (English)")
            print(f"Audio saved: {hi_filepath} (Hindi)")
            return hi_filepath  # Return Hindi filepath as primary
        except Exception as e:
            print(f"TTS Error: {e}")
            return None

    def shutdown(self):
        """Clean shutdown of TTS engine"""
        self.engine.stop()
