from deep_translator import GoogleTranslator
from tts_helper import TTSHelper
import time

def test_translation_and_tts():
    # Test words that might appear in sign language detection
    test_words = [
        "hello",
        "goodbye"
    ]

    print("Initializing TTS Helper...")
    tts = TTSHelper()

    print("\nTesting translation and audio generation:")
    print("-" * 50)

    for word in test_words:
        print(f"\nTesting word: '{word}'")

        # Use deep-translator for translation (to French for example)
        try:
            translated = GoogleTranslator(source='auto', target='hindi').translate(word)
        except Exception as e:
            print(f"Translation failed for '{word}' with error: {e}")
            translated = word

        if translated != word:
            print(f"Translation successful: {word} → {translated}")
        else:
            print(f"Translation might have failed for: {word}")

        # Then test audio generation
        audio_path = tts.save_to_file(word, translated)
        if audio_path:
            print(f"Audio files generated successfully")

        # Small delay to avoid any rate limiting
        time.sleep(1)

    print("\nTest complete!")

if __name__ == "__main__":
    try:
        langs_list = GoogleTranslator().get_supported_languages()
        print(f"Supported languages: {langs_list}")
        test_translation_and_tts()
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
