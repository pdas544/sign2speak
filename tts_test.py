import torch

def text_to_speech(text, output_path="outputs/output.wav"):
    try:
        # Import Coqui TTS
        from TTS.api import TTS

        # Get device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(TTS().list_models())  # List available models
        
        # Initialize TTS
        
        tts = TTS("tts_models/en/ljspeech/glow-tts",progress_bar=True).to(device)
        # tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2",progress_bar=True).to(device)
        
        # Generate speech
        tts.tts_to_file(text=text, file_path=output_path)
        
        print(f"Speech generated and saved to {output_path}")
        return output_path
        
    except ImportError:
        print("Coqui TTS not installed. Please install it with: pip install TTS")
        return None
    except Exception as e:
        print(f"Error in TTS: {e}")
        return None

tts_output = text_to_speech("beautiful")
if tts_output:
    print(f"Speech saved to {tts_output}")
else:
    print("TTS generation failed.")
# text_to_speech("Teacher")