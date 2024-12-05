from pydub import AudioSegment
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from audio_transformer_model import AudioTransformer

def preprocess_audio(file_path, processor, model):
    """
    Preprocesses an audio file into logits for the classifier.

    Args:
        file_path (str): Path to the audio file.
        processor: Hugging Face processor (e.g., Wav2Vec2Processor).
        model: Pretrained feature extractor model (e.g., Wav2Vec2Model).

    Returns:
        torch.Tensor: Extracted features (logits) for the classifier.
    """
    # Load audio and resample
    audio = AudioSegment.from_file(file_path).set_frame_rate(16000).set_channels(1)
    audio_tensor = torch.tensor(audio.get_array_of_samples(), dtype=torch.float32)

    # Extract logits using the feature extraction model
    inputs = processor(audio_tensor, return_tensors="pt", sampling_rate=16000, padding=True)
    with torch.no_grad():
        features = model(**inputs).last_hidden_state  # Shape: (1, sequence_length, hidden_dim)
    
    return features.squeeze(0)  # Shape: (sequence_length, hidden_dim)

input_dim = 1024
hidden_dim = 256
num_heads = 8
num_layers = 4

classifier = AudioTransformer(input_dim=input_dim, hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers)
classifier.load_state_dict(torch.load("./drum_classifier.pt"))
classifier.eval()

# Load Wav2Vec2 feature extraction components
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
feature_extractor = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h")

# Preprocess the audio file
audio_file_path = "./test_audio/song_1.mp3"
logits = preprocess_audio(audio_file_path, processor, feature_extractor)  # Shape: (sequence_length, input_dim)

# Reshape logits for batch processing
logits = logits.unsqueeze(0)  # Add batch dimension: Shape: (1, sequence_length, input_dim)

# Perform inference
with torch.no_grad():
    output = classifier(logits).squeeze()  # Output is the probability of being a drum audio

# Interpret the result
print(f"Prediction: {output.item():.4f}")
if output.item() > 0.5:
    print("The audio file is classified as DRUMS.")
else:
    print("The audio file is classified as NOT DRUMS.")