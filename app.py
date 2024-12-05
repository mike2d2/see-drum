import os
import random
import shutil
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pydub import AudioSegment
import numpy as np
import torchaudio

from audio_transformer_model import AudioTransformer
from logits_dataset import LogitsDataset

# Paths
DATASET_PATH = "./datasets"
DRUM_AUDIO_PATH = os.path.join(DATASET_PATH, "drum_audio")
ALL_AUDIO_PATH = os.path.join(DATASET_PATH, "all_audio")
LOGITS_PATH = os.path.join(DATASET_PATH, "logits")
TRAIN_LOGITS_PATH = os.path.join(LOGITS_PATH, "train")
TEST_LOGITS_PATH = os.path.join(LOGITS_PATH, "test")

TRAIN_DATA_PATH = os.path.join(DATASET_PATH, "musdb18hq", "train")
TEST_DATA_PATH = os.path.join(DATASET_PATH, "musdb18hq", "test")

# Create folders if they don't exist
os.makedirs(DRUM_AUDIO_PATH, exist_ok=True)
os.makedirs(ALL_AUDIO_PATH, exist_ok=True)
os.makedirs(TRAIN_LOGITS_PATH, exist_ok=True)
os.makedirs(TEST_LOGITS_PATH, exist_ok=True)

# Load Wav2Vec2 Model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
model.eval()  # Set model to evaluation mode


def embed_audio_to_encoder_output(file_path):
    """Extract encoder output for a given audio file."""
    # Load and preprocess the audio file
    audio = AudioSegment.from_file(file_path).set_frame_rate(16000).set_channels(1)
    audio_tensor = torch.tensor(audio.get_array_of_samples(), dtype=torch.float32)
    inputs = processor(audio_tensor, return_tensors="pt", sampling_rate=16000, padding=True)
    
    with torch.no_grad():
        # Forward pass with `output_hidden_states=True`
        outputs = model(**inputs, output_hidden_states=True)
    
    # Encoder output is the last hidden state
    encoder_output = outputs.hidden_states[-1]  # Shape: (batch_size, sequence_length, hidden_dim)
    
    return encoder_output


def process_datasets(segment_duration=5, num_segments=5, split="train"):
    """
    Process datasets to extract segments, save logits, and create x and y tensors.
    """
    data_path = TRAIN_DATA_PATH if split == "train" else TEST_DATA_PATH
    logits_path = TRAIN_LOGITS_PATH if split == "train" else TEST_LOGITS_PATH

    x = []  # To store paths to saved logits
    y = []  # To store corresponding labels

    count = 0
    for root, _, files in tqdm(os.walk(data_path)):
        if count > 300:
            break

        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)

                # Load the audio file
                audio = AudioSegment.from_file(file_path)

                # Break audio into 5-second segments
                segment_length = segment_duration * 1000  # Convert seconds to milliseconds
                segments = [
                    audio[i:i + segment_length] for i in range(0, len(audio), segment_length)
                ]

                # Filter segments with audible levels
                def is_audible(segment, threshold=-40):
                    # Check if the segment's average loudness exceeds the threshold (in dBFS)
                    return segment.dBFS > threshold

                audible_segments = [seg for seg in segments if is_audible(seg)]

                # Limit to N segments
                selected_segments = audible_segments[:num_segments]

                # Skip processing if no audible segments
                if not selected_segments:
                    print(f"No audible segments found in {file}. Skipping.")
                    continue

                # Process each selected segment individually
                for idx, segment in enumerate(selected_segments):
                    # Save the segment to a temporary file for feature extraction
                    temp_file_path = f"./tmp/temp_segment_{idx}.wav"
                    segment.export(temp_file_path, format="wav")

                    # Save logits for this segment
                    segment_filename = f"{os.path.splitext(file)[0]}_segment_{idx}.npy"
                    dest_path = os.path.join(logits_path, segment_filename)
                    x_logits = embed_audio_to_encoder_output(temp_file_path)

                    x_logits = x_logits.squeeze(0) # Removes batch of 1 dimension

                    # Determine label based on filename
                    label = 1 if any(keyword in file.lower() for keyword in ["drum", "drums", "percussion"]) else 0

                    # Append to x and y
                    x.append(x_logits)
                    y.append(label)

                    count+=1

    x = np.array(x)  # Shape: (num_samples, 249, 1024)
    y = np.array(y)  # Shape: (num_samples,)


    np.save(f"{logits_path}/logits.npy", x)
    np.save(f"{logits_path}/labels.npy", y)

def train_and_test_model():
    """Train and test the model using precomputed logits."""
    # Collect training data

    # Labels: drum (1), non-drum (0)
    train_logits_path = os.path.join(TRAIN_LOGITS_PATH, "logits.npy")
    train_labels_path = os.path.join(TRAIN_LOGITS_PATH, "labels.npy")
    test_logits_path = os.path.join(TEST_LOGITS_PATH, "logits.npy")
    test_labels_path = os.path.join(TEST_LOGITS_PATH, "labels.npy")

    # Create datasets and dataloaders
    train_dataset = LogitsDataset(train_logits_path, train_labels_path)
    test_dataset = LogitsDataset(test_logits_path, test_labels_path)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # # Initialize model
    # class DrumClassifier(torch.nn.Module):
    #     def __init__(self):
    #         super(DrumClassifier, self).__init__()
    #         self.fc = torch.nn.Sequential(
    #             torch.nn.Linear(1024, 256),
    #             torch.nn.ReLU(),
    #             torch.nn.Linear(256, 1),
    #             torch.nn.Sigmoid()
    #         )

    #     def forward(self, x):
    #         return self.fc(x)

    batch_size = 16
    sequence_length = 249
    input_dim = 1024  # Hidden dimension of logits
    hidden_dim = 256  # Reduced dimensionality
    num_heads = 8
    num_layers = 4
    classifier = AudioTransformer(input_dim=input_dim, hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)
    criterion = torch.nn.BCELoss()

    # Training loop
    for epoch in range(10):  # 10 epochs
        classifier.train()
        total_loss = 0
        for logits, labels in train_loader:
            optimizer.zero_grad()

            # Forward pass
            outputs = classifier(logits).squeeze()
            labels = labels.float()
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

    # Testing loop
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for logits, labels in test_loader:
            outputs = classifier(logits).squeeze()
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    print(f"Test Accuracy: {correct / total * 100:.2f}%")

    # Save the model
    torch.save(classifier.state_dict(), "./drum_classifier.pt")
    print("Model training and testing completed. Model saved!")


if __name__ == "__main__":
    print("Processing training data...")
    process_datasets(split="train")
    print("Processing testing data...")
    process_datasets(split="test")
    print("Training and testing model...")
    train_and_test_model()