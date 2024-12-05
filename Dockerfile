# Use NVIDIA PyTorch as base image
FROM nvcr.io/nvidia/pytorch:23.08-py3

# Set a working directory
WORKDIR /app

# Install necessary libraries
RUN apt-get update && apt-get install -y \
    ffmpeg \
    sox \
    git \
    vim \
    && apt-get clean

# Install Python dependencies
RUN pip install --no-cache-dir \
    transformers \
    datasets \
    torch torchaudio \
    pydub

# Copy the application code into the container
COPY . /app

# Default command
CMD ["python", "app.py"]
