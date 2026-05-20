# VoicePrint-Python

A professional audio processing and speaker verification system using ECAPA-TDNN embeddings and Mel-spectrogram analysis.

## 📋 Overview

This project provides a complete pipeline for speaker verification, combining traditional audio feature extraction with state-of-the-art deep learning embeddings. The system extracts speaker-specific features from audio samples and verifies speaker identity through cosine similarity comparison.

## ✨ Features

- **Professional Audio Processing**: Configurable feature extraction with Mel-spectrograms
- **Speaker Embedding Extraction**: ECAPA-TDNN model pre-trained on VoxCeleb
- **Real-time Verification**: Cosine similarity-based speaker comparison
- **Visualization Tools**: Waveform and Mel-spectrogram plotting
- **Pre-processing Pipeline**: Resampling, mono conversion, normalization, and pre-emphasis

## 🏗️ Architecture

### 1. Audio Processor (`ProfessionalAudioProcessor`)
- Frame length: 25ms (400 samples at 16kHz)
- Frame step: 10ms (160 samples)
- FFT points: 512 (31.25 Hz resolution)
- Mel filters: 80 bands
- Window type: Hamming
- Output: Log-Mel Spectrograms

### 2. Speaker Embedding (`ECAPA-TDNN`)
- Pre-trained on VoxCeleb corpus
- 192-dimensional speaker embeddings
- Robust to channel and recording variations

## 🚀 Quick Start

### Prerequisites


```bash
pip install torch torchaudio speechbrain matplotlib




# Initialize processor
processor = ProfessionalAudioProcessor()

# Extract features
features = processor.extract_features("audio.wav")

# Load speaker embedding model
from speechbrain.pretrained import EncoderClassifier
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb"
)

# Extract embeddings
embedding = extract_speaker_embedding("audio.wav", classifier)

# Verify speakers
emb1 = extract_speaker_embedding("speaker1.wav", classifier)
emb2 = extract_speaker_embedding("speaker2.wav", classifier)
similarity = torch.nn.functional.cosine_similarity(emb1, emb2, dim=0)
