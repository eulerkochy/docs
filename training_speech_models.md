
# Training Speech Models like Kokoro and Zonos for Indic Languages

This document outlines the process of training and fine-tuning speech models for Indic languages. While Kokoro and Zonos are designed as text-to-speech (TTS) systems, many of the principles outlined here also apply to speech-to-text (STT) tasks. In both cases, data quality, compute requirements, phonemization, model architecture, and training infrastructure play a critical role in achieving high performance.

---

## Table of Contents

1. [Overview](#overview)
2. [Data Requirements](#data-requirements)
   - [Quantity](#quantity)
   - [Quality](#quality)
   - [Data Sources for Indic Languages](#data-sources-for-indic-languages)
3. [Phonemization](#phonemization)
4. [Sampling Rate](#sampling-rate)
5. [Underlying Model Architecture](#underlying-model-architecture)
   - [For TTS Systems](#for-tts-systems)
   - [For STT Systems](#for-stt-systems)
6. [Training Infrastructure and Compute Requirements](#training-infrastructure-and-compute-requirements)
   - [Training from Scratch](#training-from-scratch)
   - [Fine-Tuning](#fine-tuning)
7. [Specific Considerations for Indic Languages](#specific-considerations-for-indic-languages)
8. [Example Code Snippets](#example-code-snippets)
9. [Conclusion](#conclusion)
10. [References](#references)

---

## Overview

Training high‑quality speech models from scratch is a resource‑intensive undertaking. For many Indic languages, fine‑tuning an existing pre‑trained model on domain‑specific data is often more practical. This guide discusses both approaches, covering:

- **Data Requirements:** The volume and quality of audio and transcription data.
- **Phonemization:** Converting text into phonemes using tools such as eSpeak‑NG or Indic‑specific phonemizers.
- **Sampling Rate:** Ensuring consistency in audio quality (commonly 16kHz for speech recognition, while TTS models might use 24kHz or 44kHz).
- **Model Architecture:** Options range from lightweight transformer‑based models (e.g., Wav2Vec 2.0, Conformer) to end‑to‑end systems.
- **Training Infrastructure:** Hardware (GPUs/TPUs, distributed training), software frameworks, and orchestration tools.
- **Language‑Specific Considerations:** Addressing tonal variations, complex morphology, code‑switching, and script variations common in Indic languages.

---

## 1. Data Requirements

### 1.1. Quantity

- **Training from Scratch:**  
  - **TTS:** Robust models may require *thousands* of hours of high‑quality transcribed audio (e.g., 1,000–10,000+ hours).  
  - **STT:** Similarly, large vocabulary or multi‑speaker systems benefit from extensive datasets.
- **Fine-Tuning:**  
  - With a good pre‑trained model (e.g., one trained on large English datasets), 100–500 hours of data in your target Indic language can yield decent performance. In some cases, even 10–50 hours may show improvement, though with limited results.

### 1.2. Quality

- **Clean Audio:** Ensure minimal background noise, echo, or distortion.  
- **Accurate Transcriptions:** Transcripts must be meticulously correct—including punctuation, handling of numbers, abbreviations, and special characters—to avoid teaching incorrect mappings.
- **Speaker Diversity:** Include voices from various genders, ages, and accents to improve generalization.
- **Domain Coverage:** Match your data to the intended application (e.g., conversational, broadcast, academic).
- **Balanced Phoneme Distribution:** A well-represented set of phonemes ensures robust performance.

### 1.3. Data Sources for Indic Languages

- **Common Voice (Mozilla):** Crowdsourced audio available in several Indic languages.
- **Linguistic Data Consortium for Indian Languages (LDC-IL):** A resource that may require membership or fees.
- **Government Initiatives & Academic Institutions:** Often release high-quality datasets.
- **Commissioned Recordings:** Professional recordings offer high control over quality.
- **Web Scraping (with caution):** Audio from podcasts or broadcasts (ensure legal rights).
- **OpenSLR:** Open‑source speech and language resources.

---

## 2. Phonemization

A phonemizer converts text into a sequence of phonemes (the basic sound units), which is essential for accurate speech synthesis and recognition.

- **eSpeak‑NG:**  
  - Widely used and supports many Indic languages.  
  - May be combined with fallback rules for languages where accuracy varies.
- **Indic NLP Library:**  
  - Offers phonemization tools specifically designed for Indic scripts.
- **Custom Phonemizers:**  
  - Consider developing a tailored G2P (grapheme-to‑phoneme) module if dealing with complex or tonal languages.
- **Character-Based Models:**  
  - Some modern end‑to‑end systems operate directly on characters, simplifying the pipeline at the cost of a more challenging learning task.

---

## 3. Sampling Rate

The sampling rate defines how many times per second the audio is captured:

- **16kHz:**  
  - Common for STT systems; it strikes a balance between quality and computational efficiency.
- **24kHz – 44kHz:**  
  - Often used in TTS models (e.g., Kokoro may use 24kHz; Zonos outputs at 44kHz).  
  - Higher sampling rates capture more detail but require more compute and storage.
- **8kHz:**  
  - Lower quality; suited to limited-bandwidth applications.

**Recommendation:** Use 16kHz for most speech recognition tasks unless higher fidelity is required, and maintain consistency across your dataset and model.

---

## 4. Underlying Model Architecture

Choosing the right model architecture depends on your task (TTS vs. STT) and available resources.

### For TTS Systems

- **Lightweight Architectures (e.g., Kokoro‑like):**  
  - Often have around 82M parameters and are optimized for speed and cost‑efficiency.
- **Transformer/Hybrid Architectures (e.g., Zonos‑like):**  
  - Leverage transformers (or hybrid CNN–transformer models) for fine‑grained control over prosody, emotion, and quality.
- **Key Components:**  
  - **Text Processing:** Normalization, phonemization, and embedding layers.
  - **Acoustic Model:** Maps phoneme embeddings to spectrograms or intermediate representations.
  - **Vocoder:** Converts spectrograms into waveforms (e.g., HiFi-GAN, WaveGlow).

### For STT Systems
- **Wav2Vec 2.0:**   
	- A popular self-supervised learning framework for speech representation learning. It learns from raw audio without transcriptions, making it effective for pre-training. Kokoro is based on Wav2Vec2.
- **Conformer:**  
	- Combines convolutional neural networks (CNNs) and transformers. CNNs are good at capturing local patterns in the audio, while transformers excel at modeling long-range dependencies. This makes conformers very powerful for STT.
- **Transformer:** 
	- A purely attention-based architecture. Transformers have been very successful in  natural  language processing and  are also used in STT. 
-   **CTC (Connectionist Temporal Classification):** 
	- A loss function commonly used in STT. It allows training on unsegmented audio data, where you don't know the precise alignment between the audio and the transcript. 
-  **End-to-End Models:** 
	- These models directly map audio input to character or word output, without separate acoustic and language models. Wav2Vec 2.0 and conformer-based models are often trained end-to-end.

---

## 5. Training Infrastructure and Compute Requirements

### 5.1. Training from Scratch

- **Hardware:**  
  - **GPUs:** Multiple high‑end GPUs (e.g., NVIDIA A100, V100) with 40GB–80GB VRAM are typically needed.  
  - **TPUs:** Google’s TPUs may offer higher efficiency for large‑scale training.
- **Compute Time:**  
  - Training from scratch on thousands of hours of data can take weeks or months, even with a distributed setup.

### 5.2. Fine-Tuning

- **Hardware:**  
  - A single high‑end GPU (or even a mid‑range GPU with 12–24GB VRAM) may suffice.
- **Compute Time:**  
  - Fine‑tuning on 10–500 hours of data can take from a few hours to several days.
- **Software Frameworks:**  
  - PyTorch and TensorFlow are popular, with tools like Hugging Face Transformers and fairseq to ease development.

### 5.3. Distributed Training & Orchestration

- **Distributed Training:**  
  - Use PyTorch’s Distributed Data Parallel (DDP) or TensorFlow’s distribution strategies for scaling across multiple GPUs.
- **Containerization:**  
  - Docker and container orchestration (e.g., Kubernetes) ensure reproducible environments.
- **Monitoring:**  
  - Tools such as TensorBoard and Weights & Biases help track metrics during training.

---

## 6. Specific Considerations for Indic Languages

- **Tonal Nuances:**  
  - Some Indic languages have tonal elements; models must capture subtle pitch variations.
- **Complex Morphology:**  
  - Rich word formation and inflections necessitate careful design of language models.
- **Code-Switching:**  
  - In regions where multiple languages are mixed (e.g., Hindi-English), ensure the dataset reflects this phenomenon.
- **Script Variations:**  
  - Some languages have multiple scripts; maintain consistency across your dataset.

---

## 7. Example Code Snippets

### 7.1. Phonemization Example

Below is an example using `eSpeak-NG` for phonemizing a Hindi sentence:

```python
import subprocess

def phonemize(text, lang='hi'):
    cmd = ['espeak-ng', '-q', '--ipa', '-v', lang, text]
    ipa_output = subprocess.check_output(cmd).decode('utf-8').strip()
    return ipa_output

sample_text = "नमस्ते, आप कैसे हैं?"
phonemes = phonemize(sample_text, lang='hi')
print("Phoneme sequence:", phonemes)
```

### 7.2 Model training skeleton
Below is a simplified PyTorch training loop for an acoustic model (for TTS) or encoder (for STT):

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Dummy dataset – replace with actual preprocessed mel-spectrograms or features and phoneme sequences
class TTSDataset(Dataset):
    def __init__(self, audio_files, transcripts):
        self.audio_files = audio_files
        self.transcripts = transcripts

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        mel = torch.load(self.audio_files[idx])
        phonemes = self.transcripts[idx]  # Ensure this is appropriately encoded
        return mel, phonemes

# A simple acoustic model placeholder
class AcousticModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AcousticModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        output = self.decoder(encoded)
        return output

# Instantiate dataset and dataloader
dataset = TTSDataset(audio_files=['sample_mel.pt'], transcripts=["phoneme_sequence"])
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Model, optimizer, and loss function
model = AcousticModel(input_dim=80, hidden_dim=256, output_dim=80)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# Training loop
for epoch in range(50):
    for mel, phonemes in dataloader:
        # Convert phoneme sequences to embeddings if required (omitted for brevity)
        optimizer.zero_grad()
        output = model(mel)
        loss = criterion(output, mel)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} loss: {loss.item():.4f}")
```

_Note:_ This is a simplified skeleton. In practice, you would include advanced features such as attention mechanisms, custom loss functions (e.g., CTC for STT), and integration with a vocoder for TTS.

---

## 8. Conclusion

Training speech models for Indic languages is a challenging yet rewarding endeavor. Whether you choose to train from scratch or fine‑tune pre‑trained models, success hinges on high‑quality data, appropriate phonemization, consistent audio sampling rates, and carefully designed architectures. With distributed training and modern deep learning frameworks, you can tailor systems to the rich linguistic diversity of Indic languages while addressing unique challenges like tone, morphology, and code‑switching.


## 9. References

-   [Kokoro GitHub Repository](https://github.com/hexgrad/kokoro)
-   [Zonos GitHub Repository](https://github.com/Zyphra/Zonos)
-   [eSpeak-NG GitHub Repository](https://github.com/espeak-ng/espeak-ng)
-   [Conformers](https://arxiv.org/abs/2005.08100)
-   [Mann Ki Baat Dataset](https://huggingface.co/datasets/ai4bharat/Mann-ki-Baat/viewer/indic2en/hindi)
