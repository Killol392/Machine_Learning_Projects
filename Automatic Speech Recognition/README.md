# Automatic Speech Recognition (ASR) using Whisper

## Overview
This project implements a complete speech-to-text analysis pipeline using OpenAI Whisper on a curated subset of the **LibriSpeech** dataset. It covers audio inspection, signal analysis, ASR transcription, text preprocessing, linguistic EDA, audio-based EDA, feature extraction (text + audio), and ASR evaluation using **Word Error Rate (WER)** and **Character Error Rate (CER)**.

## Dataset
**File:** `127105-1-1.zip`  
**Source:** Subset of LibriSpeech (provided via Canvas; external data not used)  
**Contents:**  
- 37 `.flac` audio files  
- 1 transcript file (`*.trans.txt`)  
- Sampling rate: **16 kHz**  
- Bit depth: **16-bit PCM**  
- Durations: **2–16 seconds** (full table in project notebook and in the provided PDF) :contentReference[oaicite:0]{index=0}

Consistent sampling rate and depth ensure clean speech quality appropriate for ASR.

## Pipeline Summary

### 1. Audio Processing
- Extracted file metadata: sampling rate, duration, bit depth, file size.  
- Observed consistent **16 kHz / 16-bit** across all clips (pages 1–2 of the PDF) :contentReference[oaicite:1]{index=1}.  
- File size variation strictly reflects duration (page 3) :contentReference[oaicite:2]{index=2}.  
- Visualized **waveform**, **STFT spectrogram**, and **Mel-spectrogram**, confirming clean recordings with clear formants and stable intensity.

### 2. Whisper Transcription
- Loaded Whisper **base** model.  
- Transcribed all 37 audio clips.  
- Output stored as `(utt_id, whisper_text)`.  
- Manual inspection shows near-perfect alignment with reference transcripts.

### 3. Text Preprocessing
Constructed a reproducible cleaning pipeline:
- Lowercasing  
- Regex-based filler removal  
- Punctuation normalization  
- spaCy tokenization & lemmatization  
- Stopword removal  
- Optional (not activated for fairness): spell-checking, multilingual handling

Notes on errors and multilingual handling are detailed in the submitted analysis (pages 4–5) :contentReference[oaicite:3]{index=3}.

### 4. Exploratory Data Analysis

#### Text EDA
- Total tokens: **240**  
- Unique tokens: **188**  
- **Top frequent words:** story, tell, take, know, lady, woman  
- **Hapax legomena:** 154 unique words appearing once (page 5) :contentReference[oaicite:4]{index=4}  
- Confirms a vocabulary-rich literary dataset.

#### Audio EDA
- Waveform & spectrogram show clean pauses, strong voiced regions, consistent articulation.  
- **Words per second (WPS): ~1.3–4.0**, median ≈ **2.93** (page 6) :contentReference[oaicite:5]{index=5}.  
- Very few fillers detected (total: 4), aligning with script-read audio.

### 5. Feature Extraction

#### Text Features
- **Bag of Words (BoW)** - sparse count vectors  
- **TF-IDF** - weighted, topic-discriminative features  
- **1–2 gram models** - capture short phrases  
- Extracted top TF-IDF keywords per utterance (e.g., *evening*, *draw*, *unanimous*, *groan*).

#### Audio Features
Computed using `librosa`:
- **MFCCs (13-coefficient matrix)**  
- **RMS energy**  
- **Zero-Crossing Rate (ZCR)**  
(as described on page 7) :contentReference[oaicite:6]{index=6}

### 6. Evaluation (WER & CER)

Compared Whisper outputs with reference transcripts (`*.trans.txt`):

- **Overall WER:** **2.44%**  
- **Overall CER:** **1.29%**  
- **Median WER:** 0.00  
- **Most errors were single-word substitutions** (page 8) :contentReference[oaicite:7]{index=7}  
- Highest WER (~0.27) occurred in one utterance requiring lexical precision.

Whisper performs extremely well on this clean, audiobook-style dataset.

## Repository Structure
Automatic Speech Recognition/
│── 127105-1-1.zip
│── ASR_Analysis.ipynb
│── README.md

## How to Run
pip install openai-whisper librosa soundfile spacy jiwer
python -m spacy download en_core_web_sm

unzip 127105-1-1.zip -d data/

Then run the notebook cell-by-cell.

## Key Takeaways
- Whisper achieves **high transcription accuracy** on structured read-speech.  
- Text cleaning significantly affects fair WER/CER comparison.  
- Combining **MFCCs + TF-IDF** yields a useful dual-modality representation.  
- Very low error rates show Whisper is robust even without fine-tuning.

## Summary
This project delivers a clean, end-to-end ASR workflow: audio analysis -> Whisper transcription -> preprocessing -> EDA -> feature extraction -> evaluation. It focuses on clarity, reproducibility, and correctness, making it suitable for ML coursework, research, and real-world ASR experimentation.
