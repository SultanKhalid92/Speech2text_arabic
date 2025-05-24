# Arabic Audio-to-Text Pipeline with Speaker Diarization

A comprehensive Python pipeline that processes Arabic audio content from YouTube videos, performing speaker diarization, transcription, and text summarization.

## Features

- YouTube audio extraction and processing
- Audio denoising for improved quality
- Speaker diarization (identifies different speakers)
- Arabic speech transcription using Faster Whisper
- Text summarization with timestamps
- Multiple output formats (TXT & DOCX)

## Prerequisites

### Required Software
- Python 3.8 or higher
- FFmpeg (for audio processing)
- CUDA-capable GPU (optional, for faster processing)

### Required Accounts
- HuggingFace Account (for model access)
  - Need to accept terms for `pyannote/speaker-diarization-3.1`
  - Generate and save your HuggingFace token

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure settings in `settings.yaml`:
```yaml
youtube_url: "YOUR_YOUTUBE_URL"
audio_path: "audio.wav"
clean_audio_path: "audio_denoised.wav"
transcript_path: "transcript_whisper_large.txt"
summary_path: "summary"
hf_token: "YOUR_HUGGINGFACE_TOKEN"
ffmpeg_location: "PATH_TO_FFMPEG"
```

## 🚀 Usage

### Using Python Script
Run the main script:
```bash
python main.py
```

### Using Jupyter Notebook
Open and run `Speech2text.ipynb` sequentially.

## 🔧 Pipeline Components

### 1. Audio Extraction
- Uses `yt_dlp` to download YouTube audio
- Converts to WAV format at 16kHz
- Handled by [`extract_audio()`](main.py) function

### 2. Audio Denoising
- Uses `librosa` and `noisereduce`
- Improves audio quality for better transcription
- Implemented in [`denoise_audio()`](main.py) function

### 3. Speaker Diarization
- Uses `pyannote.audio` model
- Identifies different speakers in the audio
- Provides speaker timestamps
- Implemented in [`run_diarization()`](main.py) function

### 4. Transcription
- Uses `faster-whisper` model
- Optimized for Arabic language
- Includes timestamp segmentation
- Handled by [`transcribe_audio()`](main.py) function

### 5. Text Processing
- Combines speaker information with transcription
- Generates structured output with timestamps
- Creates both detailed and summarized versions

## Output Files

| File | Description |
|------|-------------|
| `audio.wav` | Original extracted audio |
| `audio_denoised.wav` | Noise-reduced audio |
| `transcript_whisper_large.txt` | Full transcript with speaker labels |
| `summary.txt` | Text summary |
| `summary.docx` | Formatted Word document summary |