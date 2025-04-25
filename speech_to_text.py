#!/usr/bin/env python3
"""
speech_to_text.py

A basic CLI tool to transcribe short WAV audio clips using either:
 - Google Web Speech API (via SpeechRecognition)
 - Hugging Face Wav2Vec2 model

Usage:
  python speech_to_text.py --method [google|wav2vec2] path/to/audio.wav

"""
import argparse
import sys

# Suppress Hugging Face informational warnings about missing weights
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

# Google Web Speech API via SpeechRecognition
import speech_recognition as sr

# Wav2Vec2 model via Hugging Face
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import soundfile as sf
import numpy as np
from torchaudio.transforms import Resample


def transcribe_google(audio_path: str) -> str:
    """
    Transcribe using Google Web Speech API.
    Requires internet connection.
    """
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "[ERROR] Could not understand audio"
    except sr.RequestError as e:
        return f"[ERROR] API request failed: {e}"


def transcribe_wav2vec2(audio_path: str) -> str:
    """
    Transcribe using Facebook wav2vec2-base-960h model.
    Downloads model on first run.
    Pads or resamples too-short clips to avoid convolution errors.
    """
    # Read audio file (numpy array) and original sample rate
    speech, sample_rate = sf.read(audio_path)

    # If stereo, convert to mono by averaging channels
    if speech.ndim > 1:
        speech = np.mean(speech, axis=1)

    # Resample to 16kHz if needed
    target_rate = 16000
    if sample_rate != target_rate:
        speech_tensor = torch.from_numpy(speech).float()
        resampler = Resample(orig_freq=sample_rate, new_freq=target_rate)
        speech = resampler(speech_tensor).numpy()
        sample_rate = target_rate

    # Ensure minimal length for convolution (kernel size=10)
    min_samples = 10
    if speech.shape[0] < min_samples:
        padding = min_samples - speech.shape[0]
        speech = np.pad(speech, (0, padding), mode='constant', constant_values=0)

    # Load processor & model
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    # Prepare input
    inputs = processor(speech, sampling_rate=sample_rate, return_tensors="pt", padding=True)

    # Perform inference
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)

    # Decode to text
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription.lower()


def main():
    parser = argparse.ArgumentParser(description="Transcribe WAV audio using Google or Wav2Vec2")
    parser.add_argument("--method", choices=["google", "wav2vec2"], default="google",
                        help="Transcription method to use")
    parser.add_argument("audio", help="Path to the WAV audio file to transcribe")
    args = parser.parse_args()

    if args.method == "google":
        result = transcribe_google(args.audio)
    else:
        result = transcribe_wav2vec2(args.audio)

    print("Transcription:", result)


if __name__ == "__main__":
    main()