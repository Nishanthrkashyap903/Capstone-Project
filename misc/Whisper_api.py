import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
from pydub import AudioSegment
import librosa
import numpy as np

import os

# Function to split audio into chunks of specified duration (in milliseconds)
def split_audio_into_chunks(audio, chunk_length_ms):
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunks.append(audio[i:i + chunk_length_ms])
    return chunks

# Function to convert MP3 to WAV
def convert_mp3_to_wav(mp3_path, wav_path):
    sound = AudioSegment.from_mp3(mp3_path)
    sound.export(wav_path, format="wav")

# Load MP3 and convert to WAV
mp3_path = "English_Songs/Cheri Cheri Lady.mp3"  # Path to your MP3 file
wav_path = "temp/converted_audio.wav"  # Temporary WAV file path
convert_mp3_to_wav(mp3_path, wav_path)

processor = AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3-turbo")

# Load the full WAV audio using pydub
audio = AudioSegment.from_wav(wav_path)

# Set chunk length (e.g., 30 seconds = 30,000 milliseconds)
chunk_length_ms = 30 * 1000  # 30-second chunks
audio_chunks = split_audio_into_chunks(audio, chunk_length_ms)

# Prepare for transcription
full_transcription = []
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)

# Loop through each chunk, transcribe it, and combine the transcriptions
for i, chunk in enumerate(audio_chunks):
    # Export the current chunk to a temporary WAV file
    chunk_wav_path = f"temp/chunk_{i}.wav"
    chunk.export(chunk_wav_path, format="wav")
    
    # Load the chunk with librosa and ensure 16kHz sampling rate
    audio_chunk, sample_rate = librosa.load(chunk_wav_path, sr=16000)
    
    # Preprocess the audio chunk for Whisper
    inputs = processor(audio_chunk, return_tensors="pt", sampling_rate=16000).input_features
    inputs = inputs.to(device)
    
    # Transcribe the chunk
    with torch.no_grad():
        generated_ids = model.generate(inputs)
    
    # Decode the transcription
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
    
    # Append transcription to the full transcription list
    full_transcription.append(transcription[0])
    
    # Delete the temporary chunk WAV file after use
    os.remove(chunk_wav_path)

# After all chunks are transcribed, combine the results
final_transcription = " ".join(full_transcription)
print("Full Transcription:", final_transcription)

# Delete the original WAV file after transcription
os.remove(wav_path)