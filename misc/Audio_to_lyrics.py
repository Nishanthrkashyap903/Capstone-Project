import os
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from pydub import AudioSegment
import librosa

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

# Set up paths
songs_folder = "English_Songs"  # Path to your folder containing MP3 files
lyrics_folder = "lyrics"  # Folder where the lyrics will be saved
temp_folder = "temp"  # Temporary folder for intermediate WAV files

# Create necessary directories
os.makedirs(lyrics_folder, exist_ok=True)
os.makedirs(temp_folder, exist_ok=True)

# Load the processor and model
processor = AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3-turbo")
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)

# Set chunk length (e.g., 30 seconds = 30,000 milliseconds)
chunk_length_ms = 30 * 1000  # 30-second chunks

# Get list of MP3 files in the songs folder
mp3_files = [f for f in os.listdir(songs_folder) if f.endswith(".mp3")]

# Loop through each MP3 file and process
for mp3_file in mp3_files:
    mp3_path = os.path.join(songs_folder, mp3_file)  # Full path to the MP3 file
    wav_path = os.path.join(temp_folder, "converted_audio.wav")  # Temporary WAV file path

    # Convert MP3 to WAV
    convert_mp3_to_wav(mp3_path, wav_path)

    # Load the full WAV audio using pydub
    audio = AudioSegment.from_wav(wav_path)

    # Split audio into chunks
    audio_chunks = split_audio_into_chunks(audio, chunk_length_ms)

    # Prepare for transcription
    full_transcription = []

    # Loop through each chunk, transcribe it, and combine the transcriptions
    for i, chunk in enumerate(audio_chunks):
        # Export the current chunk to a temporary WAV file
        chunk_wav_path = os.path.join(temp_folder, f"chunk_{i}.wav")
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
    
    # Save the transcription to a text file named after the song
    song_name = os.path.splitext(mp3_file)[0]
    lyrics_path = os.path.join(lyrics_folder, f"{song_name}.txt")
    with open(lyrics_path, "w") as f:
        f.write(final_transcription.strip())  # Save the final transcription

    print(f"Transcription for '{mp3_file}' saved to '{lyrics_path}'.")

    # Delete the original WAV file after transcription
    os.remove(wav_path)

print("All songs have been processed and lyrics have been saved.")
