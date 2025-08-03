# Import required libraries
import base64
import io
import modal
import numpy as np
import requests
import torch.nn as nn
import torchaudio.transforms as T
import torch
from pydantic import BaseModel
import soundfile as sf
import librosa

# Import custom model architecture
from src.components.model import AudioCNN

# Initialize Modal app
app = modal.App("audio-cnn-inference")

# Define container image with required dependencies
image = (modal.Image.debian_slim()
         # Install Python dependencies
         .pip_install_from_requirements("requirements.txt")
         # Install system libraries for audio processing
         .apt_install(["libsndfile1"])
         .add_local_python_source("src.components.model"))  # Add local model code

# Create volume for storing model weights
model_volume = modal.Volume.from_name("esc-model")


class AudioProcessor:  # Audio processing class for converting raw audio to spectrograms
    def __init__(self):
        # Define audio transformation pipeline
        self.transform = nn.Sequential(
            T.MelSpectrogram(  # Convert to mel spectrogram
                sample_rate=22050,
                n_fft=1024,
                hop_length=512,
                n_mels=128,
                f_min=0,
                f_max=11025
            ),
            T.AmplitudeToDB()  # Convert amplitude to decibel scale
        )

    def process_audio_chunk(self, audio_data):
        # Convert numpy array to PyTorch tensor
        waveform = torch.from_numpy(audio_data).float()
        # Add batch dimension
        waveform = waveform.unsqueeze(0)
        # Apply transformations
        spectrogram = self.transform(waveform)
        # Add channel dimension
        return spectrogram.unsqueeze(0)


class InferenceRequest(BaseModel):  # Pydantic model for request validation
    audio_data: str  # Base64 encoded audio data


@app.cls(image=image, gpu="A10G", volumes={"/models": model_volume}, scaledown_window=15)
class AudioClassifier:  # Main classifier class deployed as a Modal service
    @modal.enter()
    def load_model(self):
        """Initialize model when container starts"""
        print("Loading models on enter")
        # Set device (GPU if available)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Load saved model checkpoint
        checkpoint = torch.load('/models/best_model.pth',
                                map_location=self.device)
        self.classes = checkpoint['classes']  # Get class labels

        # Initialize model architecture
        self.model = AudioCNN(num_classes=len(self.classes))
        # Load trained weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        # Initialize audio processor
        self.audio_processor = AudioProcessor()
        print("Model loaded on enter")

    @modal.fastapi_endpoint(method="POST")
    def inference(self, request: InferenceRequest):
        """Handle inference requests"""
        # Decode base64 audio data
        audio_bytes = base64.b64decode(request.audio_data)

        # Read audio file using soundfile
        audio_data, sample_rate = sf.read(
            io.BytesIO(audio_bytes), dtype="float32")

        # Convert stereo to mono if needed
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Resample if needed
        if sample_rate != 44100:
            audio_data = librosa.resample(
                y=audio_data, orig_sr=sample_rate, target_sr=44100)

        # Process audio into spectrogram
        spectrogram = self.audio_processor.process_audio_chunk(audio_data)
        spectrogram = spectrogram.to(self.device)

        # Run inference
        with torch.no_grad():
            output, feature_maps = self.model(
                spectrogram, return_feature_maps=True)

            # Handle NaN values
            output = torch.nan_to_num(output)
            # Convert to probabilities
            probabilities = torch.softmax(output, dim=1)
            # Get top 3 predictions
            top3_probs, top3_indicies = torch.topk(probabilities[0], 3)

            # Format predictions
            predictions = [{"class": self.classes[idx.item()], "confidence": prob.item()}
                           for prob, idx in zip(top3_probs, top3_indicies)]

            # Prepare visualization data from feature maps
            viz_data = {}
            for name, tensor in feature_maps.items():
                if tensor.dim() == 4:  # [batch_size, channels, height, width]
                    # Average across channels
                    aggregated_tensor = torch.mean(tensor, dim=1)
                    squeezed_tensor = aggregated_tensor.squeeze(0)
                    numpy_array = squeezed_tensor.cpu().numpy()
                    clean_array = np.nan_to_num(numpy_array)
                    viz_data[name] = {
                        "shape": list(clean_array.shape),
                        "values": clean_array.tolist()
                    }

            # Prepare input spectrogram for visualization
            spectrogram_np = spectrogram.squeeze(0).squeeze(0).cpu().numpy()
            clean_spectrogram = np.nan_to_num(spectrogram_np)

            # Prepare waveform data (downsampled for visualization)
            max_samples = 8000
            waveform_sample_rate = 44100
            if len(audio_data) > max_samples:
                step = len(audio_data) // max_samples
                waveform_data = audio_data[::step]
            else:
                waveform_data = audio_data

        # Build response
        response = {
            "predictions": predictions,
            "visualization": viz_data,
            "input_spectrogram": {
                "shape": list(clean_spectrogram.shape),
                "values": clean_spectrogram.tolist()
            },
            "waveform": {
                "values": waveform_data.tolist(),
                "sample_rate": waveform_sample_rate,
                "duration": len(audio_data) / waveform_sample_rate
            }
        }

        return response


@app.local_entrypoint()
def main():
    # Test with sample audio file
    audio_data, sample_rate = sf.read("__test__/chirpingbirds.wav")

    # Prepare audio data for request
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sample_rate, format="WAV")
    audio_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    payload = {"audio_data": audio_b64}

    # Call deployed service
    server = AudioClassifier()
    url = server.inference.get_web_url()
    response = requests.post(url, json=payload)
    response.raise_for_status()

    # Process results
    result = response.json()

    # Print waveform info
    waveform_info = result.get("waveform", {})
    if waveform_info:
        values = waveform_info.get("values", {})
        print(f"First 10 values: {[round(v, 4) for v in values[:10]]}...")
        print(f"Duration: {waveform_info.get("duration", 0)}")

    # Print predictions
    print("Top predictions:")
    for pred in result.get("predictions", []):
        print(f"  -{pred["class"]} {pred["confidence"]:0.2%}")
