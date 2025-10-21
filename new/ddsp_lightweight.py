"""
Lightweight DDSP Implementation in PyTorch

Core components:
1. Harmonic Oscillator - Time-varying additive synthesis
2. Filtered Noise Generator - Learnable spectral shaping
3. Feature Extractor - f0, loudness, spectral features from audio
4. Neural Synthesizer - Maps features to synthesis parameters
5. Multi-scale Spectral Loss - For realistic audio training

No external DDSP dependencies - pure PyTorch implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import soundfile as sf
from scipy import signal as scipy_signal


# ============================================================================
# DDSP Core Components
# ============================================================================

class HarmonicOscillator(nn.Module):
    """
    Differentiable harmonic oscillator for additive synthesis.

    Generates harmonics with time-varying amplitudes and frequencies.
    """

    def __init__(self, sample_rate=22050, n_harmonics=64):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_harmonics = n_harmonics

    def forward(self, f0_hz, harmonic_amplitudes):
        """
        Generate audio from f0 and harmonic amplitudes.

        Args:
            f0_hz: (batch, n_frames) fundamental frequency in Hz
            harmonic_amplitudes: (batch, n_frames, n_harmonics) amplitude per harmonic

        Returns:
            audio: (batch, n_samples) synthesized audio
        """
        batch_size, n_frames = f0_hz.shape

        # Upsample f0 and amplitudes to audio rate
        hop_length = 512  # Frame hop in samples
        n_samples = n_frames * hop_length

        # Upsample using linear interpolation
        f0_upsampled = F.interpolate(
            f0_hz.unsqueeze(1),  # (batch, 1, n_frames)
            size=n_samples,
            mode='linear',
            align_corners=True
        ).squeeze(1)  # (batch, n_samples)

        harmonic_amplitudes_upsampled = F.interpolate(
            harmonic_amplitudes.transpose(1, 2),  # (batch, n_harmonics, n_frames)
            size=n_samples,
            mode='linear',
            align_corners=True
        ).transpose(1, 2)  # (batch, n_samples, n_harmonics)

        # Generate time vector
        t = torch.arange(n_samples, device=f0_hz.device).float() / self.sample_rate
        t = t.unsqueeze(0).expand(batch_size, -1)  # (batch, n_samples)

        # Compute phase by integrating frequency
        # phase = 2π * cumsum(f0_hz * dt)
        phase = 2 * np.pi * torch.cumsum(f0_upsampled / self.sample_rate, dim=1)

        # Generate harmonics
        audio = torch.zeros(batch_size, n_samples, device=f0_hz.device)

        for h in range(self.n_harmonics):
            harmonic_phase = phase * (h + 1)
            harmonic_signal = torch.sin(harmonic_phase)
            harmonic_signal = harmonic_signal * harmonic_amplitudes_upsampled[:, :, h]
            audio += harmonic_signal

        return audio


class FilteredNoiseGenerator(nn.Module):
    """
    Differentiable filtered noise generator.

    Generates noise with learnable time-varying spectral filtering.
    """

    def __init__(self, sample_rate=22050, n_filter_banks=64):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_filter_banks = n_filter_banks

        # Filter bank center frequencies (logarithmic spacing)
        self.register_buffer(
            'filter_freqs',
            torch.logspace(
                np.log10(20),  # 20 Hz
                np.log10(sample_rate / 2),  # Nyquist
                n_filter_banks
            )
        )

    def forward(self, filter_magnitudes):
        """
        Generate filtered noise.

        Args:
            filter_magnitudes: (batch, n_frames, n_filter_banks) filter gains

        Returns:
            noise: (batch, n_samples) filtered noise
        """
        batch_size, n_frames, _ = filter_magnitudes.shape
        hop_length = 512
        n_samples = n_frames * hop_length

        # Generate white noise
        noise = torch.randn(batch_size, n_samples, device=filter_magnitudes.device)

        # Apply FFT
        noise_fft = torch.fft.rfft(noise, dim=1)
        freqs = torch.fft.rfftfreq(n_samples, 1/self.sample_rate)
        freqs = freqs.to(filter_magnitudes.device)

        # Upsample filter magnitudes to audio rate
        filter_magnitudes_upsampled = F.interpolate(
            filter_magnitudes.transpose(1, 2),  # (batch, n_filter_banks, n_frames)
            size=n_samples,
            mode='linear',
            align_corners=True
        ).transpose(1, 2)  # (batch, n_samples, n_filter_banks)

        # Create frequency-domain filter
        # For each frequency bin, interpolate between filter bank values
        filter_response = torch.zeros(batch_size, len(freqs), device=filter_magnitudes.device)

        for i, freq in enumerate(freqs):
            # Find nearest filter banks
            # Use Gaussian interpolation between filter banks
            distances = torch.abs(torch.log(self.filter_freqs + 1e-7) - np.log(freq + 1e-7))
            weights = torch.exp(-distances**2 / 0.5)  # Gaussian kernel
            weights = weights / (weights.sum() + 1e-7)

            # Weighted sum of filter bank magnitudes
            # Average across time (simple approach)
            filter_value = (filter_magnitudes_upsampled.mean(dim=1) * weights).sum(dim=1)
            filter_response[:, i] = filter_value

        # Apply filter in frequency domain
        filtered_fft = noise_fft * filter_response

        # Convert back to time domain
        filtered_noise = torch.fft.irfft(filtered_fft, n=n_samples, dim=1)

        return filtered_noise


class FeatureExtractor:
    """
    Extract features from target audio for DDSP training.

    Extracts:
    - f0 (fundamental frequency)
    - Loudness (RMS energy)
    - MFCCs (spectral envelope)
    """

    def __init__(self, sample_rate=22050, frame_rate=43.066):  # ~512 hop @ 22050 Hz
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.hop_length = int(sample_rate / frame_rate)

    def extract(self, audio_path):
        """
        Extract features from audio file.

        Args:
            audio_path: path to audio file

        Returns:
            features: dict with 'f0', 'loudness', 'mfcc'
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)

        # Extract f0 using YIN
        f0_yin = librosa.yin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr,
            hop_length=self.hop_length
        )

        # Replace NaN with 0 (unvoiced)
        f0_yin = np.nan_to_num(f0_yin, nan=0.0)

        # Ensure f0 is positive
        f0_yin = np.maximum(f0_yin, 0.0)

        # Extract loudness (RMS)
        loudness = librosa.feature.rms(
            y=audio,
            frame_length=2048,
            hop_length=self.hop_length
        )[0]

        # Convert to dB
        loudness_db = librosa.amplitude_to_db(loudness, ref=1.0)

        # Extract MFCCs (spectral envelope)
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=30,
            hop_length=self.hop_length
        ).T  # (n_frames, n_mfcc)

        # Ensure all features have same length
        min_len = min(len(f0_yin), len(loudness_db), len(mfcc))

        features = {
            'f0': f0_yin[:min_len],
            'loudness': loudness_db[:min_len],
            'mfcc': mfcc[:min_len],
            'audio': audio,
            'n_frames': min_len
        }

        return features


# ============================================================================
# Neural Network for Parameter Generation
# ============================================================================

class DDSPSynthesizer(nn.Module):
    """
    Neural network that maps audio features to synthesis parameters.

    Architecture:
        Input: f0, loudness, MFCCs
        Processing: GRU for temporal modeling
        Output: Harmonic amplitudes, noise filter magnitudes
    """

    def __init__(
        self,
        n_harmonics=64,
        n_filter_banks=64,
        hidden_size=512,
        n_mfcc=30
    ):
        super().__init__()

        self.n_harmonics = n_harmonics
        self.n_filter_banks = n_filter_banks

        # Input: f0 (1) + loudness (1) + MFCCs (30) = 32
        input_size = 1 + 1 + n_mfcc

        # GRU for temporal modeling
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        # Output heads
        self.harmonic_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_harmonics),
            nn.Softplus()  # Ensure positive amplitudes
        )

        self.noise_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_filter_banks),
            nn.Sigmoid()  # Filter gains in [0, 1]
        )

    def forward(self, f0, loudness, mfcc):
        """
        Generate synthesis parameters from features.

        Args:
            f0: (batch, n_frames, 1)
            loudness: (batch, n_frames, 1)
            mfcc: (batch, n_frames, n_mfcc)

        Returns:
            harmonic_amplitudes: (batch, n_frames, n_harmonics)
            filter_magnitudes: (batch, n_frames, n_filter_banks)
        """
        # Concatenate features
        x = torch.cat([f0, loudness, mfcc], dim=-1)  # (batch, n_frames, input_size)

        # GRU processing
        x, _ = self.gru(x)  # (batch, n_frames, hidden_size)

        # Generate parameters
        harmonic_amplitudes = self.harmonic_head(x)
        filter_magnitudes = self.noise_head(x)

        # Scale harmonic amplitudes by loudness
        loudness_scale = torch.exp(loudness / 20.0)  # Convert dB to linear
        harmonic_amplitudes = harmonic_amplitudes * loudness_scale

        return harmonic_amplitudes, filter_magnitudes


# ============================================================================
# Complete DDSP Model
# ============================================================================

class DDSPModel(nn.Module):
    """
    Complete DDSP model: feature processing + synthesis.
    """

    def __init__(
        self,
        sample_rate=22050,
        n_harmonics=64,
        n_filter_banks=64,
        hidden_size=512
    ):
        super().__init__()

        self.sample_rate = sample_rate

        # Neural synthesizer
        self.synthesizer = DDSPSynthesizer(
            n_harmonics=n_harmonics,
            n_filter_banks=n_filter_banks,
            hidden_size=hidden_size
        )

        # Harmonic oscillator
        self.harmonic_osc = HarmonicOscillator(
            sample_rate=sample_rate,
            n_harmonics=n_harmonics
        )

        # Filtered noise generator
        self.noise_gen = FilteredNoiseGenerator(
            sample_rate=sample_rate,
            n_filter_banks=n_filter_banks
        )

        # Mix ratio (learnable)
        self.register_parameter(
            'harmonic_noise_ratio',
            nn.Parameter(torch.tensor(0.8))  # 80% harmonics, 20% noise
        )

    def forward(self, f0, loudness, mfcc):
        """
        Generate audio from features.

        Args:
            f0: (batch, n_frames, 1)
            loudness: (batch, n_frames, 1)
            mfcc: (batch, n_frames, n_mfcc)

        Returns:
            audio: (batch, n_samples)
            harmonic_audio: (batch, n_samples)
            noise_audio: (batch, n_samples)
        """
        # Generate synthesis parameters
        harmonic_amplitudes, filter_magnitudes = self.synthesizer(f0, loudness, mfcc)

        # Synthesize harmonics
        f0_hz = f0.squeeze(-1)  # (batch, n_frames)
        harmonic_audio = self.harmonic_osc(f0_hz, harmonic_amplitudes)

        # Synthesize noise
        noise_audio = self.noise_gen(filter_magnitudes)

        # Mix
        ratio = torch.sigmoid(self.harmonic_noise_ratio)
        audio = ratio * harmonic_audio + (1 - ratio) * noise_audio

        return audio, harmonic_audio, noise_audio


# ============================================================================
# Loss Functions
# ============================================================================

class MultiScaleSpectralLoss(nn.Module):
    """
    Multi-scale spectral loss for realistic audio.

    Computes spectral distance at multiple FFT sizes.
    """

    def __init__(self, fft_sizes=[2048, 1024, 512, 256]):
        super().__init__()
        self.fft_sizes = fft_sizes

    def forward(self, pred_audio, target_audio):
        """
        Compute multi-scale spectral loss.

        Args:
            pred_audio: (batch, n_samples)
            target_audio: (batch, n_samples)

        Returns:
            loss: scalar
        """
        total_loss = 0.0

        for fft_size in self.fft_sizes:
            # Compute STFT
            pred_stft = torch.stft(
                pred_audio,
                n_fft=fft_size,
                hop_length=fft_size // 4,
                window=torch.hann_window(fft_size, device=pred_audio.device),
                return_complex=True
            )

            target_stft = torch.stft(
                target_audio,
                n_fft=fft_size,
                hop_length=fft_size // 4,
                window=torch.hann_window(fft_size, device=target_audio.device),
                return_complex=True
            )

            # Magnitude loss
            pred_mag = torch.abs(pred_stft)
            target_mag = torch.abs(target_stft)

            # Log magnitude (more perceptually relevant)
            pred_log_mag = torch.log(pred_mag + 1e-5)
            target_log_mag = torch.log(target_mag + 1e-5)

            # L1 loss
            mag_loss = F.l1_loss(pred_log_mag, target_log_mag)

            total_loss += mag_loss

        return total_loss / len(self.fft_sizes)


def time_domain_loss(pred_audio, target_audio):
    """Simple L1 loss in time domain."""
    return F.l1_loss(pred_audio, target_audio)
