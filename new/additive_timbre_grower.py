"""
Additive Synthesis Timbre Grower with NCA Control

Phase 1: Proof of concept for growing timbre from simple to complex
- Extract harmonic structure from target audio
- Use NCA to control which harmonics are active over time
- Generate audio that progressively builds from single sine to full timbre
"""

import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class HarmonicAnalyzer:
    """Extract harmonic structure from target audio."""

    def __init__(self, audio_path, sr=22050, n_harmonics=32):
        self.sr = sr
        self.n_harmonics = n_harmonics

        # Load audio
        self.audio, _ = librosa.load(audio_path, sr=sr, mono=True)
        self.duration = len(self.audio) / sr

        # Analyze harmonics
        self.f0, self.harmonic_amplitudes, self.harmonic_phases = self._analyze()

    def _analyze(self):
        """Extract fundamental frequency and harmonic structure."""
        print("Analyzing target audio...")

        # Estimate fundamental frequency (F0)
        # Using multiple methods for robustness
        f0_yin = librosa.yin(self.audio, fmin=librosa.note_to_hz('C3'),
                            fmax=librosa.note_to_hz('C6'), sr=self.sr)

        # Use median for stable pitch estimate
        f0 = np.median(f0_yin[~np.isnan(f0_yin)])

        print(f"Detected F0: {f0:.2f} Hz ({librosa.hz_to_note(f0)})")

        # FFT analysis to get harmonic amplitudes and phases
        fft = np.fft.rfft(self.audio)
        freqs = np.fft.rfftfreq(len(self.audio), 1/self.sr)

        # Extract amplitude and phase for each harmonic
        harmonic_amplitudes = []
        harmonic_phases = []

        for h in range(1, self.n_harmonics + 1):
            harmonic_freq = f0 * h

            # Find closest FFT bin
            idx = np.argmin(np.abs(freqs - harmonic_freq))

            # Get complex value
            complex_val = fft[idx]
            amplitude = np.abs(complex_val)
            phase = np.angle(complex_val)

            harmonic_amplitudes.append(amplitude)
            harmonic_phases.append(phase)

        # Convert to arrays and normalize amplitudes
        harmonic_amplitudes = np.array(harmonic_amplitudes)
        harmonic_amplitudes /= (harmonic_amplitudes.max() + 1e-8)
        harmonic_phases = np.array(harmonic_phases)

        # Print harmonic structure
        print(f"\nHarmonic Structure (first 10):")
        for i in range(min(10, self.n_harmonics)):
            print(f"  H{i+1} ({f0*(i+1):.1f} Hz): amp={harmonic_amplitudes[i]:.3f}")

        return f0, harmonic_amplitudes, harmonic_phases

    def visualize_spectrum(self, output_path='harmonic_analysis.png'):
        """Visualize the extracted harmonic structure."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Plot amplitude spectrum
        harmonic_numbers = np.arange(1, self.n_harmonics + 1)
        frequencies = self.f0 * harmonic_numbers

        ax1.stem(frequencies, self.harmonic_amplitudes, basefmt=' ')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Normalized Amplitude')
        ax1.set_title('Extracted Harmonic Amplitudes')
        ax1.grid(True, alpha=0.3)

        # Plot phase spectrum
        ax2.stem(frequencies, self.harmonic_phases, basefmt=' ')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Phase (radians)')
        ax2.set_title('Harmonic Phases')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n✅ Harmonic analysis saved to {output_path}")


class NCAHarmonicController(nn.Module):
    """
    Neural Cellular Automata for controlling harmonic growth.

    The NCA learns to progressively activate harmonics from low to high,
    creating an organic "growing" timbre effect.
    """

    def __init__(self, n_harmonics=32, hidden_channels=16):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.hidden_channels = hidden_channels

        # Total channels: 1 (harmonic weight) + 1 (alpha) + hidden
        self.total_channels = 2 + hidden_channels

        # Perception: identity + gradients (simple 1D CA)
        perception_size = self.total_channels * 3  # self + left + right neighbors

        # Update network
        self.update_net = nn.Sequential(
            nn.Linear(perception_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.total_channels)
        )

        # Initialize with small weights for stability
        for layer in self.update_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def perceive(self, grid):
        """
        Perceive local neighborhood (1D cellular automata).
        grid shape: (batch, channels, harmonics)
        """
        batch, channels, n = grid.shape

        # Pad for neighborhood
        padded = F.pad(grid, (1, 1), mode='circular')

        # Extract neighborhoods: left, center, right
        left = padded[:, :, :-2]
        center = padded[:, :, 1:-1]
        right = padded[:, :, 2:]

        # Concatenate perception
        perception = torch.cat([left, center, right], dim=1)

        return perception

    def forward(self, grid, step_rate=1.0):
        """
        One update step of the cellular automata.

        Args:
            grid: (batch, channels, harmonics) current state
            step_rate: stochastic update rate (0-1)
        """
        # Perceive neighborhood
        perception = self.perceive(grid)  # (batch, channels*3, harmonics)

        # Reshape for linear layers
        batch, _, n = perception.shape
        perception = perception.permute(0, 2, 1)  # (batch, harmonics, channels*3)

        # Update via MLP
        update = self.update_net(perception)  # (batch, harmonics, total_channels)
        update = update.permute(0, 2, 1)  # (batch, total_channels, harmonics)

        # Apply update
        grid = grid + update * step_rate

        # Apply constraints
        # Alpha channel (index 1) should be in [0, 1]
        alpha = torch.sigmoid(grid[:, 1:2, :])
        grid = torch.cat([grid[:, :1, :], alpha, grid[:, 2:, :]], dim=1)

        # Apply living mask (only active cells can update neighbors)
        living_mask = (alpha > 0.1).float()
        grid = grid * living_mask

        return grid

    def initialize_seed(self, batch_size=1, device='cpu'):
        """Initialize with a single active harmonic (fundamental)."""
        grid = torch.zeros(batch_size, self.total_channels, self.n_harmonics, device=device)

        # Activate fundamental frequency (harmonic 0)
        grid[:, 0, 0] = 1.0  # Harmonic weight
        grid[:, 1, 0] = 1.0  # Alpha (alive)

        return grid

    def evolve(self, n_steps, batch_size=1, device='cpu', return_history=False):
        """
        Evolve the CA for n_steps, returning harmonic weights over time.

        Returns:
            weights: (n_steps, n_harmonics) harmonic weights at each step
        """
        grid = self.initialize_seed(batch_size, device)

        weights_history = []

        for step in range(n_steps):
            grid = self.forward(grid)

            # Extract harmonic weights (channel 0) and apply sigmoid
            weights = torch.sigmoid(grid[:, 0, :])  # (batch, n_harmonics)
            weights_history.append(weights.detach().cpu().numpy()[0])

        weights_history = np.array(weights_history)  # (n_steps, n_harmonics)

        if return_history:
            return weights_history
        else:
            return weights_history[-1]  # Return final state


class AdditiveTimbreGrower:
    """
    Synthesize audio using additive synthesis with NCA-controlled growth.
    """

    def __init__(self, harmonic_analyzer, sr=22050):
        self.analyzer = harmonic_analyzer
        self.sr = sr
        self.f0 = harmonic_analyzer.f0
        self.target_amplitudes = harmonic_analyzer.harmonic_amplitudes
        self.target_phases = harmonic_analyzer.harmonic_phases
        self.n_harmonics = len(self.target_amplitudes)

    def synthesize_frame(self, harmonic_weights, duration=0.05, fade_in=True, fade_out=True):
        """
        Synthesize a short audio frame with given harmonic weights.

        Args:
            harmonic_weights: (n_harmonics,) array of weights in [0, 1]
            duration: frame duration in seconds
            fade_in/fade_out: apply envelope to avoid clicks
        """
        n_samples = int(duration * self.sr)
        t = np.arange(n_samples) / self.sr

        audio = np.zeros(n_samples)

        # Generate each harmonic
        for h in range(self.n_harmonics):
            if harmonic_weights[h] > 0.01:  # Skip if weight too small
                freq = self.f0 * (h + 1)
                amplitude = harmonic_weights[h] * self.target_amplitudes[h]
                phase = self.target_phases[h]

                # Additive synthesis
                audio += amplitude * np.sin(2 * np.pi * freq * t + phase)

        # Apply fade in/out envelopes to avoid clicks
        if fade_in or fade_out:
            fade_samples = min(220, n_samples // 4)  # ~10ms fade at 22kHz

            if fade_in:
                fade_in_env = np.linspace(0, 1, fade_samples)
                audio[:fade_samples] *= fade_in_env

            if fade_out:
                fade_out_env = np.linspace(1, 0, fade_samples)
                audio[-fade_samples:] *= fade_out_env

        # Normalize to prevent clipping
        audio = audio / (np.abs(audio).max() + 1e-8) * 0.8

        return audio

    def grow_timbre(self, nca_controller, n_steps=96, frame_duration=0.05,
                   output_path='grown_timbre.wav', device='cpu'):
        """
        Generate audio that grows from simple to complex timbre.

        Args:
            nca_controller: NCAHarmonicController instance
            n_steps: number of evolution steps
            frame_duration: duration of each audio frame (seconds)
            output_path: where to save the audio
        """
        print(f"\n🎵 Growing timbre over {n_steps} steps...")

        # Get harmonic weight evolution from NCA
        weights_history = nca_controller.evolve(
            n_steps=n_steps,
            device=device,
            return_history=True
        )

        # Generate audio for each step
        audio_chunks = []

        for step in tqdm(range(n_steps), desc="Synthesizing"):
            weights = weights_history[step]

            # Synthesize this frame
            chunk = self.synthesize_frame(
                weights,
                duration=frame_duration,
                fade_in=(step == 0),
                fade_out=(step == n_steps - 1)
            )
            audio_chunks.append(chunk)

        # Concatenate all chunks
        full_audio = np.concatenate(audio_chunks)

        # Save
        sf.write(output_path, full_audio, self.sr)

        print(f"✅ Audio saved to {output_path}")
        print(f"   Duration: {len(full_audio)/self.sr:.2f}s")
        print(f"   RMS energy: {np.sqrt(np.mean(full_audio**2)):.4f}")

        # Visualize the growth
        self.visualize_growth(weights_history, output_path.replace('.wav', '_growth.png'))

        return full_audio, weights_history

    def visualize_growth(self, weights_history, output_path='growth_visualization.png'):
        """Visualize how harmonics grow over time."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

        # Plot 1: Harmonic activation over time (heatmap)
        im = ax1.imshow(weights_history.T, aspect='auto', cmap='viridis',
                       interpolation='bilinear', origin='lower')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Harmonic Number')
        ax1.set_title('Harmonic Growth Over Time')
        plt.colorbar(im, ax=ax1, label='Activation Weight')

        # Plot 2: Individual harmonic trajectories
        n_steps, n_harmonics = weights_history.shape
        steps = np.arange(n_steps)

        # Plot first few harmonics for clarity
        for h in range(min(8, n_harmonics)):
            ax2.plot(steps, weights_history[:, h], label=f'H{h+1}', alpha=0.7)

        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Activation Weight')
        ax2.set_title('Individual Harmonic Trajectories (first 8 harmonics)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Growth visualization saved to {output_path}")


def main():
    """
    Main execution: Extract harmonics, create NCA controller, generate growing timbre.
    """
    print("="*60)
    print("ADDITIVE SYNTHESIS TIMBRE GROWER - Phase 1")
    print("="*60)

    # Configuration
    TARGET_AUDIO = 'violin.wav'
    N_HARMONICS = 32
    N_STEPS = 96
    FRAME_DURATION = 0.05  # 50ms frames
    SR = 22050

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Step 1: Analyze target audio
    print("\n" + "="*60)
    print("STEP 1: Harmonic Analysis")
    print("="*60)

    analyzer = HarmonicAnalyzer(TARGET_AUDIO, sr=SR, n_harmonics=N_HARMONICS)
    analyzer.visualize_spectrum()

    # Step 2: Create NCA controller
    print("\n" + "="*60)
    print("STEP 2: Initialize NCA Controller")
    print("="*60)

    nca = NCAHarmonicController(n_harmonics=N_HARMONICS, hidden_channels=16)
    nca = nca.to(device)

    print(f"NCA Parameters: {sum(p.numel() for p in nca.parameters()):,}")

    # Step 3: Generate growing timbre
    print("\n" + "="*60)
    print("STEP 3: Generate Growing Timbre")
    print("="*60)

    grower = AdditiveTimbreGrower(analyzer, sr=SR)

    audio, weights = grower.grow_timbre(
        nca_controller=nca,
        n_steps=N_STEPS,
        frame_duration=FRAME_DURATION,
        output_path='grown_timbre_additive.wav',
        device=device
    )

    # Final summary
    print("\n" + "="*60)
    print("✅ PHASE 1 COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print("  📊 harmonic_analysis.png - Extracted harmonic structure")
    print("  📈 grown_timbre_additive_growth.png - Growth visualization")
    print("  🎵 grown_timbre_additive.wav - Growing timbre audio")
    print("\n🎧 Listen to grown_timbre_additive.wav to hear the timbre grow!")
    print("\nExpected: Hear harmonics progressively appearing from low to high")
    print("="*60)


if __name__ == "__main__":
    main()
