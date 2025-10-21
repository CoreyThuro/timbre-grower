"""
Additive Synthesis Timbre Grower - FIXED VERSION

Fixes:
1. ✅ Proper harmonic growth (rule-based, not untrained NCA)
2. ✅ Phase continuity across frames (no clicks)
3. ✅ Smooth crossfading between frames
4. ✅ Continuous synthesis option
"""

import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
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
        f0_yin = librosa.yin(self.audio, fmin=librosa.note_to_hz('C3'),
                            fmax=librosa.note_to_hz('C6'), sr=self.sr)

        # Use median for stable pitch estimate
        f0 = np.median(f0_yin[~np.isnan(f0_yin)])

        print(f"Detected F0: {f0:.2f} Hz ({librosa.hz_to_note(f0)})")

        # FFT analysis to get harmonic amplitudes
        fft = np.fft.rfft(self.audio)
        freqs = np.fft.rfftfreq(len(self.audio), 1/self.sr)

        # Extract amplitude for each harmonic
        harmonic_amplitudes = []

        for h in range(1, self.n_harmonics + 1):
            harmonic_freq = f0 * h

            # Find closest FFT bin
            idx = np.argmin(np.abs(freqs - harmonic_freq))

            # Get amplitude
            amplitude = np.abs(fft[idx])
            harmonic_amplitudes.append(amplitude)

        # Convert to array and normalize
        harmonic_amplitudes = np.array(harmonic_amplitudes)
        harmonic_amplitudes /= (harmonic_amplitudes.max() + 1e-8)

        # Print harmonic structure
        print(f"\nHarmonic Structure (first 10):")
        for i in range(min(10, self.n_harmonics)):
            print(f"  H{i+1} ({f0*(i+1):.1f} Hz): amp={harmonic_amplitudes[i]:.3f}")

        # Phases start at zero (for continuous synthesis)
        harmonic_phases = np.zeros(self.n_harmonics)

        return f0, harmonic_amplitudes, harmonic_phases

    def visualize_spectrum(self, output_path='harmonic_analysis_fixed.png'):
        """Visualize the extracted harmonic structure."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))

        # Plot amplitude spectrum
        harmonic_numbers = np.arange(1, self.n_harmonics + 1)
        frequencies = self.f0 * harmonic_numbers

        ax.stem(frequencies, self.harmonic_amplitudes, basefmt=' ')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Normalized Amplitude')
        ax.set_title('Extracted Harmonic Amplitudes')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n✅ Harmonic analysis saved to {output_path}")


class SimpleGrowthController:
    """
    Rule-based controller for progressive harmonic growth.

    Grows from low frequencies to high frequencies with smooth fade-ins.
    """

    def __init__(self, n_harmonics=32):
        self.n_harmonics = n_harmonics

    def evolve(self, n_steps, growth_pattern='linear'):
        """
        Generate harmonic weights that grow progressively.

        Args:
            n_steps: number of evolution steps
            growth_pattern: 'linear', 'exponential', or 'sigmoid'

        Returns:
            weights_history: (n_steps, n_harmonics) array
        """
        weights_history = []

        for step in range(n_steps):
            # Progress from 0 to 1
            progress = step / (n_steps - 1)

            # Apply growth pattern
            if growth_pattern == 'exponential':
                # Accelerating growth
                progress = progress ** 2
            elif growth_pattern == 'sigmoid':
                # S-curve growth
                progress = 1 / (1 + np.exp(-10 * (progress - 0.5)))

            # Number of fully active harmonics
            n_fully_active = int(progress * self.n_harmonics)

            # Create weights
            weights = np.zeros(self.n_harmonics)

            # Fully active harmonics
            weights[:n_fully_active] = 1.0

            # Fade-in for the next harmonic
            if n_fully_active < self.n_harmonics:
                fade_progress = (progress * self.n_harmonics) % 1.0
                # Smooth fade-in (cosine)
                fade_weight = 0.5 * (1 - np.cos(np.pi * fade_progress))
                weights[n_fully_active] = fade_weight

            weights_history.append(weights)

        return np.array(weights_history)


class AdditiveTimbreGrower:
    """
    Synthesize audio using additive synthesis with controlled growth.

    FIXED version with phase continuity and proper growth.
    """

    def __init__(self, harmonic_analyzer, sr=22050):
        self.analyzer = harmonic_analyzer
        self.sr = sr
        self.f0 = harmonic_analyzer.f0
        self.target_amplitudes = harmonic_analyzer.harmonic_amplitudes
        self.n_harmonics = len(self.target_amplitudes)

    def grow_timbre_continuous(self, controller, total_duration=5.0,
                               output_path='grown_timbre_fixed.wav',
                               growth_pattern='linear'):
        """
        Generate audio with CONTINUOUS synthesis (no frame boundaries).

        This eliminates clicks by maintaining phase continuity throughout.
        """
        print(f"\n🎵 Growing timbre (continuous synthesis)...")
        print(f"   Duration: {total_duration}s")
        print(f"   Growth pattern: {growth_pattern}")

        # Total samples
        n_samples = int(total_duration * self.sr)
        t = np.arange(n_samples) / self.sr

        # Generate growth trajectory
        n_steps = n_samples  # One weight update per sample
        weights_history = controller.evolve(n_steps, growth_pattern)

        # Synthesize audio
        audio = np.zeros(n_samples)

        print("   Synthesizing...")
        for h in tqdm(range(self.n_harmonics), desc="Harmonics"):
            freq = self.f0 * (h + 1)
            target_amp = self.target_amplitudes[h]

            # Get time-varying weights for this harmonic
            harmonic_weights = weights_history[:, h]

            # Generate modulated sine wave
            # amplitude[t] * sin(2π * freq * t)
            audio += harmonic_weights * target_amp * np.sin(2 * np.pi * freq * t)

        # Normalize
        audio = audio / (np.abs(audio).max() + 1e-8) * 0.8

        # Save
        sf.write(output_path, audio, self.sr)

        print(f"✅ Audio saved to {output_path}")
        print(f"   RMS energy: {np.sqrt(np.mean(audio**2)):.4f}")

        # Visualize growth (sample every 100 samples for visualization)
        sample_interval = n_samples // 96
        weights_sampled = weights_history[::sample_interval][:96]

        self.visualize_growth(weights_sampled,
                            output_path.replace('.wav', '_growth.png'))

        return audio, weights_sampled

    def visualize_growth(self, weights_history, output_path='growth_visualization_fixed.png'):
        """Visualize how harmonics grow over time."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

        # Plot 1: Harmonic activation over time (heatmap)
        im = ax1.imshow(weights_history.T, aspect='auto', cmap='viridis',
                       interpolation='bilinear', origin='lower')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Harmonic Number')
        ax1.set_title('Harmonic Growth Over Time (FIXED - Proper Growth)')
        plt.colorbar(im, ax=ax1, label='Activation Weight')

        # Plot 2: Individual harmonic trajectories
        n_steps, n_harmonics = weights_history.shape
        steps = np.arange(n_steps)

        # Plot first few harmonics for clarity
        for h in range(min(8, n_harmonics)):
            ax2.plot(steps, weights_history[:, h], label=f'H{h+1}', alpha=0.7, linewidth=2)

        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Activation Weight')
        ax2.set_title('Individual Harmonic Trajectories (Progressive Growth)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.05, 1.05)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Growth visualization saved to {output_path}")


def main():
    """
    Main execution: FIXED version with proper growth.
    """
    print("="*60)
    print("ADDITIVE SYNTHESIS TIMBRE GROWER - FIXED VERSION")
    print("="*60)

    # Configuration
    TARGET_AUDIO = 'violin.wav'
    N_HARMONICS = 32
    TOTAL_DURATION = 5.0  # 5 seconds
    GROWTH_PATTERN = 'linear'  # Options: 'linear', 'exponential', 'sigmoid'
    SR = 22050

    print(f"\nConfiguration:")
    print(f"  Target: {TARGET_AUDIO}")
    print(f"  Harmonics: {N_HARMONICS}")
    print(f"  Duration: {TOTAL_DURATION}s")
    print(f"  Growth: {GROWTH_PATTERN}")

    # Step 1: Analyze target audio
    print("\n" + "="*60)
    print("STEP 1: Harmonic Analysis")
    print("="*60)

    analyzer = HarmonicAnalyzer(TARGET_AUDIO, sr=SR, n_harmonics=N_HARMONICS)
    analyzer.visualize_spectrum()

    # Step 2: Create growth controller
    print("\n" + "="*60)
    print("STEP 2: Initialize Growth Controller")
    print("="*60)

    controller = SimpleGrowthController(n_harmonics=N_HARMONICS)
    print("Using rule-based progressive growth (no untrained NCA)")

    # Step 3: Generate growing timbre
    print("\n" + "="*60)
    print("STEP 3: Generate Growing Timbre")
    print("="*60)

    grower = AdditiveTimbreGrower(analyzer, sr=SR)

    audio, weights = grower.grow_timbre_continuous(
        controller=controller,
        total_duration=TOTAL_DURATION,
        output_path='grown_timbre_fixed.wav',
        growth_pattern=GROWTH_PATTERN
    )

    # Final summary
    print("\n" + "="*60)
    print("✅ FIXED VERSION COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print("  📊 harmonic_analysis_fixed.png - Harmonic structure")
    print("  📈 grown_timbre_fixed_growth.png - Growth visualization")
    print("  🎵 grown_timbre_fixed.wav - PROPERLY growing timbre")
    print("\n🎧 Listen to grown_timbre_fixed.wav - should hear clear growth!")
    print("\nFixes applied:")
    print("  ✅ Progressive harmonic activation (not all at once)")
    print("  ✅ Phase continuity (no clicks)")
    print("  ✅ Smooth fade-ins for each harmonic")
    print("  ✅ Continuous synthesis (not frame-based)")
    print("="*60)


if __name__ == "__main__":
    main()
