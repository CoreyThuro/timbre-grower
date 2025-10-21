"""
Additive Synthesis Timbre Grower - WITH FILTERED NOISE

Enhancement over DISCRETE version:
- Adds filtered noise component for transients and texture
- Extracts noise from target audio (residual after harmonic subtraction)
- Synthesizes time-varying filtered noise for each discrete stage
- Maintains discrete stage structure with ADSR envelopes
"""

import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import signal


class HarmonicAnalyzer:
    """Extract harmonic structure and noise from target audio."""

    def __init__(self, audio_path, sr=22050, n_harmonics=32):
        self.sr = sr
        self.n_harmonics = n_harmonics
        self.audio, _ = librosa.load(audio_path, sr=sr, mono=True)
        self.f0, self.harmonic_amplitudes, self.noise_profile = self._analyze()

    def _analyze(self):
        """Extract fundamental frequency, harmonics, and noise component."""
        print("Analyzing target audio...")

        # Estimate F0
        f0_yin = librosa.yin(self.audio, fmin=librosa.note_to_hz('C3'),
                            fmax=librosa.note_to_hz('C6'), sr=self.sr)
        f0 = np.median(f0_yin[~np.isnan(f0_yin)])
        print(f"Detected F0: {f0:.2f} Hz ({librosa.hz_to_note(f0)})")

        # FFT analysis for harmonic amplitudes
        fft = np.fft.rfft(self.audio)
        freqs = np.fft.rfftfreq(len(self.audio), 1/self.sr)

        harmonic_amplitudes = []
        for h in range(1, self.n_harmonics + 1):
            idx = np.argmin(np.abs(freqs - (f0 * h)))
            harmonic_amplitudes.append(np.abs(fft[idx]))

        harmonic_amplitudes = np.array(harmonic_amplitudes)
        harmonic_amplitudes /= (harmonic_amplitudes.max() + 1e-8)

        print(f"\nHarmonic Structure (first 10):")
        for i in range(min(10, self.n_harmonics)):
            print(f"  H{i+1} ({f0*(i+1):.1f} Hz): amp={harmonic_amplitudes[i]:.3f}")

        # Extract noise component
        noise_profile = self._extract_noise(f0, harmonic_amplitudes)

        return f0, harmonic_amplitudes, noise_profile

    def _extract_noise(self, f0, harmonic_amplitudes):
        """Extract noise by subtracting harmonic reconstruction from original audio."""
        print("\nExtracting noise component...")

        # Reconstruct harmonics
        n_samples = len(self.audio)
        t = np.arange(n_samples) / self.sr

        harmonic_reconstruction = np.zeros(n_samples)
        for h in range(self.n_harmonics):
            freq = f0 * (h + 1)
            amplitude = harmonic_amplitudes[h]
            harmonic_reconstruction += amplitude * np.sin(2 * np.pi * freq * t)

        # Normalize harmonic reconstruction to match original audio level
        if np.abs(harmonic_reconstruction).max() > 0:
            harmonic_reconstruction *= (np.abs(self.audio).max() / np.abs(harmonic_reconstruction).max())

        # Residual is the noise component
        residual = self.audio - harmonic_reconstruction

        # Analyze noise characteristics
        # 1. Overall RMS level
        noise_rms = np.sqrt(np.mean(residual**2))

        # 2. Spectral tilt (energy distribution across frequency)
        # Compute power spectral density
        freqs, psd = signal.welch(residual, fs=self.sr, nperseg=2048)

        # Divide into low/mid/high frequency bands
        low_band = (freqs >= 0) & (freqs < 1000)
        mid_band = (freqs >= 1000) & (freqs < 5000)
        high_band = (freqs >= 5000)

        low_energy = np.mean(psd[low_band]) if np.any(low_band) else 0
        mid_energy = np.mean(psd[mid_band]) if np.any(mid_band) else 0
        high_energy = np.mean(psd[high_band]) if np.any(high_band) else 0

        # Normalize
        total_energy = low_energy + mid_energy + high_energy + 1e-8
        spectral_tilt = np.array([
            low_energy / total_energy,
            mid_energy / total_energy,
            high_energy / total_energy
        ])

        # 3. Time-varying envelope (using frame-wise RMS)
        frame_length = 2048
        hop_length = 512
        noise_envelope = librosa.feature.rms(y=residual, frame_length=frame_length,
                                            hop_length=hop_length)[0]

        print(f"  Noise RMS: {noise_rms:.4f}")
        print(f"  Spectral tilt (low/mid/high): {spectral_tilt[0]:.3f}/{spectral_tilt[1]:.3f}/{spectral_tilt[2]:.3f}")
        print(f"  Envelope shape: {len(noise_envelope)} frames")

        return {
            'rms': noise_rms,
            'spectral_tilt': spectral_tilt,
            'envelope': noise_envelope,
            'residual': residual  # Store for visualization
        }


class ADSREnvelope:
    """Generate ADSR envelope for a note."""

    def __init__(self, sr=22050):
        self.sr = sr

    def generate(self, duration, attack=0.05, decay=0.1, sustain_level=0.7, release=0.2):
        """
        Generate ADSR envelope.

        Args:
            duration: total duration in seconds
            attack: attack time (0 → 1)
            decay: decay time (1 → sustain_level)
            sustain_level: sustain amplitude (0-1)
            release: release time (sustain_level → 0)

        Returns:
            envelope: numpy array of amplitude values
        """
        n_samples = int(duration * self.sr)

        # Calculate sample counts for each phase
        attack_samples = int(attack * self.sr)
        decay_samples = int(decay * self.sr)
        release_samples = int(release * self.sr)
        sustain_samples = n_samples - attack_samples - decay_samples - release_samples

        # Ensure non-negative
        sustain_samples = max(0, sustain_samples)

        envelope = []

        # Attack: 0 → 1
        if attack_samples > 0:
            attack_env = np.linspace(0, 1, attack_samples)
            envelope.extend(attack_env)

        # Decay: 1 → sustain_level
        if decay_samples > 0:
            decay_env = np.linspace(1, sustain_level, decay_samples)
            envelope.extend(decay_env)

        # Sustain: constant at sustain_level
        if sustain_samples > 0:
            sustain_env = np.ones(sustain_samples) * sustain_level
            envelope.extend(sustain_env)

        # Release: sustain_level → 0
        if release_samples > 0:
            release_env = np.linspace(sustain_level, 0, release_samples)
            envelope.extend(release_env)

        envelope = np.array(envelope)

        # Ensure exact length
        if len(envelope) < n_samples:
            envelope = np.pad(envelope, (0, n_samples - len(envelope)))
        elif len(envelope) > n_samples:
            envelope = envelope[:n_samples]

        return envelope


class DiscreteStageController:
    """
    Controls discrete stages of harmonic growth.

    Each stage adds more harmonics and plays as a complete note.
    """

    def __init__(self, n_harmonics=32):
        self.n_harmonics = n_harmonics

    def generate_stages(self, strategy='linear'):
        """
        Generate the sequence of harmonic states.

        Args:
            strategy: 'linear' (1, 2, 3, ...), 'exponential' (1, 2, 4, 8, ...),
                     'fibonacci' (1, 1, 2, 3, 5, 8, ...)

        Returns:
            stages: list of arrays, each array is active harmonics for that stage
        """
        stages = []

        if strategy == 'linear':
            # Add harmonics one at a time
            for n in range(1, self.n_harmonics + 1):
                stage = np.zeros(self.n_harmonics)
                stage[:n] = 1.0
                stages.append(stage)

        elif strategy == 'exponential':
            # 1, 2, 4, 8, 16, 32
            n = 1
            while n <= self.n_harmonics:
                stage = np.zeros(self.n_harmonics)
                stage[:n] = 1.0
                stages.append(stage)
                n *= 2
            # Final stage: all harmonics
            stage = np.ones(self.n_harmonics)
            stages.append(stage)

        elif strategy == 'fibonacci':
            # 1, 1, 2, 3, 5, 8, 13, 21, 34...
            a, b = 1, 1
            while a <= self.n_harmonics:
                stage = np.zeros(self.n_harmonics)
                stage[:min(a, self.n_harmonics)] = 1.0
                stages.append(stage)
                a, b = b, a + b
            # Final stage: all harmonics
            stage = np.ones(self.n_harmonics)
            stages.append(stage)

        return stages


class AdditiveDiscreteGrowerWithNoise:
    """
    Synthesize audio with discrete growing stages + filtered noise.

    Each stage is a complete note with:
    - Harmonics (additive synthesis)
    - Filtered noise (transients/texture)
    - ADSR envelope
    - Silence after
    """

    def __init__(self, harmonic_analyzer, sr=22050):
        self.analyzer = harmonic_analyzer
        self.sr = sr
        self.f0 = harmonic_analyzer.f0
        self.target_amplitudes = harmonic_analyzer.harmonic_amplitudes
        self.noise_profile = harmonic_analyzer.noise_profile
        self.n_harmonics = len(self.target_amplitudes)
        self.envelope_gen = ADSREnvelope(sr=sr)

    def _generate_filtered_noise(self, duration, noise_amount=1.0):
        """
        Generate filtered noise based on target noise profile.

        Args:
            duration: duration in seconds
            noise_amount: scaling factor (0-1) for how much noise to include

        Returns:
            filtered_noise: numpy array of filtered noise
        """
        n_samples = int(duration * self.sr)

        # Generate white noise
        white_noise = np.random.randn(n_samples)

        # Apply spectral shaping using biquad filters
        # Low band: 0-1000 Hz
        # Mid band: 1000-5000 Hz
        # High band: 5000+ Hz

        spectral_tilt = self.noise_profile['spectral_tilt']

        # Design filters
        nyquist = self.sr / 2

        # Lowpass for low band (0-1000 Hz)
        sos_low = signal.butter(4, 1000 / nyquist, btype='low', output='sos')
        low_component = signal.sosfilt(sos_low, white_noise) * spectral_tilt[0]

        # Bandpass for mid band (1000-5000 Hz)
        sos_mid = signal.butter(4, [1000 / nyquist, 5000 / nyquist], btype='band', output='sos')
        mid_component = signal.sosfilt(sos_mid, white_noise) * spectral_tilt[1]

        # Highpass for high band (5000+ Hz)
        sos_high = signal.butter(4, 5000 / nyquist, btype='high', output='sos')
        high_component = signal.sosfilt(sos_high, white_noise) * spectral_tilt[2]

        # Combine bands
        filtered_noise = low_component + mid_component + high_component

        # Scale by target RMS and noise_amount
        target_rms = self.noise_profile['rms'] * noise_amount
        if np.abs(filtered_noise).max() > 0:
            current_rms = np.sqrt(np.mean(filtered_noise**2))
            filtered_noise = filtered_noise * (target_rms / (current_rms + 1e-8))

        return filtered_noise

    def synthesize_stage(self, active_harmonics, duration=0.5,
                        attack=0.05, decay=0.1, sustain_level=0.7, release=0.2,
                        noise_amount=1.0):
        """
        Synthesize one stage: harmonics + filtered noise with ADSR envelope.

        Args:
            active_harmonics: (n_harmonics,) array of 0s and 1s
            duration: note duration
            attack, decay, sustain_level, release: ADSR parameters
            noise_amount: how much noise to include (0-1)

        Returns:
            audio: numpy array
        """
        n_samples = int(duration * self.sr)
        t = np.arange(n_samples) / self.sr

        # Generate ADSR envelope
        envelope = self.envelope_gen.generate(duration, attack, decay,
                                             sustain_level, release)

        # Synthesize harmonics
        harmonic_audio = np.zeros(n_samples)
        for h in range(self.n_harmonics):
            if active_harmonics[h] > 0.01:
                freq = self.f0 * (h + 1)
                amplitude = self.target_amplitudes[h]
                harmonic_audio += amplitude * np.sin(2 * np.pi * freq * t)

        # Generate filtered noise
        noise_audio = self._generate_filtered_noise(duration, noise_amount)

        # Mix harmonics and noise (roughly 80% harmonic, 20% noise by default)
        harmonic_weight = 0.8
        noise_weight = 0.2
        audio = harmonic_weight * harmonic_audio + noise_weight * noise_audio

        # Apply envelope
        audio = audio * envelope

        # Normalize
        if np.abs(audio).max() > 0:
            audio = audio / np.abs(audio).max() * 0.8

        return audio

    def grow_timbre_discrete(self, controller, strategy='linear',
                            stage_duration=0.5, silence_duration=0.2,
                            output_path='grown_timbre_with_noise.wav',
                            attack=0.05, decay=0.1, sustain_level=0.7, release=0.2,
                            noise_amount=1.0):
        """
        Generate audio with discrete growing stages (harmonics + filtered noise).

        Args:
            controller: DiscreteStageController instance
            strategy: growth strategy
            stage_duration: duration of each note
            silence_duration: silence between notes
            output_path: where to save
            attack, decay, sustain_level, release: ADSR envelope parameters
            noise_amount: how much noise to include (0-1)

        Returns:
            audio, stages
        """
        print(f"\n🎵 Growing timbre with filtered noise (discrete stages)...")
        print(f"   Strategy: {strategy}")
        print(f"   Stage duration: {stage_duration}s")
        print(f"   Silence duration: {silence_duration}s")
        print(f"   ADSR: A={attack}s, D={decay}s, S={sustain_level}, R={release}s")
        print(f"   Noise amount: {noise_amount}")

        # Generate stages
        stages = controller.generate_stages(strategy=strategy)
        print(f"   Total stages: {len(stages)}")

        # Synthesize each stage
        audio_segments = []
        silence = np.zeros(int(silence_duration * self.sr))

        for i, stage in enumerate(tqdm(stages, desc="Stages")):
            # Synthesize this stage with noise
            stage_audio = self.synthesize_stage(
                stage, duration=stage_duration,
                attack=attack, decay=decay,
                sustain_level=sustain_level, release=release,
                noise_amount=noise_amount
            )

            audio_segments.append(stage_audio)

            # Add silence (except after last stage)
            if i < len(stages) - 1:
                audio_segments.append(silence)

        # Concatenate
        full_audio = np.concatenate(audio_segments)

        # Save
        sf.write(output_path, full_audio, self.sr)

        print(f"\n✅ Audio saved to {output_path}")
        print(f"   Total duration: {len(full_audio)/self.sr:.2f}s")
        print(f"   RMS energy: {np.sqrt(np.mean(full_audio**2)):.4f}")

        # Visualize
        self.visualize_analysis(output_path.replace('.wav', '_analysis.png'))

        return full_audio, stages

    def visualize_analysis(self, output_path='noise_analysis.png'):
        """Visualize harmonic analysis and noise extraction."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Harmonic amplitudes
        ax1 = axes[0, 0]
        harmonic_numbers = np.arange(1, self.n_harmonics + 1)
        ax1.stem(harmonic_numbers, self.target_amplitudes, basefmt=' ')
        ax1.set_xlabel('Harmonic Number')
        ax1.set_ylabel('Normalized Amplitude')
        ax1.set_title('Extracted Harmonic Amplitudes')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Noise spectral tilt
        ax2 = axes[0, 1]
        bands = ['Low\n(0-1kHz)', 'Mid\n(1-5kHz)', 'High\n(5kHz+)']
        ax2.bar(bands, self.noise_profile['spectral_tilt'], color='steelblue', alpha=0.7)
        ax2.set_ylabel('Normalized Energy')
        ax2.set_title('Noise Spectral Distribution')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3, axis='y')

        # Plot 3: Original audio waveform
        ax3 = axes[1, 0]
        t_audio = np.arange(len(self.analyzer.audio)) / self.sr
        ax3.plot(t_audio, self.analyzer.audio, alpha=0.7, linewidth=0.5)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Amplitude')
        ax3.set_title('Original Audio Waveform')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Extracted noise residual
        ax4 = axes[1, 1]
        residual = self.noise_profile['residual']
        t_residual = np.arange(len(residual)) / self.sr
        ax4.plot(t_residual, residual, alpha=0.7, linewidth=0.5, color='orange')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Amplitude')
        ax4.set_title('Extracted Noise Component')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Analysis visualization saved to {output_path}")


def main():
    """
    Main execution: Discrete stages with filtered noise.
    """
    print("="*60)
    print("ADDITIVE SYNTHESIS - WITH FILTERED NOISE")
    print("="*60)

    # Configuration
    TARGET_AUDIO = 'violin.wav'
    N_HARMONICS = 32
    STRATEGY = 'linear'  # Options: 'linear', 'exponential', 'fibonacci'
    STAGE_DURATION = 0.5  # Duration of each note
    SILENCE_DURATION = 0.2  # Silence between notes

    # ADSR Envelope parameters
    ATTACK = 0.05  # 50ms attack
    DECAY = 0.1    # 100ms decay
    SUSTAIN_LEVEL = 0.7  # 70% sustain level
    RELEASE = 0.2  # 200ms release

    # Noise amount (0-1)
    NOISE_AMOUNT = 0.5  # 50% of extracted noise level

    SR = 22050

    print(f"\nConfiguration:")
    print(f"  Target: {TARGET_AUDIO}")
    print(f"  Harmonics: {N_HARMONICS}")
    print(f"  Strategy: {STRATEGY}")
    print(f"  Stage duration: {STAGE_DURATION}s")
    print(f"  Silence: {SILENCE_DURATION}s")
    print(f"  Noise amount: {NOISE_AMOUNT}")

    # Step 1: Analyze
    print("\n" + "="*60)
    print("STEP 1: Harmonic and Noise Analysis")
    print("="*60)
    analyzer = HarmonicAnalyzer(TARGET_AUDIO, sr=SR, n_harmonics=N_HARMONICS)

    # Step 2: Controller
    print("\n" + "="*60)
    print("STEP 2: Initialize Discrete Stage Controller")
    print("="*60)
    controller = DiscreteStageController(n_harmonics=N_HARMONICS)

    # Step 3: Grow with noise
    print("\n" + "="*60)
    print("STEP 3: Generate Discrete Growing Stages with Filtered Noise")
    print("="*60)
    grower = AdditiveDiscreteGrowerWithNoise(analyzer, sr=SR)

    audio, stages = grower.grow_timbre_discrete(
        controller=controller,
        strategy=STRATEGY,
        stage_duration=STAGE_DURATION,
        silence_duration=SILENCE_DURATION,
        output_path='grown_timbre_with_noise.wav',
        attack=ATTACK,
        decay=DECAY,
        sustain_level=SUSTAIN_LEVEL,
        release=RELEASE,
        noise_amount=NOISE_AMOUNT
    )

    # Summary
    print("\n" + "="*60)
    print("✅ DISCRETE STAGES WITH FILTERED NOISE COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print("  🎵 grown_timbre_with_noise.wav - Discrete stages with noise/transients")
    print("  📈 grown_timbre_with_noise_analysis.png - Harmonic and noise analysis")
    print("\n🎧 Each stage now includes:")
    print("   • Harmonics (additive synthesis)")
    print("   • Filtered noise (transients/texture from target)")
    print("   • ADSR envelope")
    print("   • Silence between stages")
    print("\nThe noise component captures:")
    print("   • Bow stroke noise (for violin)")
    print("   • Air noise, key clicks (for wind instruments)")
    print("   • Attack transients (for percussion)")
    print("   • Any non-harmonic timbral characteristics")
    print("\nExperiment with NOISE_AMOUNT (0-1) to adjust the balance!")
    print("="*60)


if __name__ == "__main__":
    main()
