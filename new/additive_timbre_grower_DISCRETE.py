"""
Additive Synthesis Timbre Grower - DISCRETE STAGES VERSION

Goal: Each harmonic "state" plays as a complete note with ADSR envelope,
      followed by silence, then the next richer harmonic state plays.

Pattern:
  Stage 1: H1 alone (attack → sustain → decay) → silence
  Stage 2: H1+H2 (attack → sustain → decay) → silence
  Stage 3: H1+H2+H3 (attack → sustain → decay) → silence
  ...
  Stage N: All harmonics (full timbre)
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
        self.audio, _ = librosa.load(audio_path, sr=sr, mono=True)
        self.f0, self.harmonic_amplitudes = self._analyze()

    def _analyze(self):
        """Extract fundamental frequency and harmonic structure."""
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

        return f0, harmonic_amplitudes


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

        elif strategy == 'powers_of_2_plus_1':
            # 1, 2, 4, 8, 16 (powers) but add 1 between each
            powers = [1]
            n = 2
            while n <= self.n_harmonics:
                powers.append(n)
                n *= 2

            # Interleave with +1
            sequence = []
            for i, p in enumerate(powers[:-1]):
                sequence.append(p)
                sequence.append(p + 1)
            sequence.append(powers[-1])

            for n in sequence:
                stage = np.zeros(self.n_harmonics)
                stage[:min(n, self.n_harmonics)] = 1.0
                stages.append(stage)

        return stages


class AdditiveDiscreteGrower:
    """
    Synthesize audio with discrete growing stages.

    Each stage is a complete note with ADSR envelope and silence after.
    """

    def __init__(self, harmonic_analyzer, sr=22050):
        self.analyzer = harmonic_analyzer
        self.sr = sr
        self.f0 = harmonic_analyzer.f0
        self.target_amplitudes = harmonic_analyzer.harmonic_amplitudes
        self.n_harmonics = len(self.target_amplitudes)
        self.envelope_gen = ADSREnvelope(sr=sr)

    def synthesize_stage(self, active_harmonics, duration=0.5,
                        attack=0.05, decay=0.1, sustain_level=0.7, release=0.2):
        """
        Synthesize one stage: a note with specific harmonics active.

        Args:
            active_harmonics: (n_harmonics,) array of 0s and 1s
            duration: note duration
            attack, decay, sustain_level, release: ADSR parameters

        Returns:
            audio: numpy array
        """
        n_samples = int(duration * self.sr)
        t = np.arange(n_samples) / self.sr

        # Generate ADSR envelope
        envelope = self.envelope_gen.generate(duration, attack, decay,
                                             sustain_level, release)

        # Synthesize harmonics
        audio = np.zeros(n_samples)
        for h in range(self.n_harmonics):
            if active_harmonics[h] > 0.01:
                freq = self.f0 * (h + 1)
                amplitude = self.target_amplitudes[h]
                audio += amplitude * np.sin(2 * np.pi * freq * t)

        # Apply envelope
        audio = audio * envelope

        # Normalize
        if np.abs(audio).max() > 0:
            audio = audio / np.abs(audio).max() * 0.8

        return audio

    def grow_timbre_discrete(self, controller, strategy='linear',
                            stage_duration=0.5, silence_duration=0.2,
                            output_path='grown_timbre_discrete.wav',
                            attack=0.05, decay=0.1, sustain_level=0.7, release=0.2):
        """
        Generate audio with discrete growing stages.

        Args:
            controller: DiscreteStageController instance
            strategy: growth strategy
            stage_duration: duration of each note
            silence_duration: silence between notes
            output_path: where to save
            attack, decay, sustain_level, release: ADSR envelope parameters

        Returns:
            audio, stages
        """
        print(f"\n🎵 Growing timbre (discrete stages)...")
        print(f"   Strategy: {strategy}")
        print(f"   Stage duration: {stage_duration}s")
        print(f"   Silence duration: {silence_duration}s")
        print(f"   ADSR: A={attack}s, D={decay}s, S={sustain_level}, R={release}s")

        # Generate stages
        stages = controller.generate_stages(strategy=strategy)
        print(f"   Total stages: {len(stages)}")

        # Synthesize each stage
        audio_segments = []
        silence = np.zeros(int(silence_duration * self.sr))

        for i, stage in enumerate(tqdm(stages, desc="Stages")):
            # Synthesize this stage
            stage_audio = self.synthesize_stage(
                stage, duration=stage_duration,
                attack=attack, decay=decay,
                sustain_level=sustain_level, release=release
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
        self.visualize_discrete_growth(stages, output_path.replace('.wav', '_stages.png'))

        return full_audio, stages

    def visualize_discrete_growth(self, stages, output_path='discrete_stages.png'):
        """Visualize the discrete stages."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

        # Plot 1: Stage progression (bar chart style)
        stage_matrix = np.array(stages).T  # (n_harmonics, n_stages)

        im = ax1.imshow(stage_matrix, aspect='auto', cmap='viridis',
                       interpolation='nearest', origin='lower')
        ax1.set_xlabel('Stage Number')
        ax1.set_ylabel('Harmonic Number')
        ax1.set_title('Discrete Harmonic Stages (Each stage = complete note)')
        plt.colorbar(im, ax=ax1, label='Active (1) or Inactive (0)')

        # Plot 2: Harmonic count per stage
        n_harmonics_per_stage = [np.sum(stage) for stage in stages]
        ax2.bar(range(len(stages)), n_harmonics_per_stage, color='steelblue', alpha=0.7)
        ax2.set_xlabel('Stage Number')
        ax2.set_ylabel('Number of Active Harmonics')
        ax2.set_title('Harmonic Complexity Per Stage')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Stage visualization saved to {output_path}")


def main():
    """
    Main execution: Discrete stages version.
    """
    print("="*60)
    print("ADDITIVE SYNTHESIS - DISCRETE STAGES VERSION")
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

    SR = 22050

    print(f"\nConfiguration:")
    print(f"  Target: {TARGET_AUDIO}")
    print(f"  Harmonics: {N_HARMONICS}")
    print(f"  Strategy: {STRATEGY}")
    print(f"  Stage duration: {STAGE_DURATION}s")
    print(f"  Silence: {SILENCE_DURATION}s")

    # Step 1: Analyze
    print("\n" + "="*60)
    print("STEP 1: Harmonic Analysis")
    print("="*60)
    analyzer = HarmonicAnalyzer(TARGET_AUDIO, sr=SR, n_harmonics=N_HARMONICS)

    # Step 2: Controller
    print("\n" + "="*60)
    print("STEP 2: Initialize Discrete Stage Controller")
    print("="*60)
    controller = DiscreteStageController(n_harmonics=N_HARMONICS)

    # Step 3: Grow
    print("\n" + "="*60)
    print("STEP 3: Generate Discrete Growing Stages")
    print("="*60)
    grower = AdditiveDiscreteGrower(analyzer, sr=SR)

    audio, stages = grower.grow_timbre_discrete(
        controller=controller,
        strategy=STRATEGY,
        stage_duration=STAGE_DURATION,
        silence_duration=SILENCE_DURATION,
        output_path='grown_timbre_discrete.wav',
        attack=ATTACK,
        decay=DECAY,
        sustain_level=SUSTAIN_LEVEL,
        release=RELEASE
    )

    # Summary
    print("\n" + "="*60)
    print("✅ DISCRETE STAGES VERSION COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print("  🎵 grown_timbre_discrete.wav - Discrete growing stages")
    print("  📈 grown_timbre_discrete_stages.png - Stage visualization")
    print("\n🎧 Each stage is a complete note with ADSR envelope!")
    print("   You should hear:")
    print("   1. Simple tone (H1) → fades out → silence")
    print("   2. Richer tone (H1+H2) → fades out → silence")
    print("   3. Even richer → ... → full timbre")
    print("\nExperiment with different strategies:")
    print("  - 'linear': adds 1 harmonic per stage")
    print("  - 'exponential': 1, 2, 4, 8, 16, 32")
    print("  - 'fibonacci': 1, 1, 2, 3, 5, 8, 13, 21...")
    print("="*60)


if __name__ == "__main__":
    main()
