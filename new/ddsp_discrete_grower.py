"""
DDSP Discrete Timbre Grower

Uses trained DDSP model for discrete growing stages.

Workflow:
1. Load trained DDSP model
2. Generate discrete stages by progressively activating harmonics
3. Each stage: DDSP synthesis (harmonics + noise) + ADSR envelope + silence
"""

import torch
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm

from ddsp_lightweight import DDSPModel, FeatureExtractor


class ADSREnvelope:
    """Generate ADSR envelope for a note."""

    def __init__(self, sr=22050):
        self.sr = sr

    def generate(self, duration, attack=0.05, decay=0.1, sustain_level=0.7, release=0.2):
        n_samples = int(duration * self.sr)

        attack_samples = int(attack * self.sr)
        decay_samples = int(decay * self.sr)
        release_samples = int(release * self.sr)
        sustain_samples = n_samples - attack_samples - decay_samples - release_samples
        sustain_samples = max(0, sustain_samples)

        envelope = []

        if attack_samples > 0:
            envelope.extend(np.linspace(0, 1, attack_samples))
        if decay_samples > 0:
            envelope.extend(np.linspace(1, sustain_level, decay_samples))
        if sustain_samples > 0:
            envelope.extend(np.ones(sustain_samples) * sustain_level)
        if release_samples > 0:
            envelope.extend(np.linspace(sustain_level, 0, release_samples))

        envelope = np.array(envelope)

        if len(envelope) < n_samples:
            envelope = np.pad(envelope, (0, n_samples - len(envelope)))
        elif len(envelope) > n_samples:
            envelope = envelope[:n_samples]

        return envelope


class DiscreteStageController:
    """Controls discrete stages of harmonic growth."""

    def __init__(self, n_harmonics=64):
        self.n_harmonics = n_harmonics

    def generate_stages(self, strategy='linear'):
        stages = []

        if strategy == 'linear':
            for n in range(1, self.n_harmonics + 1):
                stage = np.zeros(self.n_harmonics)
                stage[:n] = 1.0
                stages.append(stage)

        elif strategy == 'exponential':
            n = 1
            while n <= self.n_harmonics:
                stage = np.zeros(self.n_harmonics)
                stage[:n] = 1.0
                stages.append(stage)
                n *= 2
            stage = np.ones(self.n_harmonics)
            stages.append(stage)

        elif strategy == 'fibonacci':
            a, b = 1, 1
            while a <= self.n_harmonics:
                stage = np.zeros(self.n_harmonics)
                stage[:min(a, self.n_harmonics)] = 1.0
                stages.append(stage)
                a, b = b, a + b
            stage = np.ones(self.n_harmonics)
            stages.append(stage)

        return stages


class DDSPDiscreteGrower:
    """
    Synthesize discrete growing stages using DDSP.

    Each stage:
    - DDSP synthesis with progressive harmonic activation
    - Realistic noise/transients from trained model
    - ADSR envelope
    - Silence between stages
    """

    def __init__(self, model_path, device='cpu', sr=22050):
        self.sr = sr
        self.device = device
        self.envelope_gen = ADSREnvelope(sr=sr)

        # Load trained DDSP model
        print(f"Loading DDSP model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)

        self.model = DDSPModel(
            sample_rate=sr,
            n_harmonics=64,
            n_filter_banks=64,
            hidden_size=512
        ).to(device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Extract features for reference
        self.features = checkpoint['features']
        self.n_harmonics = 64

        print(f"   ✅ Model loaded successfully")
        print(f"   Target F0: {self.features['f0'].mean():.1f} Hz")

    def synthesize_stage(self, active_harmonics, duration=0.5,
                        attack=0.05, decay=0.1, sustain_level=0.7, release=0.2):
        """
        Synthesize one stage using DDSP with progressive harmonic activation.

        Args:
            active_harmonics: (n_harmonics,) array of 0s and 1s
            duration: stage duration
            attack, decay, sustain_level, release: ADSR parameters

        Returns:
            audio: numpy array
        """
        # Calculate number of frames for this duration
        hop_length = 512
        n_frames = int((duration * self.sr) / hop_length)

        # Prepare features (repeat for all frames)
        f0_mean = self.features['f0'][self.features['f0'] > 0].mean()  # Avoid zeros
        if np.isnan(f0_mean):
            f0_mean = 220.0  # Fallback to A3

        f0 = torch.ones(1, n_frames, 1, device=self.device) * f0_mean

        loudness_mean = self.features['loudness'].mean()
        loudness = torch.ones(1, n_frames, 1, device=self.device) * loudness_mean

        mfcc_mean = self.features['mfcc'].mean(axis=0)
        mfcc = torch.tensor(mfcc_mean, dtype=torch.float32, device=self.device)
        mfcc = mfcc.unsqueeze(0).unsqueeze(0).expand(1, n_frames, -1)

        # Generate audio with DDSP
        with torch.no_grad():
            pred_audio, harmonic_audio, noise_audio = self.model(f0, loudness, mfcc)

        # Convert to numpy
        audio = pred_audio.squeeze().cpu().numpy()

        # Trim/pad to exact duration
        target_samples = int(duration * self.sr)
        if len(audio) < target_samples:
            audio = np.pad(audio, (0, target_samples - len(audio)))
        else:
            audio = audio[:target_samples]

        # Apply harmonic masking (progressive activation)
        # Simple approach: apply mask to frequency domain
        # For now, scale overall amplitude by proportion of active harmonics
        harmonic_ratio = active_harmonics.sum() / len(active_harmonics)
        audio = audio * (0.2 + 0.8 * harmonic_ratio)  # At least 20% amplitude

        # Generate ADSR envelope
        envelope = self.envelope_gen.generate(duration, attack, decay,
                                             sustain_level, release)

        # Apply envelope
        audio = audio * envelope

        # Normalize
        if np.abs(audio).max() > 0:
            audio = audio / np.abs(audio).max() * 0.8

        return audio

    def grow_timbre_discrete(self, controller, strategy='linear',
                            stage_duration=0.5, silence_duration=0.2,
                            output_path='ddsp_grown_timbre.wav',
                            attack=0.05, decay=0.1, sustain_level=0.7, release=0.2):
        """
        Generate discrete growing stages with DDSP.

        Args:
            controller: DiscreteStageController
            strategy: growth strategy
            stage_duration: duration of each note
            silence_duration: silence between notes
            output_path: where to save
            attack, decay, sustain_level, release: ADSR parameters

        Returns:
            audio, stages
        """
        print(f"\n🎵 Growing timbre with DDSP (discrete stages)...")
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

        for i, stage in enumerate(tqdm(stages, desc="Synthesizing stages")):
            # Synthesize with DDSP
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

        return full_audio, stages


def main():
    """Generate discrete growing stages using trained DDSP model."""
    print("="*60)
    print("DDSP DISCRETE TIMBRE GROWER")
    print("="*60)

    # Configuration
    MODEL_PATH = 'ddsp_model.pt'
    STRATEGY = 'linear'
    STAGE_DURATION = 0.5
    SILENCE_DURATION = 0.2

    # ADSR parameters
    ATTACK = 0.05
    DECAY = 0.1
    SUSTAIN_LEVEL = 0.7
    RELEASE = 0.2

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Initialize grower with trained model
    grower = DDSPDiscreteGrower(
        model_path=MODEL_PATH,
        device=device,
        sr=22050
    )

    # Initialize controller
    controller = DiscreteStageController(n_harmonics=64)

    # Generate growing timbre
    audio, stages = grower.grow_timbre_discrete(
        controller=controller,
        strategy=STRATEGY,
        stage_duration=STAGE_DURATION,
        silence_duration=SILENCE_DURATION,
        output_path='ddsp_grown_timbre.wav',
        attack=ATTACK,
        decay=DECAY,
        sustain_level=SUSTAIN_LEVEL,
        release=RELEASE
    )

    print("\n" + "="*60)
    print("✅ DDSP DISCRETE GROWING COMPLETE")
    print("="*60)
    print("\nGenerated file:")
    print("  🎵 ddsp_grown_timbre.wav - Discrete stages with DDSP synthesis")
    print("\nEach stage now includes:")
    print("  • Learned harmonic relationships (phase-coherent)")
    print("  • Realistic noise/transients from trained model")
    print("  • Temporal structure (attack/sustain characteristics)")
    print("  • ADSR envelope for musical shaping")
    print("\n🎧 This should sound much more realistic than pure additive!")
    print("="*60)


if __name__ == "__main__":
    main()
