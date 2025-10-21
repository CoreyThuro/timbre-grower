# Diagnostic Test: Can BigVGAN round-trip reconstruct the target audio?
# This tests whether the vocoder pipeline can EVER produce good audio

import librosa
import torch
import soundfile as sf
import numpy as np
import bigvgan

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load BigVGAN
    print("Loading BigVGAN...")
    bigvgan_model = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_base_22khz_80band', use_cuda_kernel=False)
    bigvgan_model = bigvgan_model.to(device).eval()
    print("BigVGAN loaded!")

    # Load original violin audio
    print("\nLoading target audio...")
    target_audio, sr = librosa.load('violin.wav', sr=22050)
    print(f"Original audio duration: {len(target_audio)/sr:.2f}s")
    print(f"Original RMS energy: {np.sqrt(np.mean(target_audio**2)):.4f}")

    # Convert to mel spectrogram (using same params as training)
    print("\nConverting to mel spectrogram...")
    target_mel = librosa.feature.melspectrogram(
        y=target_audio,
        sr=sr,
        n_fft=1024,
        hop_length=256,
        n_mels=80
    )
    target_mel_db = librosa.power_to_db(target_mel, ref=np.max)
    print(f"Spectrogram shape: {target_mel.shape}")
    print(f"dB range: [{target_mel_db.min():.1f}, {target_mel_db.max():.1f}]")

    # Convert back to power
    print("\nConverting dB back to power...")
    target_mel_power = 10**(target_mel_db / 10.0)

    # Prepare for BigVGAN (expects [batch, mels, time])
    mel_tensor = torch.tensor(target_mel_power, dtype=torch.float32).unsqueeze(0).to(device)

    # Vocode it
    print("\nVocoding with BigVGAN...")
    with torch.no_grad():
        reconstructed_audio = bigvgan_model(mel_tensor).squeeze().cpu().numpy()

    print(f"Reconstructed audio duration: {len(reconstructed_audio)/sr:.2f}s")
    print(f"Reconstructed RMS energy: {np.sqrt(np.mean(reconstructed_audio**2)):.4f}")

    # Save files for comparison
    sf.write('roundtrip_test.wav', reconstructed_audio, sr)
    sf.write('original_for_comparison.wav', target_audio, sr)

    print("\n" + "="*60)
    print("DIAGNOSTIC TEST COMPLETE")
    print("="*60)
    print("\nFiles saved:")
    print("  - original_for_comparison.wav (reference)")
    print("  - roundtrip_test.wav (BigVGAN reconstruction)")
    print("\n📝 LISTEN TO BOTH FILES AND COMPARE:")
    print("  ✅ If they sound similar → Vocoder works, problem is NCA")
    print("  ❌ If they sound different → Vocoder is the bottleneck")
    print("\nExpected: If roundtrip lacks violin timbre, the vocoder")
    print("pipeline is fundamentally limited for this task.")
    print("="*60)