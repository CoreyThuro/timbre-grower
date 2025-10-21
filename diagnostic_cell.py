# DIAGNOSTIC: Check what DDSP is actually producing
print("🔍 DDSP Model Diagnostics\n")

model.eval()
with torch.no_grad():
    # Test on a segment
    test_batch = sample_batch(all_features, 1, SEGMENT_FRAMES)

    # Get synthesis parameters
    harmonic_amps, filter_mags = model.synthesizer(
        test_batch['f0'],
        test_batch['loudness'],
        test_batch['mfcc']
    )

    print("1. Harmonic Amplitudes Analysis:")
    print(f"   Shape: {harmonic_amps.shape}")
    print(f"   Mean per harmonic (first 10):")
    for i in range(min(10, N_HARMONICS)):
        mean_amp = harmonic_amps[0, :, i].mean().item()
        max_amp = harmonic_amps[0, :, i].max().item()
        print(f"     Harmonic {i+1}: mean={mean_amp:.6f}, max={max_amp:.6f}")

    # Check if only fundamental is active
    total_energy = harmonic_amps.sum(dim=(0, 1))
    fundamental_energy = harmonic_amps[:, :, 0].sum()
    print(f"\n   Fundamental energy: {fundamental_energy.item():.4f}")
    print(f"   Total energy: {total_energy.item():.4f}")
    print(f"   Fundamental %: {100 * fundamental_energy / total_energy:.1f}%")

    # Generate audio
    pred_audio, harmonic_audio, noise_audio = model(
        test_batch['f0'],
        test_batch['loudness'],
        test_batch['mfcc']
    )

    print(f"\n2. Audio Output Analysis:")
    print(f"   Harmonic audio RMS: {harmonic_audio.pow(2).mean().sqrt().item():.6f}")
    print(f"   Noise audio RMS: {noise_audio.pow(2).mean().sqrt().item():.6f}")
    print(f"   Mixed audio RMS: {pred_audio.pow(2).mean().sqrt().item():.6f}")
    print(f"   Harmonic/Noise ratio: {torch.sigmoid(model.harmonic_noise_ratio).item():.3f}")

print("\n3. Target Audio Analysis:")
print(f"   F0 range: {test_batch['f0'].min():.1f} - {test_batch['f0'].max():.1f} Hz")
print(f"   Loudness range: {test_batch['loudness'].min():.1f} - {test_batch['loudness'].max():.1f} dB")
print(f"   Target audio RMS: {test_batch['audio'].pow(2).mean().sqrt().item():.6f}")

print("\n4. Model Parameters:")
print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"   Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Check if model weights actually changed from initialization
print("\n5. Weight Update Check:")
first_layer_weight = model.synthesizer.gru.weight_ih_l0
print(f"   GRU first layer weight norm: {first_layer_weight.norm().item():.4f}")
print(f"   GRU first layer weight mean: {first_layer_weight.mean().item():.6f}")

# Visualize harmonic amplitudes
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Plot 1: Harmonic amplitudes over time
harmonic_amps_np = harmonic_amps[0].cpu().numpy()
im = axes[0].imshow(harmonic_amps_np.T, aspect='auto', origin='lower', cmap='viridis')
axes[0].set_ylabel('Harmonic Number')
axes[0].set_xlabel('Time Frame')
axes[0].set_title('DDSP Harmonic Amplitudes Over Time')
plt.colorbar(im, ax=axes[0])

# Plot 2: Mean amplitude per harmonic
mean_amps = harmonic_amps_np.mean(axis=0)
axes[1].bar(range(len(mean_amps)), mean_amps)
axes[1].set_xlabel('Harmonic Number')
axes[1].set_ylabel('Mean Amplitude')
axes[1].set_title('Average Amplitude per Harmonic')
axes[1].set_yscale('log')

# Plot 3: Waveform comparison
target_wav = test_batch['audio'][0].cpu().numpy()[:1000]
pred_wav = pred_audio[0].cpu().numpy()[:1000]
axes[2].plot(target_wav, label='Target', alpha=0.7)
axes[2].plot(pred_wav, label='Predicted', alpha=0.7)
axes[2].set_xlabel('Sample')
axes[2].set_ylabel('Amplitude')
axes[2].set_title('Waveform Comparison (first 1000 samples)')
axes[2].legend()

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("DIAGNOSIS:")
if fundamental_energy / total_energy > 0.9:
    print("⚠️  PROBLEM: Model is only using fundamental frequency!")
    print("   This explains the sine tone output.")
    print("   Possible causes:")
    print("   - Learning rate too high (try 1e-4 instead of 1e-3)")
    print("   - Batch size too large (try 8 instead of 16)")
    print("   - Training too fast (model not converging properly)")
    print("   - Spectral loss not penalizing harmonic absence")
else:
    print("✅ Model is using multiple harmonics correctly")
print("="*60)
