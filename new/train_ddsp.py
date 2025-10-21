"""
Train DDSP model on target audio.

Usage:
    python train_ddsp.py --audio violin.wav --epochs 1000 --output ddsp_model.pt
"""

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from ddsp_lightweight import (
    DDSPModel,
    FeatureExtractor,
    MultiScaleSpectralLoss,
    time_domain_loss
)


def train_ddsp(
    audio_path,
    n_epochs=1000,
    batch_size=1,
    learning_rate=1e-3,
    device='cpu',
    output_path='ddsp_model.pt'
):
    """
    Train DDSP model on single audio file.

    Args:
        audio_path: path to target audio
        n_epochs: number of training epochs
        batch_size: batch size (typically 1 for single file)
        learning_rate: learning rate
        device: 'cuda' or 'cpu'
        output_path: where to save trained model
    """
    print("="*60)
    print("DDSP MODEL TRAINING")
    print("="*60)

    # Extract features
    print(f"\n📊 Extracting features from {audio_path}...")
    extractor = FeatureExtractor(sample_rate=22050)
    features = extractor.extract(audio_path)

    print(f"   F0 range: {features['f0'].min():.1f} - {features['f0'].max():.1f} Hz")
    print(f"   Loudness range: {features['loudness'].min():.1f} - {features['loudness'].max():.1f} dB")
    print(f"   Number of frames: {features['n_frames']}")
    print(f"   Audio duration: {len(features['audio']) / 22050:.2f}s")

    # Prepare tensors
    f0 = torch.tensor(features['f0'], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # (1, n_frames, 1)
    loudness = torch.tensor(features['loudness'], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    mfcc = torch.tensor(features['mfcc'], dtype=torch.float32).unsqueeze(0)  # (1, n_frames, n_mfcc)
    target_audio = torch.tensor(features['audio'], dtype=torch.float32).unsqueeze(0)  # (1, n_samples)

    # Move to device
    f0 = f0.to(device)
    loudness = loudness.to(device)
    mfcc = mfcc.to(device)
    target_audio = target_audio.to(device)

    # Create model
    print(f"\n🏗️  Building DDSP model...")
    model = DDSPModel(
        sample_rate=22050,
        n_harmonics=64,
        n_filter_banks=64,
        hidden_size=512
    ).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Model parameters: {n_params:,}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Loss functions
    spectral_loss_fn = MultiScaleSpectralLoss().to(device)

    # Training loop
    print(f"\n🎯 Training for {n_epochs} epochs...")
    print(f"   Device: {device}")
    print(f"   Learning rate: {learning_rate}")

    losses = []
    best_loss = float('inf')

    for epoch in tqdm(range(n_epochs), desc="Training"):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        pred_audio, harmonic_audio, noise_audio = model(f0, loudness, mfcc)

        # Trim to match target length (handle edge cases)
        min_len = min(pred_audio.shape[1], target_audio.shape[1])
        pred_audio = pred_audio[:, :min_len]
        target_audio_trimmed = target_audio[:, :min_len]

        # Compute losses
        spec_loss = spectral_loss_fn(pred_audio, target_audio_trimmed)
        time_loss = time_domain_loss(pred_audio, target_audio_trimmed)

        # Combined loss (weighted)
        total_loss = 1.0 * spec_loss + 0.1 * time_loss

        # Backward pass
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Track loss
        loss_value = total_loss.item()
        losses.append(loss_value)

        # Save best model
        if loss_value < best_loss:
            best_loss = loss_value
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'features': features,
            }, output_path)

        # Print progress
        if (epoch + 1) % 100 == 0:
            print(f"\nEpoch {epoch+1}/{n_epochs}")
            print(f"  Total loss: {loss_value:.6f}")
            print(f"  Spectral loss: {spec_loss.item():.6f}")
            print(f"  Time loss: {time_loss.item():.6f}")
            print(f"  Best loss: {best_loss:.6f}")

    # Plot training curve
    print(f"\n📈 Plotting training curve...")
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('DDSP Training Loss')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig('ddsp_training_loss.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✅ Training complete!")
    print(f"   Best loss: {best_loss:.6f}")
    print(f"   Model saved to: {output_path}")
    print(f"   Training curve saved to: ddsp_training_loss.png")

    return model, features


def test_synthesis(model_path, features, device='cpu'):
    """Test the trained model by generating audio."""
    print(f"\n🎵 Testing synthesis...")

    # Load model
    checkpoint = torch.load(model_path, map_location=device)

    model = DDSPModel(
        sample_rate=22050,
        n_harmonics=64,
        n_filter_banks=64,
        hidden_size=512
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Prepare features
    f0 = torch.tensor(features['f0'], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    loudness = torch.tensor(features['loudness'], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    mfcc = torch.tensor(features['mfcc'], dtype=torch.float32).unsqueeze(0).to(device)

    # Generate audio
    with torch.no_grad():
        pred_audio, harmonic_audio, noise_audio = model(f0, loudness, mfcc)

    # Save outputs
    import soundfile as sf

    pred_audio_np = pred_audio.squeeze().cpu().numpy()
    harmonic_audio_np = harmonic_audio.squeeze().cpu().numpy()
    noise_audio_np = noise_audio.squeeze().cpu().numpy()

    sf.write('ddsp_reconstructed.wav', pred_audio_np, 22050)
    sf.write('ddsp_harmonic_only.wav', harmonic_audio_np, 22050)
    sf.write('ddsp_noise_only.wav', noise_audio_np, 22050)

    print(f"   ✅ Reconstructed audio saved to: ddsp_reconstructed.wav")
    print(f"   ✅ Harmonic component saved to: ddsp_harmonic_only.wav")
    print(f"   ✅ Noise component saved to: ddsp_noise_only.wav")

    return pred_audio_np


def main():
    parser = argparse.ArgumentParser(description='Train DDSP model')
    parser.add_argument('--audio', type=str, default='violin.wav', help='Target audio file')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--output', type=str, default='ddsp_model.pt', help='Output model path')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')

    args = parser.parse_args()

    # Check for CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        args.device = 'cpu'

    # Train
    model, features = train_ddsp(
        audio_path=args.audio,
        n_epochs=args.epochs,
        learning_rate=args.lr,
        device=args.device,
        output_path=args.output
    )

    # Test synthesis
    test_synthesis(args.output, features, device=args.device)

    print("\n" + "="*60)
    print("✅ DDSP TRAINING COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("  1. Listen to ddsp_reconstructed.wav to verify quality")
    print("  2. Use this model for discrete growing stages synthesis")
    print("  3. Adjust n_epochs or learning_rate if needed")


if __name__ == "__main__":
    main()
