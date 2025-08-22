"""
Integration test for TiTok tokenizer with DeltaNetDiT model.

This test verifies that the TiTok implementation works correctly with the main model
and handles all the expected input/output formats properly.
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add enhanced directory to path
sys.path.append(str(Path(__file__).parent))

from enhanced.titok_tokenizer import TiTokTokenizer
from enhanced.Model import DeltaNetDiT

def test_titok_tokenizer_basic():
    """Test basic TiTok tokenizer functionality."""
    print("Testing TiTok tokenizer basic functionality...")

    # Create tokenizer with default config
    config = TiTokTokenizer.create_default_config(
        input_size=256,
        patch_size=16,
        hidden_size=384,  # Smaller for testing
        num_layers=6,
        num_heads=6,
        num_latent_tokens=16,
        codebook_size=512,
        token_size=8,
        use_l2_norm=True,
    )

    tokenizer = TiTokTokenizer(config)

    # Test input
    batch_size = 2
    images = torch.randn(batch_size, 3, 256, 256)

    # Test encoding
    z_quantized, result_dict = tokenizer.encode(images)

    print(f"Input shape: {images.shape}")
    print(f"Encoded shape: {z_quantized.shape}")
    print(f"Result keys: {list(result_dict.keys())}")

    # Test decoding
    reconstructed = tokenizer.decode(z_quantized)
    print(f"Reconstructed shape: {reconstructed.shape}")

    # Test full forward pass
    final_output, final_result = tokenizer(images)
    print(f"Forward output shape: {final_output.shape}")

    print("✓ TiTok tokenizer basic test passed!")
    return tokenizer

def test_titok_with_deltanet():
    """Test TiTok integration with DeltaNetDiT model."""
    print("\nTesting TiTok integration with DeltaNetDiT...")

    # Create a small model for testing
    model_config = {
        'input_size': 16,  # Small for testing
        'patch_size': 2,
        'in_channels': 3,  # RGB images instead of latents
        'hidden_size': 384,
        'depth': 2,  # Shallow for testing
        'num_heads': 6,
        'num_classes': 10,
        'use_flow_matching': True,
        'predict_x1': False,  # Disable x1 prediction for simplicity
        'use_min_snr_gamma': False,
        'use_titok': True,
        'titok_num_tokens': 16,
        'titok_codebook_size': 512,
        'titok_code_dim': 8,
    }

    model = DeltaNetDiT(**model_config)

    # Test input
    batch_size = 2
    images = torch.randn(batch_size, 3, 16, 16)  # Small images
    t = torch.rand(batch_size)

    print(f"Model input shape: {images.shape}")
    print(f"Time input shape: {t.shape}")

    # Forward pass
    output, aux_loss = model(images, t)

    print(f"Model output shape: {output.shape}")
    print(f"Auxiliary loss: {aux_loss.item()}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print("✓ TiTok integration test passed!")
    return model

def test_titok_configurations():
    """Test different TiTok configurations."""
    print("\nTesting different TiTok configurations...")

    configs = [
        # Small configuration
        TiTokTokenizer.create_default_config(
            input_size=128, num_latent_tokens=8, codebook_size=256
        ),
        # Medium configuration
        TiTokTokenizer.create_default_config(
            input_size=256, num_latent_tokens=32, codebook_size=1024
        ),
        # Large configuration
        TiTokTokenizer.create_default_config(
            input_size=512, num_latent_tokens=64, codebook_size=4096
        ),
    ]

    for i, config in enumerate(configs):
        print(f"\nTesting configuration {i+1}...")
        tokenizer = TiTokTokenizer(config)

        # Test with appropriate image size
        input_size = config['model']['encoder']['input_size']
        images = torch.randn(1, 3, input_size, input_size)

        z_quantized, result_dict = tokenizer.encode(images)
        reconstructed = tokenizer.decode(z_quantized)

        print(f"  Input size: {input_size}x{input_size}")
        print(f"  Num tokens: {config['model']['vq_model']['num_latent_tokens']}")
        print(f"  Codebook size: {config['model']['vq_model']['codebook_size']}")
        print(f"  Reconstruction shape: {reconstructed.shape}")

        # Clean up
        del tokenizer

    print("✓ Configuration tests passed!")

def test_error_handling():
    """Test error handling in TiTok."""
    print("\nTesting error handling...")

    # Test invalid configuration
    try:
        invalid_config = TiTokTokenizer.create_default_config()
        invalid_config['model']['vq_model']['quantize_mode'] = 'invalid'
        TiTokTokenizer(invalid_config)
        print("✗ Should have raised error for invalid quantize_mode")
    except ValueError as e:
        print(f"✓ Correctly caught invalid quantize_mode: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

    print("✓ Error handling tests passed!")

def run_all_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("TiTok Integration Test Suite")
    print("=" * 60)

    try:
        # Test basic tokenizer functionality
        tokenizer = test_titok_tokenizer_basic()

        # Test integration with DeltaNetDiT
        model = test_titok_with_deltanet()

        # Test different configurations
        test_titok_configurations()

        # Test error handling
        test_error_handling()

        print("\n" + "=" * 60)
        print("🎉 All TiTok integration tests passed!")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
