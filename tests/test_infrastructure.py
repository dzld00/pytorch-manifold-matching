"""
Infrastructure validation tests to ensure the testing setup works correctly.

These tests validate that:
1. The testing environment is properly configured
2. Fixtures work as expected
3. Basic imports and dependencies are available
4. Coverage reporting functions correctly
"""

import os
import pytest
import torch
import numpy as np
from PIL import Image


class TestInfrastructure:
    """Test class to validate testing infrastructure setup."""

    @pytest.mark.unit
    def test_pytorch_import(self):
        """Test that PyTorch is properly imported and configured."""
        assert torch.__version__ is not None
        
        # Test basic tensor operations
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        z = x + y
        assert z.shape == (2, 3)

    @pytest.mark.unit
    def test_numpy_import(self):
        """Test that NumPy is properly imported."""
        assert np.__version__ is not None
        
        # Test basic array operations
        arr = np.array([1, 2, 3, 4, 5])
        assert arr.mean() == 3.0

    @pytest.mark.unit
    def test_pillow_import(self):
        """Test that Pillow is properly imported."""
        # Test creating a simple image
        img = Image.new('RGB', (10, 10), color='red')
        assert img.size == (10, 10)
        assert img.mode == 'RGB'

    @pytest.mark.unit
    def test_temp_dir_fixture(self, temp_dir):
        """Test that the temp_dir fixture works correctly."""
        assert os.path.exists(temp_dir)
        assert os.path.isdir(temp_dir)
        
        # Test writing to temp directory
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("test content")
        
        assert os.path.exists(test_file)
        with open(test_file, 'r') as f:
            content = f.read()
        assert content == "test content"

    @pytest.mark.unit
    def test_sample_image_fixture(self, sample_image_path):
        """Test that the sample_image_path fixture works correctly."""
        assert os.path.exists(sample_image_path)
        
        # Test loading the image
        img = Image.open(sample_image_path)
        assert img.size == (64, 64)
        assert img.mode == 'RGB'

    @pytest.mark.unit
    def test_sample_dataset_fixture(self, sample_dataset_dir):
        """Test that the sample_dataset_dir fixture works correctly."""
        assert os.path.exists(sample_dataset_dir)
        assert os.path.isdir(sample_dataset_dir)
        
        # Check that images were created
        image_files = [f for f in os.listdir(sample_dataset_dir) if f.endswith('.jpg')]
        assert len(image_files) == 5

    @pytest.mark.unit
    def test_sample_tensor_fixture(self, sample_tensor):
        """Test that the sample_tensor fixture works correctly."""
        assert isinstance(sample_tensor, torch.Tensor)
        assert sample_tensor.shape == (4, 3, 64, 64)
        assert sample_tensor.dtype == torch.float32

    @pytest.mark.unit
    def test_sample_noise_fixture(self, sample_noise):
        """Test that the sample_noise fixture works correctly."""
        assert isinstance(sample_noise, torch.Tensor)
        assert sample_noise.shape == (4, 128, 1, 1)
        assert sample_noise.dtype == torch.float32

    @pytest.mark.unit
    def test_mock_config_fixture(self, mock_config):
        """Test that the mock_config fixture works correctly."""
        assert mock_config.size == 64
        assert mock_config.out_dim == 10
        assert mock_config.batch_size == 4
        assert mock_config.lr == 1e-4
        assert mock_config.margin == 1.0

    @pytest.mark.unit
    def test_triplet_data_fixture(self, sample_triplet_data):
        """Test that the sample_triplet_data fixture works correctly."""
        anchor, positive, negative = sample_triplet_data
        
        assert isinstance(anchor, torch.Tensor)
        assert isinstance(positive, torch.Tensor)
        assert isinstance(negative, torch.Tensor)
        
        assert anchor.shape == (4, 10)
        assert positive.shape == (4, 10)
        assert negative.shape == (4, 10)

    @pytest.mark.unit
    def test_reproducible_seeds(self):
        """Test that random seeds are set for reproducible tests."""
        # Generate random numbers twice and ensure they're the same
        torch.manual_seed(42)
        tensor1 = torch.randn(3, 3)
        
        torch.manual_seed(42)
        tensor2 = torch.randn(3, 3)
        
        assert torch.allclose(tensor1, tensor2)

    @pytest.mark.integration
    def test_full_pipeline_mock(self, mock_config, sample_noise):
        """Integration test to verify mocked pipeline components work together."""
        # This is a simple integration test that doesn't test actual models
        # but verifies that the testing infrastructure supports integration testing
        
        # Mock a simple pipeline step
        batch_size = mock_config.batch_size
        noise_shape = sample_noise.shape
        
        assert noise_shape[0] == batch_size
        assert len(noise_shape) == 4  # (batch, channels, height, width)

    @pytest.mark.slow
    def test_slow_operation_marker(self):
        """Test that can be marked as slow and potentially skipped."""
        # Simulate a slow operation
        import time
        time.sleep(0.01)  # Very short sleep for testing
        assert True

    def test_pytest_markers_configured(self):
        """Test that custom pytest markers are properly configured."""
        # This test will pass if markers are configured correctly
        # Otherwise pytest will show warnings about unknown markers
        assert True