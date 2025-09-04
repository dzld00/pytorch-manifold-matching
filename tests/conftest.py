"""
Shared pytest fixtures for testing the image generation project.

This file contains common fixtures used across unit and integration tests.
"""

import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
import torch
import numpy as np
from PIL import Image


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_image_path(temp_dir):
    """Create a sample RGB image for testing."""
    image_path = os.path.join(temp_dir, "sample.jpg")
    # Create a 64x64 RGB image
    image = Image.new('RGB', (64, 64), color='red')
    image.save(image_path)
    return image_path


@pytest.fixture
def sample_dataset_dir(temp_dir):
    """Create a sample dataset directory with multiple images."""
    dataset_path = os.path.join(temp_dir, "dataset")
    os.makedirs(dataset_path)
    
    # Create multiple sample images
    for i in range(5):
        image_path = os.path.join(dataset_path, f"image_{i}.jpg")
        # Create images with different colors for variety
        colors = ['red', 'green', 'blue', 'yellow', 'purple']
        image = Image.new('RGB', (64, 64), color=colors[i])
        image.save(image_path)
    
    return dataset_path


@pytest.fixture
def mock_device():
    """Mock device for testing without requiring GPU."""
    with patch('torch.cuda.is_available', return_value=False):
        yield torch.device('cpu')


@pytest.fixture
def sample_tensor():
    """Create a sample tensor for testing."""
    return torch.randn(4, 3, 64, 64)


@pytest.fixture
def sample_noise():
    """Create sample noise tensor for generator testing."""
    return torch.randn(4, 128, 1, 1)


@pytest.fixture
def sample_batch_size():
    """Standard batch size for tests."""
    return 4


@pytest.fixture
def sample_embedding_dim():
    """Standard embedding dimension for ML network tests."""
    return 10


@pytest.fixture
def mock_config():
    """Mock configuration object for testing."""
    config = Mock()
    config.size = 64
    config.out_dim = 10
    config.batch_size = 4
    config.lr = 1e-4
    config.beta1 = 0
    config.beta2 = 0.9
    config.margin = 1.0
    config.alpha = 1e-2
    config.num_epochs = 1
    config.num_samples = 16
    config.data_path = "/tmp/test_data"
    config.name = "test_results"
    config.load_model = "no"
    config.g_model_name = ""
    config.ml_model_name = ""
    config.n_threads = 1
    return config


@pytest.fixture
def mock_model_state_dict():
    """Mock state dictionary for model loading tests."""
    return {
        'linear1.weight': torch.randn(2048, 128),
        'linear1.bias': torch.randn(2048),
        'model.0.conv1.weight': torch.randn(128, 128, 3, 3),
    }


@pytest.fixture(autouse=True)
def setup_reproducible_tests():
    """Set random seeds for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


@pytest.fixture
def mock_cuda():
    """Mock CUDA availability for testing."""
    with patch('torch.cuda.is_available', return_value=True):
        yield


@pytest.fixture
def mock_no_cuda():
    """Mock CUDA unavailability for testing."""
    with patch('torch.cuda.is_available', return_value=False):
        yield


@pytest.fixture
def sample_triplet_data():
    """Create sample anchor, positive, negative embeddings for triplet loss testing."""
    batch_size = 4
    embedding_dim = 10
    
    anchor = torch.randn(batch_size, embedding_dim)
    positive = anchor + 0.1 * torch.randn(batch_size, embedding_dim)  # Similar to anchor
    negative = torch.randn(batch_size, embedding_dim)  # Different from anchor
    
    return anchor, positive, negative


@pytest.fixture
def mock_image_files():
    """Mock image file list for dataset testing."""
    return [
        "/fake/path/image1.jpg",
        "/fake/path/image2.jpg", 
        "/fake/path/image3.png",
        "/fake/path/image4.jpeg",
    ]


@pytest.fixture
def results_dir(temp_dir):
    """Create a results directory for model saving tests."""
    results_path = os.path.join(temp_dir, "results")
    os.makedirs(results_path)
    return results_path