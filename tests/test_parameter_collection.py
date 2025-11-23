"""
Test suite for complete parameter collection.

Tests all new parameters added in the COMPLETE version of master_collector.py
"""
import pytest
import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.collectors.master_collector import (
    collect_evaluation_parameters,
    validate_metadata,
    get_parameter_summary
)


class TestInferenceParameters:
    """Test all inference parameters are collected."""
    
    def test_basic_inference_params_present(self):
        """Test that basic inference parameters exist with defaults."""
        config = {
            "run_id": "test_001",
            "model_id": "test-model"
        }
        metadata = collect_evaluation_parameters(config)
        
        # Existing params
        assert "temperature" in metadata
        assert "num_threads" in metadata
        assert "random_seed" in metadata
        assert "batch_size" in metadata
        
        # Check defaults
        assert metadata["temperature"] == 0.0
        assert metadata["num_threads"] == 4
        assert metadata["random_seed"] == 42
        assert metadata["batch_size"] == 1
    
    def test_new_inference_params_present(self):
        """Test that NEW inference parameters are present."""
        config = {
            "run_id": "test_002",
            "model_id": "test-model"
        }
        metadata = collect_evaluation_parameters(config)
        
        # NEW inference params
        assert "top_p" in metadata, "top_p parameter missing"
        assert "top_k" in metadata, "top_k parameter missing"
        assert "repetition_penalty" in metadata, "repetition_penalty parameter missing"
        assert "system_prompt_id" in metadata, "system_prompt_id parameter missing"
        
        # Check defaults
        assert metadata["top_p"] == 1.0
        assert metadata["top_k"] == 50
        assert metadata["repetition_penalty"] == 1.0
        assert metadata["system_prompt_id"] == "default"
    
    def test_custom_inference_values_override_defaults(self):
        """Test that custom values override defaults."""
        config = {
            "run_id": "test_003",
            "model_id": "test-model",
            "top_p": 0.9,
            "top_k": 40,
            "repetition_penalty": 1.1,
            "system_prompt_id": "custom_v1",
        }
        metadata = collect_evaluation_parameters(config)
        
        assert metadata["top_p"] == 0.9
        assert metadata["top_k"] == 40
        assert metadata["repetition_penalty"] == 1.1
        assert metadata["system_prompt_id"] == "custom_v1"


class TestHardwareParameters:
    """Test hardware parameters."""
    
    def test_inference_backend_present(self):
        """Test that inference_backend parameter is present."""
        config = {
            "run_id": "test_004",
            "model_id": "test-model"
        }
        metadata = collect_evaluation_parameters(config)
        
        assert "inference_backend" in metadata
        assert metadata["inference_backend"] == "lighteval"  # Default
    
    def test_custom_inference_backend(self):
        """Test custom inference backend."""
        config = {
            "run_id": "test_005",
            "model_id": "test-model",
            "inference_backend": "vllm"
        }
        metadata = collect_evaluation_parameters(config)
        
        assert metadata["inference_backend"] == "vllm"


class TestBusinessParameters:
    """Test business and metrics parameters."""
    
    def test_business_params_present(self):
        """Test that all business parameters are present."""
        config = {
            "run_id": "test_006",
            "model_id": "test-model"
        }
        metadata = collect_evaluation_parameters(config)
        
        # Existing
        assert "benchmark_category" in metadata
        assert "use_case_tags" in metadata
        
        # NEW
        assert "industry_vertical" in metadata
        assert "metric_type" in metadata
        assert "dataset_version" in metadata
        
        # Check defaults
        assert metadata["industry_vertical"] == "General"
        assert metadata["metric_type"] == "Accuracy"
        assert metadata["dataset_version"] == "latest"
    
    def test_custom_business_values(self):
        """Test custom business values."""
        config = {
            "run_id": "test_007",
            "model_id": "test-model",
            "industry_vertical": "Healthcare",
            "metric_type": "F1",
            "dataset_version": "v2.0",
        }
        metadata = collect_evaluation_parameters(config)
        
        assert metadata["industry_vertical"] == "Healthcare"
        assert metadata["metric_type"] == "F1"
        assert metadata["dataset_version"] == "v2.0"


class TestModelIdentityParameters:
    """Test model identity parameters (Platform API placeholders)."""
    
    def test_model_identity_placeholders_present(self):
        """Test that model identity placeholders exist."""
        config = {
            "run_id": "test_008",
            "model_id": "test-model"
        }
        metadata = collect_evaluation_parameters(config)
        
        # Platform API placeholders
        assert "pruning_method" in metadata
        assert "sparsity_ratio" in metadata
        assert "healing_applied" in metadata
        assert "calibration_dataset" in metadata
        assert "model_release_date" in metadata
        
        # Check defaults
        assert metadata["pruning_method"] == "Unknown"
        assert metadata["sparsity_ratio"] == 0.0
        assert metadata["healing_applied"] is False
        assert metadata["calibration_dataset"] == "Unknown"
        assert isinstance(metadata["model_release_date"], str)
    
    def test_custom_model_identity_values(self):
        """Test that custom model identity values are used."""
        config = {
            "run_id": "test_009",
            "model_id": "test-model",
            "pruning_method": "Magnitude",
            "sparsity_ratio": 0.5,
            "healing_applied": True,
            "calibration_dataset": "C4",
            "model_release_date": "2024-01-15",
        }
        metadata = collect_evaluation_parameters(config)
        
        assert metadata["pruning_method"] == "Magnitude"
        assert metadata["sparsity_ratio"] == 0.5
        assert metadata["healing_applied"] is True
        assert metadata["calibration_dataset"] == "C4"
        assert metadata["model_release_date"] == "2024-01-15"


class TestMetadataValidation:
    """Test metadata validation functions."""
    
    def test_validate_metadata_success(self):
        """Test that valid metadata passes validation."""
        config = {
            "run_id": "test_010",
            "model_id": "test-model",
            "num_threads": 4,
            "temperature": 0.0,
            "random_seed": 42,
            "batch_size": 1,
        }
        metadata = collect_evaluation_parameters(config)
        
        is_valid, missing = validate_metadata(metadata)
        
        assert is_valid
        assert len(missing) == 0
    
    def test_validate_metadata_missing_fields(self):
        """Test validation with missing fields."""
        incomplete_metadata = {
            "run_id": "test_011",
            # Missing required fields
        }
        
        is_valid, missing = validate_metadata(incomplete_metadata)
        
        assert not is_valid
        assert len(missing) > 0


class TestParameterSummary:
    """Test parameter summary generation."""
    
    def test_parameter_summary_generation(self):
        """Test that parameter summary is generated correctly."""
        config = {
            "run_id": "test_012",
            "model_id": "test-model",
        }
        metadata = collect_evaluation_parameters(config)
        
        summary = get_parameter_summary(metadata)
        
        # Check that summary contains expected categories
        assert "model_identity" in summary
        assert "inference" in summary
        assert "hardware" in summary
        assert "business" in summary
        
        # Check that counts are reasonable
        assert summary["inference"] >= 4  # At least the new params
        assert summary["business"] >= 3   # At least the new params
        assert summary["model_identity"] >= 1
    
    def test_parameter_summary_counts(self):
        """Test that parameter counts make sense."""
        config = {
            "run_id": "test_013",
            "model_id": "test-model",
        }
        metadata = collect_evaluation_parameters(config)
        
        summary = get_parameter_summary(metadata)
        total = sum(summary.values())
        
        # Should have collected many parameters
        assert total > 10, f"Only {total} parameters collected, expected more"


class TestBackwardsCompatibility:
    """Test that changes are backwards compatible."""
    
    def test_minimal_config_still_works(self):
        """Test that minimal config still produces valid metadata."""
        config = {
            "run_id": "test_014",
            "model_id": "test-model",
        }
        
        # Should not raise any errors
        metadata = collect_evaluation_parameters(config)
        
        # Should have all required fields
        assert "run_id" in metadata
        assert "model_id" in metadata
        assert "timestamp_utc" in metadata
        
        # Should have defaults for new fields
        assert "top_p" in metadata
        assert "inference_backend" in metadata
        assert "industry_vertical" in metadata
        assert "pruning_method" in metadata
    
    def test_old_config_format_compatible(self):
        """Test that old config format still works."""
        # Old format config (before new params were added)
        old_config = {
            "run_id": "test_015",
            "model_id": "legacy-model",
            "temperature": 0.7,
            "num_threads": 8,
            "benchmark_category": "Chat",
        }
        
        metadata = collect_evaluation_parameters(old_config)
        
        # Old params should be preserved
        assert metadata["temperature"] == 0.7
        assert metadata["num_threads"] == 8
        assert metadata["benchmark_category"] == "Chat"
        
        # New params should have defaults
        assert metadata["top_p"] == 1.0
        assert metadata["industry_vertical"] == "General"


class TestCompleteParameterCollection:
    """Integration test for complete parameter collection."""
    
    def test_all_parameters_collected(self):
        """Test that all expected parameters are collected."""
        config = {
            "run_id": "test_016",
            "model_id": "complete-test-model",
            "top_p": 0.95,
            "top_k": 40,
            "inference_backend": "vllm",
            "industry_vertical": "Healthcare",
        }
        
        metadata = collect_evaluation_parameters(config)
        
        # Count total parameters
        param_count = len(metadata)
        
        # Should have at least 20 parameters
        assert param_count >= 20, f"Only {param_count} parameters collected"
        
        # Check key categories are represented
        categories = [
            "run_id", "timestamp_utc", "model_id",  # Identity
            "temperature", "top_p", "top_k",  # Inference
            "inference_backend",  # Hardware
            "industry_vertical", "metric_type",  # Business
            "pruning_method", "sparsity_ratio",  # Model Identity
        ]
        
        for param in categories:
            assert param in metadata, f"Parameter {param} not found"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
