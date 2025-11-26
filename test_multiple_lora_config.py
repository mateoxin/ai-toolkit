#!/usr/bin/env python3
"""
Test script to verify multiple LoRA configuration parsing
"""

import sys
import yaml
from toolkit.config_modules import ModelConfig

def test_single_lora_backward_compatibility():
    """Test that single lora_path is converted to lora_paths format"""
    print("Testing single LoRA backward compatibility...")
    
    config_dict = {
        'name_or_path': 'black-forest-labs/FLUX.1-dev',
        'is_flux': True,
        'lora_path': '/path/to/single_lora.safetensors',
        'lora_weight': 0.8
    }
    
    model_config = ModelConfig(**config_dict)
    
    assert model_config.lora_paths is not None, "lora_paths should be set"
    assert len(model_config.lora_paths) == 1, "Should have 1 LoRA"
    assert model_config.lora_paths[0]['path'] == '/path/to/single_lora.safetensors'
    assert model_config.lora_paths[0]['weight'] == 0.8
    assert model_config.lora_path is None, "lora_path should be None after conversion"
    
    print("✓ Single LoRA backward compatibility test passed")

def test_multiple_loras():
    """Test multiple LoRAs configuration"""
    print("\nTesting multiple LoRAs...")
    
    config_dict = {
        'name_or_path': 'black-forest-labs/FLUX.1-dev',
        'is_flux': True,
        'lora_paths': [
            {'path': '/path/to/lora1.safetensors', 'weight': 1.0},
            {'path': '/path/to/lora2.safetensors', 'weight': 0.7},
            {'path': '/path/to/lora3.safetensors', 'weight': 0.5}
        ]
    }
    
    model_config = ModelConfig(**config_dict)
    
    assert model_config.lora_paths is not None, "lora_paths should be set"
    assert len(model_config.lora_paths) == 3, "Should have 3 LoRAs"
    assert model_config.lora_paths[0]['weight'] == 1.0
    assert model_config.lora_paths[1]['weight'] == 0.7
    assert model_config.lora_paths[2]['weight'] == 0.5
    
    print("✓ Multiple LoRAs test passed")

def test_default_weight():
    """Test that weight defaults to 1.0 if not specified"""
    print("\nTesting default weight...")
    
    config_dict = {
        'name_or_path': 'black-forest-labs/FLUX.1-dev',
        'is_flux': True,
        'lora_paths': [
            {'path': '/path/to/lora.safetensors'}  # No weight specified
        ]
    }
    
    model_config = ModelConfig(**config_dict)
    
    assert model_config.lora_paths[0]['weight'] == 1.0, "Weight should default to 1.0"
    
    print("✓ Default weight test passed")

def test_yaml_parsing():
    """Test parsing from YAML config"""
    print("\nTesting YAML parsing...")
    
    yaml_content = """
model:
  name_or_path: black-forest-labs/FLUX.1-dev
  is_flux: true
  lora_paths:
    - path: /path/to/lora1.safetensors
      weight: 0.9
    - path: /path/to/lora2.safetensors
      weight: 0.6
"""
    
    config = yaml.safe_load(yaml_content)
    model_config = ModelConfig(**config['model'])
    
    assert len(model_config.lora_paths) == 2
    assert model_config.lora_paths[0]['weight'] == 0.9
    assert model_config.lora_paths[1]['weight'] == 0.6
    
    print("✓ YAML parsing test passed")

def test_validation_missing_path():
    """Test that validation catches missing path"""
    print("\nTesting validation (missing path)...")
    
    try:
        config_dict = {
            'name_or_path': 'black-forest-labs/FLUX.1-dev',
            'is_flux': True,
            'lora_paths': [
                {'weight': 0.8}  # Missing 'path'
            ]
        }
        
        model_config = ModelConfig(**config_dict)
        print("✗ Should have raised ValueError for missing path")
        sys.exit(1)
    except ValueError as e:
        if "must have a 'path' key" in str(e):
            print("✓ Validation test passed (missing path caught)")
        else:
            print(f"✗ Wrong error message: {e}")
            sys.exit(1)

def test_validation_invalid_type():
    """Test that validation catches invalid type"""
    print("\nTesting validation (invalid type)...")
    
    try:
        config_dict = {
            'name_or_path': 'black-forest-labs/FLUX.1-dev',
            'is_flux': True,
            'lora_paths': [
                '/path/to/lora.safetensors'  # Should be dict, not string
            ]
        }
        
        model_config = ModelConfig(**config_dict)
        print("✗ Should have raised ValueError for invalid type")
        sys.exit(1)
    except ValueError as e:
        if "must be a dict" in str(e):
            print("✓ Validation test passed (invalid type caught)")
        else:
            print(f"✗ Wrong error message: {e}")
            sys.exit(1)

def main():
    print("=" * 60)
    print("Multiple LoRA Configuration Tests")
    print("=" * 60)
    
    try:
        test_single_lora_backward_compatibility()
        test_multiple_loras()
        test_default_weight()
        test_yaml_parsing()
        test_validation_missing_path()
        test_validation_invalid_type()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())


