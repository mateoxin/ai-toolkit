# Multiple LoRA Support in AI Toolkit

AI Toolkit now supports loading and fusing multiple LoRA (Low-Rank Adaptation) models simultaneously when generating images. This allows you to combine the effects of several LoRAs in a single generation pass, each with its own weight.

## Features

- **Multiple LoRA Loading**: Load 2 or more LoRAs at once
- **Individual Weights**: Set different weights for each LoRA (0.0 to 1.0+)
- **Backward Compatibility**: Single LoRA configurations still work exactly as before
- **Support for FLUX and SDXL**: Works with both FLUX.1 models and SDXL models
- **Low VRAM Mode**: Optimized loading for FLUX models in low VRAM environments

## Configuration

### Single LoRA (Backward Compatible)

The traditional single LoRA configuration continues to work:

```yaml
model:
  name_or_path: black-forest-labs/FLUX.1-dev
  is_flux: true
  quantize: true
  dtype: float16
  lora_path: /path/to/your/lora.safetensors
  lora_weight: 1.0  # Optional, defaults to 1.0
```

### Multiple LoRAs (New Feature)

To use multiple LoRAs, replace `lora_path` with `lora_paths`:

```yaml
model:
  name_or_path: black-forest-labs/FLUX.1-dev
  is_flux: true
  quantize: true
  dtype: float16
  lora_paths:
    - path: /path/to/character_lora.safetensors
      weight: 1.0
    - path: /path/to/style_lora.safetensors
      weight: 0.7
    - path: /path/to/quality_lora.safetensors
      weight: 0.5
```

Each entry in `lora_paths` is a dictionary with:
- `path`: (Required) Path to the LoRA `.safetensors` file (local or S3)
- `weight`: (Optional) Float value for LoRA strength (default: 1.0)

## How It Works

1. **Loading Phase**: During model loading, all LoRAs are loaded sequentially
2. **Weight Application**: Each LoRA has its individual weight applied using `set_adapters()`
3. **Fusing**: All LoRAs are fused into the base model using `fuse_lora()`
4. **Memory Cleanup**: After fusing, LoRA weights are unloaded to save memory

The final effect is a combination of all LoRAs, each contributing according to its weight.

## Recommended Weight Values

Based on testing and common use cases:

| LoRA Type | Recommended Weight | Description |
|-----------|-------------------|-------------|
| Main subject/character | 0.8 - 1.0 | Primary LoRA for the main subject |
| Artistic style | 0.5 - 0.7 | Style transfer or artistic effects |
| Quality enhancement | 0.3 - 0.5 | Image quality improvements |
| Subtle effects | 0.2 - 0.4 | Minor adjustments or tweaks |

## Examples

### Example 1: Character + Style

Combine a character LoRA with a style LoRA:

```yaml
lora_paths:
  - path: /models/john_doe_character.safetensors
    weight: 1.0  # Full strength for the main character
  - path: /models/anime_style.safetensors
    weight: 0.6  # Moderate style influence
```

### Example 2: Multiple Styles Blending

Blend multiple artistic styles:

```yaml
lora_paths:
  - path: /models/watercolor.safetensors
    weight: 0.7
  - path: /models/impressionist.safetensors
    weight: 0.5
```

### Example 3: S3 Storage

Use LoRAs stored on S3:

```yaml
lora_paths:
  - path: s3://my-bucket/loras/character_v2.safetensors
    weight: 0.9
  - path: /local/path/style_lora.safetensors
    weight: 0.6
```

## Performance Notes

- **FLUX Models**: In `low_vram` mode, LoRAs are fused in parts (double/single transformer blocks) to avoid OOM errors
- **Loading Time**: Multiple LoRAs increase model loading time proportionally, but generation speed remains similar
- **Memory**: After fusing, LoRAs are removed from memory, so VRAM usage is comparable to a single fused LoRA

## Troubleshooting

### Out of Memory Errors

**Solution**: Try one or more of:
- Reduce the number of LoRAs
- Lower the weights (doesn't save memory, but might help with stability)
- Enable `low_vram: true` for FLUX models in your model config
- Use models with lower VRAM requirements

### Unexpected Results

**Solution**: LoRAs may conflict with each other. Try:
- Test each LoRA individually first to understand its effects
- Adjust weights (start with lower weights like 0.5 and increase gradually)
- Use fewer LoRAs
- Check that LoRAs are compatible with your base model

### LoRA Not Loading

**Solution**: Verify:
- File paths are correct and files exist
- For S3 paths: credentials are configured and paths are accessible
- LoRA files are `.safetensors` format
- LoRAs are compatible with your model (FLUX LoRAs for FLUX, SDXL LoRAs for SDXL)

## Technical Implementation

The implementation modifies two key files:

1. **`toolkit/config_modules.py`**: 
   - Added `lora_paths` field to `ModelConfig`
   - Automatic conversion of `lora_path` to `lora_paths` format
   - Validation of LoRA configuration

2. **`toolkit/stable_diffusion_model.py`**:
   - Modified `load_model()` to iterate through multiple LoRAs
   - Support for FLUX (with `low_vram` mode)
   - Support for SDXL and SD1.5
   - Individual weight application using `set_adapters()`

## Backward Compatibility

The implementation is fully backward compatible:

- Old configs using `lora_path` continue to work
- `lora_path` is automatically converted to `lora_paths` format internally
- `lora_weight` parameter is respected when converting from old format
- No breaking changes to existing functionality

## See Also

- Example configs in `config/examples/`:
  - `generate_single_lora_flux.yaml` - Single LoRA for FLUX
  - `generate_multiple_loras_flux.yaml` - Multiple LoRAs for FLUX
  - `generate_multiple_loras_sdxl.yaml` - Multiple LoRAs for SDXL


