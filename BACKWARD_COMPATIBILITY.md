# Backward Compatibility Guide

## Overview

The model has been renamed from `Mantis8M` to `MantisV1` and the internal transformer module has been renamed from `ViTUnit` to `TransformerUnit`. Full backward compatibility is maintained for existing code and checkpoints.

## Usage Examples

### New Code (Recommended)
```python
from mantis.architecture import MantisV1, TransformerUnit

# Create new model
model = MantisV1(seq_len=512, hidden_dim=256)

# Load from HuggingFace (even old checkpoints with ViTUnit inside)
model = MantisV1.from_pretrained("paris-noah/Mantis-8M")

# Access the transformer unit with new name
transf_unit = model.transf_unit
```

### Old Code (Still Works)
```python
from mantis.architecture import Mantis8M, ViTUnit

# Create old model
model = Mantis8M(seq_len=512, hidden_dim=256)

# Load from HuggingFace
model = Mantis8M.from_pretrained("paris-noah/Mantis-8M")

# Access the transformer unit with old name (property alias)
vit_unit = model.vit_unit  # This is an alias for model.transf_unit
```

## How It Works

### Class Aliases
- `Mantis8M` is an alias for `MantisV1`
- `ViTUnit` is an alias for `TransformerUnit`

### Property Aliases
- `MantisV1` (formerly `Mantis8M`) has a `vit_unit` property that returns `self.transf_unit`
- The property is read/write, so old code using `model.vit_unit = ...` still works

### Checkpoint Loading
The `MantisV1.from_pretrained()` method automatically handles old checkpoints:
1. Loads the checkpoint from HuggingFace
2. Renames state dict keys from `vit_unit.*` to `transf_unit.*`
3. Handles module references for seamless loading

## Migration Path

You don't need to update your code immediately. Both old and new names work interchangeably. However, for new code, consider using `MantisV1` instead of `Mantis8M` to ensure your code is future-proof as the library evolves.
