# RDO-PTQ Implementation

Post-training quantization with mixed-precision for neural network compression.

## Files

- `info.ipynb` - Code with three implementations


## What's Implemented

### From Base Paper (Shi et al. 2022)

- Layer-wise quantization
- Per-channel weight quantization  
- Rate-distortion optimization (simplified)

### What's Missing from Base Paper

- Adaptive rounding (using standard rounding)
- Full entropy estimation (using log approximation)
- Per-channel activations (using per-tensor)
- Bias rescaling

### Our Extensions

- **Mixed-precision**: Automatic 4/6/8-bit per layer
- **Sensitivity analysis**: Measures layer importance
- **Block-wise**: Jointly optimize 2-3 layers together
- **Hessian guidance**: Uses gradient variance

## Three Versions

**BasicRDOPTQ** - Core implementation (150 iters)
- Result: 4.0× compression

**ProductionRDOPTQ** - Full features (300 iters)  
- Result: 7.06× compression

**FixedRDOPTQ** - Fast version (50 iters)
- Result: 4.29× compression

## Running the Code

Open `info.ipynb` and run all cells in order. Each section shows one version with results.



## Authors
Dev Anant Pushkar  (202211017) 

Daivik Hirpara     (202211028)

Rajat Kumar Thakur (202211070)

IIIT Vadodara

## Reference

Based on: Shi et al., "Rate-Distortion Optimized Post-Training Quantization for Learned Image Compression", arXiv:2211.02854, 2022
