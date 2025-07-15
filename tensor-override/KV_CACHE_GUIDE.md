# K/V Cache Quantization Guide

K/V cache quantization significantly reduces memory usage for large context sizes, allowing you to use much larger contexts with the same GPU memory.

## Configuration Options

### Basic Configuration
```json
{
  "context_size": 132000,
  "kv_cache": {
    "type": "q4_0",
    "type_k": null,
    "type_v": null
  }
}
```

### Advanced Configuration (Separate K/V Types)
```json
{
  "context_size": 132000,
  "kv_cache": {
    "type": "f16",
    "type_k": "q4_0",
    "type_v": "q8_0"
  }
}
```

## Available K/V Cache Types

| Type | Bits | Memory Usage | Quality | Best For |
|------|------|--------------|---------|----------|
| `f32` | 32-bit | 4x | Highest | Research/debugging |
| `f16` | 16-bit | 2x | High | Default choice |
| `bf16` | 16-bit | 2x | High | Alternative to f16 |
| `q8_0` | 8-bit | 1x | Good | Balanced performance |
| `q4_0` | 4-bit | 0.5x | Acceptable | Maximum memory savings |
| `q4_1` | 4-bit | 0.5x | Acceptable | Alternative q4 |
| `q5_0` | 5-bit | 0.625x | Good | Compromise between q4/q8 |
| `q5_1` | 5-bit | 0.625x | Good | Alternative q5 |
| `iq4_nl` | 4-bit | 0.5x | Good | Improved q4 quality |

## Memory Usage for 132k Context (Kimi Model)

| Configuration | K/V Cache | Total Memory | Savings |
|---------------|-----------|--------------|---------|
| f16/f16 | 215.0 GB | 258.5 GB | 0% |
| q8_0/q8_0 | 107.5 GB | 129.5 GB | 50% |
| q4_0/q4_0 | 53.8 GB | 65.0 GB | 75% |
| q4_0/q8_0 | 80.6 GB | 97.3 GB | 62.5% |

## llama.cpp Usage

Add these flags to your llama-server command:

### Single Type
```bash
--cache-type-k q4_0 --cache-type-v q4_0
```

### Mixed Types
```bash
--cache-type-k q4_0 --cache-type-v q8_0
```

## Performance vs Memory Trade-offs

### Maximum Performance (f16)
- Highest quality
- ~215 GB for 132k context
- No quantization overhead

### Balanced (q8_0)
- Good quality
- ~107 GB for 132k context
- 50% memory savings
- Minimal performance impact

### Maximum Memory Savings (q4_0)
- Acceptable quality
- ~54 GB for 132k context
- 75% memory savings
- Some performance impact

### Recommended (q4_0/q8_0 mixed)
- Good balance
- ~81 GB for 132k context
- 62.5% memory savings
- K cache can be more aggressive since it's used less

## Integration with Tensor Override

The tensor optimizer automatically accounts for K/V cache memory when:
1. Calculating available GPU memory
2. Reserving context memory
3. Optimizing tensor placement

Your current configuration:
- Context: 132,000 tokens
- K/V cache: q4_0/q4_0
- Reserved memory: ~66 GB
- Available for tensors: ~58 GB across 4 GPUs

This leaves plenty of room for model tensors while supporting your large context size!