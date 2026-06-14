# Fast `exp` Approximation — Estrin Scheme with Cody-Waite Reduction

## Overview

The standard library `std::exp(x)` is accurate to ~1 ULP but expensive. For ML inference kernels we trade a small amount of accuracy for a large speedup using a **polynomial approximation with range reduction**.

The core idea: approximate `exp(x)` over a narrow interval where a low-degree polynomial suffices, then reconstruct the result via IEEE-754 bit manipulation.

**Accuracy**: < 3 ULP (max relative error ~1.08e-07). Verified across 10,000+ random values in `test/math_utils_test.cpp`.

---

## 1. Mathematical Identity

We rely on:

```
exp(x) = exp(n * ln(2) + r) = 2^n * exp(r)
```

where `n` is an integer and `r ∈ [-ln(2)/2, ln(2)/2] ≈ [-0.3466, 0.3466]`.

The reduction is:

```
n  = round(x / ln(2))
r  = x - n * ln(2)
```

Over this narrow interval, `exp(r)` is very close to 1 (≈ 0.71 to 1.41), so a degree-6 polynomial suffices for float precision.

---

## 2. Input Clamping

```cpp
x = _mm256_min_ps(x, _mm256_set1_ps(EXP_HI));   // 88.376
x = _mm256_max_ps(x, _mm256_set1_ps(EXP_LO));   // -88.376
```

Beyond these bounds, `exp(x)` overflows to `inf` (above ~88.37) or underflows to subnormal/zero (below ~-104). The clamp prevents the polynomial from producing garbage and avoids the range reduction going haywire for extreme inputs. Subnormal results are flushed to zero — safe for softmax since `exp(x - max_val)` is used (subnormals only occur for very negative arguments after subtracting the max).

---

## 3. Range Reduction (Cody-Waite Two-Step)

```cpp
__m256  x_log2e = _mm256_mul_ps(x, _mm256_set1_ps(LOG2EF));  // x * log2(e)
__m256i n_int   = _mm256_cvtps_epi32(x_log2e);               // round to nearest int
__m256  n       = _mm256_cvtepi32_ps(n_int);                  // back to float (exact)

x = _mm256_fnmadd_ps(n, _mm256_set1_ps(LN2_HI), x);         // x -= n * LN2_HI
x = _mm256_fnmadd_ps(n, _mm256_set1_ps(LN2_LO), x);         // x -= n * LN2_LO
```

### Why `cvtps_epi32` instead of `floor`?

`_mm256_cvtps_epi32` uses hardware **round-to-nearest** (banker's rounding), giving `n = round(x * log2(e))`. This ensures `r = x - n*ln2 ∈ [-ln2/2, ln2/2]` — the symmetric interval centered at 0 where the polynomial has the smallest error. The old implementation used `floor_ps` which skewed the interval to `[0, ln2)`, requiring asymmetric polynomial coefficients and losing 1 bit of accuracy.

### Why two-step subtraction (Cody-Waite)?

If we used a single `ln2` constant, computing `n * ln2` would lose the low bits of `ln2` because `n` is an integer and `ln2` has 24 bits of mantissa. The product `n * ln2` rounds to ~12 significant bits, and the subtraction `x - n*ln2` suffers catastrophic cancellation.

The Cody-Waite trick splits `ln2` into two parts:

```
LN2_HI = 6.9314575e-1f   (0x3f317200 — low 12 mantissa bits zeroed)
LN2_LO = 1.4286068e-6f   (the residual)
LN2_HI + LN2_LO = ln(2) to < 1 ULP
```

`n * LN2_HI` is **exactly representable** (because `LN2_HI` has trailing zeros), so the first subtraction is exact. The second subtraction (`- n * LN2_LO`) corrects the remaining error. This gives `r` accurate to full float precision (~24 bits).

---

## 4. Polynomial Approximation (Estrin's Scheme)

We approximate `exp(r) ≈ P(r)` using a **degree-6 minimax polynomial** fitted via the Remez algorithm:

```
P(r) = 1 + r + r²/2! + r³/6 + r⁴/24 + r⁵/120 + r⁶/720
```

Coefficients (minimax-optimal floats):

```
P0 = 1.0000000000000000f
P1 = 1.0000000000000000f
P2 = 4.9999999999940024e-1f   (≈ 1/2!)
P3 = 1.6666666664684413e-1f   (≈ 1/3!)
P4 = 4.1666666647390810e-2f   (≈ 1/4!)
P5 = 8.3333358523025905e-3f   (≈ 1/5!)
P6 = 1.3888889927481680e-3f   (≈ 1/6!)
```

### Why Estrin over Horner?

Horner's method evaluates left-to-right with 7 sequential FMAs:

```
y = P6
y = y*x + P5    (FMA 1)
y = y*x + P4    (FMA 2)
y = y*x + P3    (FMA 3)
y = y*x + P2    (FMA 4)
y = y*x + P1    (FMA 5)
y = y*x + P0    (FMA 6)
```

Each FMA depends on the previous result. With FMA latency of ~4 cycles, the critical path is **7 × 4 = 28 cycles**.

Estrin's scheme restructures the evaluation to expose **instruction-level parallelism (ILP)**:

```
Level 0  (3 independent FMAs — all issue same cycle):
  p01 = P1*x + P0
  p23 = P3*x + P2
  p45 = P5*x + P4

Level 1  (2 independent FMAs):
  p0123 = p23*x² + p01
  p4567 = P6*x²  + p45

Level 2  (1 FMA):
  result = p4567*x⁴ + p0123
```

The critical path depth is only **4 FMAs (~16 cycles)** — a ~43% reduction from Horner's 7. The trade-off is computing `x²` and `x⁴` upfront (2 extra multiplies), but these are independent and issue in parallel with the Level 0 FMAs.

```cpp
__m256 x2 = _mm256_mul_ps(x, x);
__m256 x4 = _mm256_mul_ps(x2, x2);

__m256 p01 = _mm256_fmadd_ps(_mm256_set1_ps(P1), x,  _mm256_set1_ps(P0));
__m256 p23 = _mm256_fmadd_ps(_mm256_set1_ps(P3), x,  _mm256_set1_ps(P2));
__m256 p45 = _mm256_fmadd_ps(_mm256_set1_ps(P5), x,  _mm256_set1_ps(P4));

__m256 p0123 = _mm256_fmadd_ps(p23, x2, p01);
__m256 p4567 = _mm256_fmadd_ps(_mm256_set1_ps(P6), x2, p45);

return _mm256_fmadd_ps(p4567, x4, p0123);
```

---

## 5. Scaling by `2^n` via IEEE-754 Bit Manipulation

Now we have `exp(r) ≈ P(r)` and need to multiply by `2^n` to get `exp(x)`.

Instead of computing `2^n` as a float via `powf`, we **directly construct the float** by manipulating its exponent field.

### IEEE 754 Single-Precision Recap

```
31 30      23 22                    0
┌──┬─────────┬───────────────────────┐
│ S │  E+127 │       M               │
└──┴─────────┴───────────────────────┘
sign  exponent      mantissa
      (biased)
```

Value = `(-1)^S * (1.M) * 2^(E - 127)`

For `2^n`:
- Sign = 0
- Biased exponent = `n + 127`
- Mantissa = 0

So `2^n` as a float is: `(n + 127) << 23`

### The code:

```cpp
__m256i pow2n = _mm256_slli_epi32(
                    _mm256_add_epi32(n_int, _mm256_set1_epi32(0x7f)), 23);
return _mm256_mul_ps(y, _mm256_castsi256_ps(pow2n));
```

Step-by-step:

1. **`add_epi32(0x7f)`** — `n + 127`. The biased exponent for `2^n`.
2. **`slli_epi32(23)`** — shift left by 23, placing the exponent in bits [30:23]. The mantissa bits are zero → `1.0 * 2^n`.
3. **`castsi256_ps`** — reinterpret the 8 integers as 8 floats `2^n`, without any conversion (same bit pattern).
4. **`mul_ps`** — `exp(r) * 2^n = exp(x)`.

### Visual example: `x = 2.0`

```
n = round(2.0 * 1.442695) = round(2.88539) = 3
r = 2.0 - 3 * 0.693147 = 2.0 - 2.07944 = -0.07944
exp(r) ≈ P(-0.07944) ≈ 0.9236
2^n = 2^3 = 8
exp(2.0) ≈ 0.9236 * 8 = 7.389
            (true exp(2.0) = 7.389056...)
```

`2^3 = 8` in IEEE 754: exponent = `3 + 127 = 130 = 0x82`, so bits = `0x82 << 23 = 0x41000000`.

---

## 6. Instruction Count

The entire approximation uses **~15 SIMD instructions** per 8 floats:

| Step | Instructions |
|---|---|
| Clamp | `min_ps`, `max_ps` |
| Range reduce | `mul_ps`, `cvtps_epi32`, `cvtepi32_ps`, `fnmadd_ps` × 2 |
| Polynomial (Estrin) | `mul_ps` × 2 (x², x⁴), `fmadd_ps` × 5 |
| Scale | `add_epi32`, `slli_epi32`, `castsi256_ps`, `mul_ps` |
| **Total** | **~15 instr / 8 elems ≈ 1.9 instr/elem** |

Versus `std::exp` which calls into libm (~hundreds of instructions with branches, special-case checks, and a full 64-bit reduction).

The Estrin scheme achieves **1.5× better ILP** than Horner at the cost of 2 extra multiplies, reducing the critical path from ~28 cycles to ~16 cycles.

---

## 7. Accuracy

- Maximum ULP error: **< 3 ULP** across the normal range `[-87.3, 87.3]`
- Maximum relative error: **~1.08e-07** (~22 bits of mantissa accuracy)
- Subnormal inputs (`|x| > 88.37`): flushed to 0 (safe for softmax — subtract-max ensures inputs are in `[-88, 0]`)

This is roughly **float precision** (23-bit mantissa), so the approximation is indistinguishable from `std::exp` when the result is stored as a `float`. For transformer inference, this accuracy level is standard — softmax outputs at float precision are used as sampling weights, and 1e-7 relative error does not affect the ranking.

### Why accuracy matters for softmax

Softmax computes `exp(x_i - max) / sum(exp(x_j - max))`. The subtraction `x_i - max` maps the largest element to `exp(0) = 1.0` and all others to smaller values. The critical requirement is that the **relative ordering** of softmax outputs is preserved (for sampling/beam search). With < 3 ULP error, the ranking is bit-identical to `std::exp` for all practical input ranges.

---

## 8. References

- Cody & Waite, "Software Manual for Elementary Functions" (1980) — Cody-Waite range reduction
- SLEEF 3.6 (Shibatch) — minimax polynomial coefficients and Estrin evaluation
- Schraudolph (1999) — IEEE-754 exponent field trick for `2^n`
- Estrin (1960) — polynomial evaluation with exposed ILP
