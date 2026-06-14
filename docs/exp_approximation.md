# Fast `exp` Approximation — Mathematics & Bit Manipulation

## Overview

The standard library `std::exp(x)` is accurate to ~1 ULP but expensive. For ML inference kernels we can trade a small amount of accuracy for a large speedup by using a **polynomial approximation with range reduction**.

The idea: instead of approximating `exp(x)` directly over the full float domain (where polynomials diverge quickly), we reduce the input to a small interval where a low-degree polynomial suffices, then reconstruct the result.

---

## 1. Mathematical Identity

We rely on:

```
exp(x) = exp(k * ln(2) + r) = exp(ln(2^k)) * exp(r) = 2^k * exp(r)
```

where `k` is an integer and `r ∈ [-ln(2)/2, ln(2)/2] ≈ [-0.3466, 0.3466]`.

The reduction is:

```
k  = round(x / ln(2)) = floor(x * log2(e) + 0.5)
r  = x - k * ln(2)
```

`log2(e) ≈ 1.442695` is the key constant — it converts the base: `exp(x) = 2^(x * log2(e))`.

### Why reduce to `[-0.3466, 0.3466]`?

Over this narrow interval, `exp(r)` is very close to 1 (≈ 0.71 to 1.41), so a degree-6 polynomial is enough for ~22 bits of accuracy (~float precision).

---

## 2. Range Reduction (lines 45-48)

```cpp
__m256 fx = _mm256_fmadd_ps(x, _mm256_set1_ps(LOG2EF), _mm256_set1_ps(0.5f));
fx = _mm256_floor_ps(fx);
```

- `LOG2EF = 1.442695f = log2(e)`
- `x * log2(e) + 0.5` then floor gives `round(x * log2(e))` = the integer `k`

```cpp
x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(LN2_HI), x);
x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(LN2_LO), x);
```

- `FNMADD(a,b,c)` computes `-a*b + c`
- So this is: `x = x - fx * LN2_HI - fx * LN2_LO = x - k * ln(2) = r`
- The **HI/LO split** of `ln(2)` gives extra precision via two FMA instructions (one for the high bits, one for the low), avoiding rounding error in the subtraction.

Constants:
```
LN2_HI = 6.93145751953125e-1f   (= 0x3F317200, 12 significand bits)
LN2_LO = 1.428606820309417e-6f  (= 0x36BFFE20, the residual)
            sum = 0.69314718055995... ≈ ln(2)
```

---

## 3. Polynomial Approximation (lines 49-55)

We approximate `exp(r) ≈ P(r)` using a **degree-6 minimax polynomial** from [Schraudolph 1999 / Niclas Schunning's SLEEF library](https://github.com/shibatch/sleef):

```
P(r) = P0 + P1*r + P2*r^2 + P3*r^3 + P4*r^4 + P5*r^5 + P6*r^6
```

Coefficients:

```
P0 = 1.0000000754895593f
P1 = 6.931472284335791e-1f
P2 = 2.402264895851545e-1f
P3 = 5.550332399887598e-2f
P4 = 9.618038735174234e-3f
P5 = 1.339045359498462e-3f
P6 = 1.540357332908606e-4f
```

Note `P0 ≈ 1.0`, `P1 ≈ ln(2) ≈ 0.6931` — the first two terms match the Taylor series `exp(r) = 1 + r + r²/2! + ...`, so higher coefficients correct for the truncation error.

### Horner's Method with FMA

Evaluated left-to-right using FMA (fused multiply-add), which is **both faster and more accurate** than separate mul+add:

```
y = P6
y = y * r + P5    // via FMA
y = y * r + P4    // via FMA
...
y = y * r + P0
```

Each step: `_mm256_fmadd_ps(y, x, coeff)` computes `y*x + coeff` in one instruction with a single rounding.

---

## 4. Scaling by `2^k` via IEEE 754 Bit Manipulation (lines 56-59)

Now we have `exp(r) ≈ P(r)` and need to multiply by `2^k` to get `exp(x)`.

Instead of a multiply (which would require computing `2^k` as a float via `powf`), we **directly construct the float** by manipulating its exponent field.

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

For `2^k`:
- Sign = 0
- Biased exponent = `k + 127`
- Mantissa = 0

So `2^k` as a float is: `(k + 127) << 23`

### The code:

```cpp
__m256i imm0 = _mm256_cvttps_epi32(fx);   // (1) convert float k to int
imm0 = _mm256_add_epi32(imm0, _mm256_set1_epi32(0x7f));  // (2) add 127 bias
imm0 = _mm256_slli_epi32(imm0, 23);        // (3) shift into exponent position
return _mm256_mul_ps(y, _mm256_castsi256_ps(imm0));  // (4) multiply
```

Step-by-step:

1. **`cvttps_epi32`** — truncates `fx` (the float `k`) to a 32-bit integer. `k` is now in an integer register.
2. **`add_epi32(0x7f)`** — `k + 127`. The biased exponent for `2^k`.
3. **`slli_epi32(imm0, 23)`** — shift left by 23, placing the exponent in bits [30:23]. The mantissa bits are zero → `1.0 * 2^k`.
4. **`castsi256_ps`** — reinterpret the 8 integers as 8 floats `2^k`, without any conversion (same bit pattern).
5. **`mul_ps`** — `exp(r) * 2^k = exp(x)`.

### Visual example: `x = 2.0`

```
k = floor(2.0 * 1.442695 + 0.5) = floor(2.88539 + 0.5) = floor(3.38539) = 3
r = 2.0 - 3 * 0.693147 = 2.0 - 2.07944 = -0.07944
exp(r) ≈ P(-0.07944) ≈ 0.9236
2^k = 2^3 = 8
exp(2.0) ≈ 0.9236 * 8 = 7.389
            (true exp(2.0) = 7.389056...)
```

`2^3 = 8` in IEEE 754: exponent = `3 + 127 = 130 = 0x82`, so bits = `0x82 << 23 = 0x41000000`.

---

## 5. Clamping (lines 43-44)

```cpp
x = _mm256_min_ps(x, _mm256_set1_ps(EXP_HI));
x = _mm256_max_ps(x, _mm256_set1_ps(EXP_LO));
```

`EXP_HI = 88.376`, `EXP_LO = -88.376`.

Beyond these bounds, `exp(x)` overflows to `inf` (above ~88.37) or underflows to `0` (below ~-103.97). The clamp prevents the polynomial from producing garbage and avoids the range reduction going haywire for extreme inputs. (`-88.376` is slightly tighter than the true underflow boundary because the polynomial accuracy degrades near the edges.)

---

## 6. Instruction Count

The entire approximation uses **12 SIMD instructions** per 8 floats:

| Step | Instructions |
|---|---|
| Clamp | `min_ps`, `max_ps` |
| Range reduce | `fmadd_ps`, `floor_ps`, `fnmadd_ps`, `fnmadd_ps` |
| Polynomial (Horner-FMA) | `fmadd_ps` × 6 |
| Scale | `cvttps_epi32`, `add_epi32`, `slli_epi32`, `castsi256_ps`, `mul_ps` |
| **Total** | **12 instr / 8 elems = 1.5 instr/elem** |

Versus `std::exp` which calls into libm (~hundreds of instructions with branches, special-case checks, and a full 64-bit reduction).

---

## 7. Expected Accuracy

- Maximum relative error: ~**1.5 × 10⁻⁷** (~22 bits)
- Median relative error: ~**5 × 10⁻⁸**
- This is roughly **float precision** (23-bit mantissa), so it's indistinguishable from `std::exp` when the result is stored as a `float`.

For transformer inference, this accuracy level is standard — softmax outputs at float precision are used as sampling weights, and 1e-7 relative error does not affect the ranking.
