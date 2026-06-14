# Notes

## Scalar operations 

- Performing actions *1* step at a time.

```cpp
int n = 1000000;
vector<float> data(n, 1.0f);
float s = 0;
for (size_t i = 0; i < n; i++) s += data[i];
```

- Each iteration would load one `float` from memory to CPU (ALU) and add it to `s`.
- Thus, a million separate addition operations would have be done serially.
- The scalar loop processes 1M floats in 1M iterations. 

---

## SIMD (Single Instruction, Multiple Data)

- Using `AVX2` (Advanced Vector Extension 2).
- In SIMD C++, an **intrinsic** is a built-in compiler function that gives direct access to specific CPU instructions.

- SIMD registers are **256 bits** wide, called `YMM` registers.
- 32 * 8 = 256. 

> Thus, a 256 bits register can hold 8 floats simultaneously.

- SIMD is a class of parallel computation where one instruction operates on multiple data elements simultaneously. Instead of processing one float at a time, a single AVX2 instruction can process 8 floats in parallel.

- A register is defined by : `__m256 <name>`.
- All intrinsics functions are called using functions with naming : `_mm<width>_<operation>_<suffix>`

- `<width>` = 256 (AVX/AVX2)
- `<operation>` = functional name (add, mul, load, shuffle, etc.)
- `<suffix>` = element type:
  - ps  = packed single (float32)
  - pd  = packed double (float64)
  - ss  = scalar single (float32, lowest lane only)
  - sd  = scalar double (float64, lowest lane only)
  - epi8  / epi16 / epi32 / epi64  = signed integer (8/16/32/64-bit)
  - epu8  / epu16 / epu32 / epu64  = unsigned integer
  - si256 = opaque 256-bit integer (e.g. cast operations)

---

## Key Concepts

- Vector width: number of bits in the register (128-bit XMM, 256-bit YMM, 512-bit ZMM)
- Lane: a single element slot within a vector
- Packed vs Scalar: packed ops act on all lanes; scalar ops act on the lowest lane only
- Alignment: memory addresses should be aligned to the vector width for best performance (32-byte for AVX2)
- Throughput vs Latency: throughput = how many per cycle (with pipelining), latency = cycles until result is ready

---

## Aligned & Unaligned memory

```txt
Aligned (address % 32 == 0):
RAM:    [0x...20][0x...21]...[0x...3F]  ← one clean 32-byte chunk
SIMD:   ████████████████████████████████  loads in one shot

Unaligned (address % 32 != 0):
RAM:    ...[0x...1C][0x...1D][0x...1E][0x...1F] | [0x...20]...
                                               ^boundary crossed
SIMD:   must stitch two chunks together (older CPUs crash here)
```

---

## Cache-Memory Hierarchy

```text
========================================================================
                          CORE 0 (Example)
+----------------------------------------------------------------------+
|  [ Execution Units / SIMD Registers ]                                |
|   e.g., YMM (256-bit) / ZMM (512-bit)                                |
+----------------------------------------------------------------------+
         ▲                                      ▲
   ~0.5 ns Latency                        ~0.5 ns Latency
         ▼                                      ▼
+-----------------------+              +-----------------------+
|   L1 Instruction Cache|              |      L1 Data Cache    |  <-- Handled in 
|         (32 KB)       |              |   (32 KB - 48 KB)     |      64-byte chunks
+-----------------------+              +-----------------------+
         ▲                                      ▲
         +------------------+-------------------+
                            ▼
                    ~1.0 - 1.5 ns Latency
                            ▼
               +-------------------------+
               |     L2 Unified Cache    |  <-- Usually private 
               |    (512 KB - 2 MB)      |      to each core
               +-------------------------+
                            ▲
============================|===========================================
                            ▼  ~4 - 5 ns Latency (Interconnect / Mesh Bus)
+----------------------------------------------------------------------+
|                        L3 Cache (Last Level Cache)                   |
|                        Shared across ALL cores                       |
|                             (4 MB - 96 MB+)                          |
+----------------------------------------------------------------------+
                            ▲
                            ▼  ~60 - 100 ns Latency (Memory Controller)
+----------------------------------------------------------------------+
|                         System RAM (DDR4 / DDR5)                     |
|                             (8 GB - 128 GB+)                         |
+----------------------------------------------------------------------+
```

### Cache Line

A cache line is a fixed-size block of memory, exactly 64 bytes on virtually all modern x86 (Intel/AMD) and ARM processors.

- When code requests a single 4-byte floating-point number from RAM, the CPU doesn't just fetch those 4 bytes. It fetches the entire 64-byte chunk of memory containing that float and copies it into the L1 Data Cache.

---

## SIMD Register Layout

```text
__m256 register (256 bits wide)
┌────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
│ lane 0 │ lane 1 │ lane 2 │ lane 3 │ lane 4 │ lane 5 │ lane 6 │ lane 7 │
│  32bit │  32bit │  32bit │  32bit │  32bit │  32bit │  32bit │  32bit │
└────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘
```

---

## Unpack

Using `SSE` (128 bit) registers.

```
Lane 3       Lane 2       Lane 1       Lane 0
[ 32 bits ]  [ 32 bits ]  [ 32 bits ]  [ 32 bits ]
127      96  95       64  63       32  31        0  <- Bit positions
```

- These 128 bits are treated as **4 distinct 32-bit floating-point lanes**.

- `_mm_unpacklo_ps` and `_mm_unpackhi_ps`.
  - these are like interleaving or *zipper-merging*.
  - The instruction always starts by pulling from Lane 0 (low) or Lane 2 (high) of the first argument X, followed immediately by the corresponding lane of the second argument Y.

### `_mm_unpacklo_ps`

```
Inputs:
X = [ X3 , X2 , X1 , X0 ]
Y = [ Y3 , Y2 , Y1 , Y0 ]

Output:
[ Y1 , X1 , Y0 , X0 ]
  ^    ^    ^    ^
Lane3 Lane2 Lane1 Lane0
```

### `_mm_unpackhi_ps`

```
Inputs:
X = [ X3 , X2 , X1 , X0 ]
Y = [ Y3 , Y2 , Y1 , Y0 ]

Output:
[ Y3 , X3 , Y2 , X2 ]
  ^    ^    ^    ^
Lane3 Lane2 Lane1 Lane0
```

> The X-Y Alternation: No matter what, the resulting register will always alternate elements: `[ Y_lane, X_lane, Y_lane, X_lane ]`. The lo or hi variant simply determines whether you are sampling from the lower lanes (0, 1) or upper lanes (2, 3) of the source registers.

---

## Shuffle (`_mm_shuffle_ps`)

- It allows to route any lane from two source registers into any lane of the destination register.
- For this, the instruction requires a third argument: an 8-bit immediate mask (usually constructed using the _MM_SHUFFLE macro).

> The shuffle instruction splits the destination register right down the middle:
> - Lanes 0 and 1 of the output must come from Register X.
> - Lanes 2 and 3 of the output must come from Register Y.

- `_MM_SHUFFLE(fp3, fp2, fp1, fp0)`
  - It is a literal map. Each argument is a 2-bit integer (0, 1, 2, or 3) indicating exactly which source lane to copy:

```
_mm_shuffle_ps(X, Y, _MM_SHUFFLE(fp3, fp2, fp1, fp0))
                                  |    |    |    |
    Selects lane from Y for Lane 3 --+    |    |    +-- Selects lane from X for Lane 0
         Selects lane from Y for Lane 2 --+    +-- Selects lane from X for Lane 1
```

### Example

```cpp
_mm_shuffle_ps(tmp0, tmp2, _MM_SHUFFLE(1, 0, 1, 0))
```

Inputs are :
- `tmp0 (X) = [ b1, a1, b0, a0 ]`
- `tmp2 (Y) = [ d1, c1, d0, c0 ]`

1. `Output Lane 0` : The macro specifies fp0 = 0. 
  - This looks at Lane 0 of the first register (tmp0), which is a0.

2. `Output Lane 1` : The macro specifies fp1 = 1. 
  - This looks at Lane 1 of the first register (tmp0), which is b0.

3. `Output Lane 2` : The macro specifies fp2 = 0. 
  - This looks at Lane 0 of the second register (tmp2), which is c0.

4. `Output Lane 3` : The macro specifies fp3 = 1. 
  - This looks at Lane 1 of the second register (tmp2), which is d0.


> Resulting Register: `[ d0, c0, b0, a0 ]` (which read left-to-right in standard notation is `[ a0, b0, c0, d0 ]`).

### Little-Endian memory format.

- In a little-endian system, the least significant byte (the lowest value part of a number) is stored at the lowest memory address (the starting point in RAM).

- When loading data from memory into a 128-bit SIMD register, the data fills the register from the lowest address to the highest address.

```
Array Index:    [   0   |   1   |   2   |   3   ]
Register Lane:  [ Lane 3| Lane 2| Lane 1| Lane 0]
Value:          [  a3   |  a2   |  a1   |  a0   ]
```

> **Index 0 = Lane 0 = Far Right**