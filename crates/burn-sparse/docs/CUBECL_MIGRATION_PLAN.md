# CubeCL Migration Plan for burn-sparse

**Decision Date:** TBD
**Current Status:** Pre-decision analysis
**Recommendation:** Migrate AFTER Phase 2 (CUDA working)

---

## 🎯 **What is CubeCL?**

CubeCL is Burn's unified GPU compute abstraction that compiles a single Rust kernel definition to:
- CUDA (NVIDIA)
- HIP (AMD ROCm)
- Metal (Apple)
- SPIR-V / WGSL (WebGPU)
- CPU SIMD (fallback)

**Benefits:**
- Write once, run everywhere
- Automatic optimization via autotuning
- Comptime specialization (e.g., tensor core detection)
- Maintained by Burn team (consistent with ecosystem)

**Drawbacks:**
- Learning curve
- Some CUDA-specific features harder to express
- Potential performance overhead (vs handwritten CUDA)

---

## 📊 **Migration Options Comparison**

### Option 1: Migrate NOW (before CUDA)
**Timeline:** Start immediately, 6-8 weeks

**Pros:**
- No duplicate work (skip CUDA entirely)
- Future-proof from start
- AMD/Metal support from day 1
- Consistent with Burn ecosystem

**Cons:**
- Delays GPU support by 4-6 weeks
- CubeCL learning curve blocks progress
- No CUDA reference implementation
- Higher risk (CubeCL unfamiliar)

**Best for:**
- Team with CubeCL experience
- AMD ROCm is critical requirement
- Long-term maintenance priority
- Can wait for GPU support

---

### Option 2: Migrate AFTER Phase 2 (CUDA first) ⭐ **RECOMMENDED**
**Timeline:** CUDA in 4 weeks, then CubeCL migration in 4 weeks

**Pros:**
- GPU support delivered quickly (CUDA working)
- CUDA as reference for CubeCL translation
- Lower risk (CUDA proven)
- Users get value sooner
- Can compare CubeCL vs CUDA performance

**Cons:**
- Duplicate work (write CUDA, then rewrite in CubeCL)
- Temporary CUDA maintenance
- AMD/Metal delayed by 8 weeks

**Best for:**
- Need working GPU path ASAP
- Want to validate performance before migrating
- Okay with 4 weeks "throwaway" CUDA work
- Prefer incremental risk

**Why this is recommended:**
1. **Momentum:** Working GPU path motivates users/contributors
2. **Validation:** CUDA proves sparse kernels work before CubeCL complexity
3. **Reference:** CUDA code serves as spec for CubeCL translation
4. **Fallback:** If CubeCL migration fails, still have CUDA

---

### Option 3: Never migrate (CUDA + WGPU separate)
**Timeline:** N/A - maintain separate kernels indefinitely

**Pros:**
- Maximum control per backend
- Handwritten CUDA = peak performance
- No abstraction overhead
- Simpler mental model (one kernel = one file)

**Cons:**
- 3-4x code duplication (CUDA, HIP, WGPU, CPU)
- Divergence risk (CUDA gets optimized, WGPU lags)
- Maintenance burden scales with backends
- Out of sync with Burn's direction

**Best for:**
- CUDA-only deployment (unlikely)
- Team with strong CUDA expertise, no CubeCL experience
- Performance is absolute priority (squeeze every cycle)

**Why NOT recommended:**
- Burn ecosystem moving to CubeCL
- AMD/Metal support increasingly important
- Maintenance burden unsustainable long-term

---

## 🗺️ **Recommended Migration Path (Option 2)**

### Phase 1: Core + CPU (Weeks 1-8)
**Focus:** CPU reference implementation

```
burn-sparse/
  kernel/
    api.rs          # SparseKernel trait
    dispatch.rs     # Capability routing
    cpu.rs          # CPU reference (pure Rust)
    mod.rs
```

**No GPU code yet.** CPU is Rust-native, no CubeCL needed.

**Deliverable:** Working sparse training on CPU

---

### Phase 2: CUDA (Weeks 9-12) ← **Value delivery**
**Focus:** Production GPU path, NVIDIA only

```
burn-sparse/
  kernel/
    cuda/
      mod.rs
      spmm_csr.rs       # Handwritten CUDA or cuSPARSE wrapper
      spmm_blockcsr.rs  # Custom kernel
      sddmm.rs          # Custom kernel
      nm_2_4.rs         # Tensor core kernel
```

**Implementation approach:**
- **Easy path:** Wrap cuSPARSE for CSR SpMM (1 week)
- **Medium path:** Custom BlockCSR kernel (2 weeks)
- **Hard path:** N:M 2:4 tensor core kernel (4 weeks, optional)

**Key insight:** cuSPARSE handles most of the work. We're wrapping, not rewriting.

**Deliverable:** GPU-accelerated sparse training, NVIDIA GPUs

**User value:** Can train large sparse models on GPUs

---

### Phase 2.5: CubeCL Migration (Weeks 13-16) ← **Modernization**
**Focus:** Replace CUDA with CubeCL, expand to AMD/Metal

```
burn-sparse/
  kernel/
    cubecl/
      mod.rs
      spmm_csr.rs       # Translates CUDA spmm_csr
      spmm_blockcsr.rs  # Translates CUDA blockcsr
      sddmm.rs          # Translates CUDA sddmm
      layout/
        csr_device.rs   # GPU memory layout helpers
        blockcsr_device.rs
      autotune.rs       # CubeCL autotuning config
      dispatch.rs       # Runtime specialization
```

**Migration steps:**
1. **Week 13:** Learn CubeCL by translating CSR SpMM
   - Take working CUDA kernel as spec
   - Translate to CubeCL compute language
   - Validate against CUDA reference (should match)

2. **Week 14:** Translate BlockCSR + SddMM
   - More complex kernels
   - Shared memory, tiling strategies

3. **Week 15:** Autotuning + specialization
   - Use CubeCL's comptime features
   - Detect tensor cores, optimize tile sizes
   - Benchmark vs handwritten CUDA

4. **Week 16:** Multi-backend testing
   - Test on AMD ROCm (if available)
   - Test on Metal (if available)
   - WebGPU fallback (WGSL)

**Deliverable:** CubeCL kernels match or beat CUDA performance

**Deprecation:** Mark `kernel/cuda/` as deprecated, point to `kernel/cubecl/`

---

### Phase 3: CUDA Removal (Week 17+)
**After validation, remove old CUDA code**

```diff
  kernel/
-   cuda/  # DELETE
    cubecl/
    cpu.rs
    api.rs
    dispatch.rs
```

**Keep CUDA around until:**
- CubeCL performance validated (within 5% of handwritten CUDA)
- Multi-backend tested (AMD, Metal, WGPU)
- No regressions in benchmarks

**Then:** Delete CUDA directory, update docs, celebrate 🎉

---

## 🔧 **CubeCL Implementation Strategy**

### How CubeCL works for sparse kernels

**1. Define compute kernel in Rust:**
```rust
#[cube]
fn spmm_csr_kernel<F: Float>(
    values: &Tensor<F>,
    col_indices: &Tensor<u32>,
    row_pointers: &Tensor<u32>,
    dense_b: &Tensor<F>,
    output: &mut Tensor<F>,
) {
    // CubeCL compute language (subset of Rust)
    let row = ABSOLUTE_POS;
    let row_start = row_pointers[row];
    let row_end = row_pointers[row + 1];

    for i in row_start..row_end {
        let col = col_indices[i];
        let val = values[i];

        for k in 0..dense_b.shape(1) {
            output[row][k] += val * dense_b[col][k];
        }
    }
}
```

**2. Runtime dispatch with backend selection:**
```rust
match B::name() {
    "cuda" => CubeCLKernel::<CudaRuntime>::spmm(a, b),
    "rocm" => CubeCLKernel::<HipRuntime>::spmm(a, b),
    "metal" => CubeCLKernel::<MetalRuntime>::spmm(a, b),
    "wgpu" => CubeCLKernel::<WgpuRuntime>::spmm(a, b),
    _ => CpuKernel::spmm(a, b), // Fallback
}
```

**3. Comptime specialization for hardware features:**
```rust
#[cube]
fn spmm_blockcsr_kernel<F: Float>(...) {
    if comptime!(cuda_sm >= 80) {
        // Use tensor cores (mma.sp for N:M)
        #[cfg(feature = "tensor_cores")]
        tensor_core_multiply(...);
    } else {
        // Fallback to regular SIMD
        regular_block_multiply(...);
    }
}
```

**4. Autotuning for tile sizes:**
```rust
let autotune_config = AutotuneConfig::new()
    .with_params(&[
        ("TILE_SIZE", &[16, 32, 64, 128]),
        ("BLOCK_SIZE", &[8, 16, 32]),
    ])
    .build();

let best_kernel = autotune_config.run(spmm_kernel, &inputs);
```

---

### CubeCL advantages for sparse kernels

**1. N:M format becomes truly portable:**
- NVIDIA: mma.sp (tensor cores)
- AMD: software decode (future RDNA3+ may have HW)
- Metal: software decode
- WebGPU: software decode

One kernel, comptime switches per backend.

**2. BlockCSR optimizations auto-adapt:**
- Detects shared memory size
- Adjusts tile size
- Uses SIMD width (warp size)

**3. SddMM gets backend-specific optimizations:**
- CUDA: coalesced memory access
- AMD: cache hierarchy tuning
- Metal: threadgroup memory

**4. Testing becomes easier:**
- Run same kernel on all backends
- Compare outputs for correctness
- Benchmark across hardware

---

## 📈 **Performance Expectations**

### CubeCL vs Handwritten CUDA

**Typical overhead:**
- Simple kernels (SpMM CSR): 0-5% slower
- Complex kernels (BlockCSR): 5-10% slower
- Autotuned kernels: Can be **faster** than handwritten (if you didn't hand-tune)

**For burn-sparse:**
- **CSR SpMM:** Wrapping cuSPARSE anyway, CubeCL overhead negligible
- **BlockCSR:** Custom kernel, expect 5-10% overhead initially
- **SddMM:** No cuSPARSE equivalent, CubeCL may be *faster* (autotuning finds best tile size)

**Why we're okay with 5-10% overhead:**
- Portability >> 5% perf
- Maintenance burden >> 10% perf
- Users on AMD get 10x speedup (vs no GPU support)

---

## 🎓 **CubeCL Learning Path**

**Week 1: Read CubeCL docs**
- Understand compute language (subset of Rust)
- Learn memory model (global, shared, local)
- Study tensor abstractions

**Week 2: Port simple kernel (CSR SpMM)**
- Start with naive CPU-like implementation
- Translate to CubeCL compute language
- Compare output vs CUDA reference

**Week 3: Optimize (shared memory, tiling)**
- Add shared memory for B matrix
- Tile accumulation
- Benchmark vs CUDA

**Week 4: Multi-backend testing**
- Test on AMD (if available)
- Test on WGPU
- Fix backend-specific issues

---

## 🚀 **Decision Checkpoint**

**Before starting Phase 2.5 (CubeCL migration), validate:**

1. ✅ **CUDA working?**
   - CSR SpMM benchmarked vs cuSPARSE
   - SddMM correctness validated
   - End-to-end training works

2. ✅ **CubeCL maturity?**
   - Check Burn's CubeCL usage (is it production?)
   - Any known sparse kernel limitations?
   - Community support available?

3. ✅ **Team readiness?**
   - Someone familiar with CubeCL?
   - Time budget for 4 weeks?
   - Okay with potential performance tuning?

If **all yes** → proceed with migration

If **any no** → stay on CUDA, revisit in 6 months

---

## 📝 **Migration Checklist**

### Pre-migration (Phase 2 complete)
- [ ] CUDA CSR SpMM working
- [ ] CUDA SddMM working
- [ ] BlockCSR kernel working (optional)
- [ ] Benchmarks vs cuSPARSE collected
- [ ] End-to-end training validated
- [ ] Performance baseline documented

### Migration (Phase 2.5)
- [ ] CubeCL CSR SpMM translated
- [ ] CubeCL SddMM translated
- [ ] BlockCSR translated (if exists)
- [ ] Autotuning configured
- [ ] Multi-backend tested (CUDA, ROCm, WGPU)
- [ ] Performance parity validated (within 10%)

### Post-migration cleanup
- [ ] Deprecate `kernel/cuda/` directory
- [ ] Update documentation (point to CubeCL)
- [ ] Add migration guide for users
- [ ] Delete CUDA code (after 1 release grace period)

---

## 💡 **Alternative: Hybrid Approach**

**Could we keep both?**

Yes, but not recommended. Here's how:

```
kernel/
  api.rs              # SparseKernel trait
  dispatch.rs         # Runtime routing
  cubecl/             # Multi-backend (default)
  cuda/               # CUDA-only fast path (optional)
  cpu.rs              # Reference
```

**Dispatch logic:**
```rust
if cfg!(feature = "cuda-native") && B::name() == "cuda" {
    // Use handwritten CUDA (max performance)
    CudaKernel::spmm(a, b)
} else {
    // Use CubeCL (portable)
    CubeCLKernel::spmm(a, b)
}
```

**When this makes sense:**
- Performance-critical production (CUDA-native)
- Development/portability (CubeCL)

**Downsides:**
- Double maintenance
- Divergence risk
- Confusing for users

**Verdict:** Only if 5% performance difference is critical. Usually not.

---

## 🎯 **Final Recommendation**

**For burn-sparse v2.0:**

1. **Phase 1 (8 weeks):** CPU reference
2. **Phase 2 (4 weeks):** CUDA production ← **Deliver here first**
3. **Phase 2.5 (4 weeks):** CubeCL migration ← **Then modernize**
4. **Phase 3+:** Delete CUDA, fully CubeCL

**Rationale:**
- Users get GPU support fast (week 12)
- Lower risk (CUDA proven)
- CUDA serves as reference for CubeCL
- Still modernize to CubeCL (week 16)
- Total delay vs "CubeCL only": ~4 weeks, but less risky

**When to reconsider:**
- If CubeCL already has sparse kernel examples (check Burn repo)
- If team has CubeCL expert available
- If AMD support is critical from day 1

**Decision date:** After Phase 1 complete (week 8)
**Revisit if:** CubeCL ecosystem evolves significantly

---

## 📞 **Questions for Burn Team**

Before final decision, ask:

1. **Does CubeCL have sparse kernel examples?**
   - If yes: migration easier
   - If no: we'd be pioneering

2. **What's CubeCL's maturity for custom kernels?**
   - Production-ready?
   - Any known sharp edges?

3. **Performance expectations?**
   - Is 5-10% overhead acceptable?
   - Or is matching CUDA critical?

4. **AMD/Metal priority?**
   - Are users asking for it?
   - Or is CUDA 90% of use cases?

5. **Maintenance philosophy?**
   - Burn wants all backends in CubeCL?
   - Or okay with CUDA-specific code?

**Get answers → Make decision → Execute migration**
