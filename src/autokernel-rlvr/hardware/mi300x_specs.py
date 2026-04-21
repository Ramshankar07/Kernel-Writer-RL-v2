"""
AMD Instinct MI300X hardware specifications.

Source: AMD MI300X product brief (GD-176).
All performance figures are peak theoretical unless noted.

Structured as plain constants so other modules (dataset builder,
reward function, system prompts) can import the numbers directly
rather than embedding magic values.
"""

# ---------------------------------------------------------------------------
# AI Peak Theoretical Performance
# ---------------------------------------------------------------------------
# Each entry: (without_sparsity_tflops, with_sparsity_tflops)

AI_PEAK = {
    "TF32":  (653.7,  1307.4),   # TFLOPs
    "FP16":  (1307.4, 2614.9),   # TFLOPs
    "BF16":  (1307.4, 2614.9),   # TFLOPs
    "INT8":  (2614.9, 5229.8),   # TOPS
    "FP8":   (2614.9, 5229.8),   # TFLOPs
}

# Convenience aliases (dense = no sparsity, sparse = with sparsity)
FP8_TFLOPS_DENSE  = AI_PEAK["FP8"][0]   # 2614.9
FP8_TFLOPS_SPARSE = AI_PEAK["FP8"][1]   # 5229.8
FP16_TFLOPS_DENSE = AI_PEAK["FP16"][0]  # 1307.4
BF16_TFLOPS_DENSE = AI_PEAK["BF16"][0]  # 1307.4

# ---------------------------------------------------------------------------
# HPC Peak Theoretical Performance (TFLOPs, dense)
# ---------------------------------------------------------------------------

HPC_PEAK = {
    "FP64_vector": 81.7,
    "FP32_vector": 163.4,
    "FP64_matrix": 163.4,
    "FP32_matrix": 163.4,
}

# ---------------------------------------------------------------------------
# Physical / Electrical Specifications
# ---------------------------------------------------------------------------

SPECS = {
    "form_factor":               "OAM module",
    "lithography_compute_dies":  "5nm FinFET",
    "lithography_io_dies":       "6nm FinFET",
    "compute_units":             304,
    "matrix_cores":              1216,
    "stream_processors":         19_456,
    "peak_engine_clock_mhz":     2100,
    "memory_capacity_gb":        192,           # HBM3, up to
    "memory_bandwidth_tbs":      5.3,           # TB/s, max peak theoretical
    "memory_interface_bits":     8192,
    "infinity_cache_mb":         256,           # last level
    "memory_clock_gts":          5.2,           # GT/s, up to
    "scaleup_links_count":       7,
    "scaleup_link_bandwidth_gbs": 128,          # GB/s per Infinity Fabric link
    "host_io_bandwidth_gbs":     128,           # PCIe Gen5 x16
    "scaleout_bandwidth_gbs":    128,           # PCIe Gen5 x16
    "max_tbp_watts":             750,
    "ras": (
        "Full-chip ECC memory, page retirement, page avoidance"
    ),
}

# ---------------------------------------------------------------------------
# Decoders / Virtualization
# ---------------------------------------------------------------------------

MULTIMEDIA = {
    "decoder_groups":       4,
    "decoder_codec_groups": "HEVC/H.265, AVC/H.264, VP9, AV1",
    "jpeg_cores":           32,
    "jpeg_cores_per_group": 8,
    "virtualization":       "SR-IOV, up to 8 partitions",
}

# ---------------------------------------------------------------------------
# AMD Programming Model Constants (useful for kernel authors)
# ---------------------------------------------------------------------------

WAVEFRONT_SIZE        = 64     # threads per wavefront (vs NVIDIA warp = 32)
MAX_LDS_PER_CU_KB     = 64     # Local Data Share per CU (KB)
VECTOR_REGISTERS_PER_LANE = 512  # VGPRs per lane (typical upper bound)
MAX_OCCUPANCY_WAVES_PER_CU = 40  # waves in-flight per CU (architecture max)

# Practical bandwidth utilization target for memory-bound kernels.
# Sustained real-world peak on MI300X is ~4.7 TB/s (~89% of theoretical).
SUSTAINED_BW_TBS = 4.7

# ---------------------------------------------------------------------------
# Human-readable summary (used in system prompts)
# ---------------------------------------------------------------------------

SUMMARY = f"""AMD Instinct MI300X — Key Numbers for Kernel Writers
======================================================
Compute (dense / with sparsity):
  FP8    : {FP8_TFLOPS_DENSE:>8.1f} / {FP8_TFLOPS_SPARSE:>8.1f} TFLOPs
  BF16   : {BF16_TFLOPS_DENSE:>8.1f} / {AI_PEAK["BF16"][1]:>8.1f} TFLOPs
  FP16   : {FP16_TFLOPS_DENSE:>8.1f} / {AI_PEAK["FP16"][1]:>8.1f} TFLOPs
  TF32   : {AI_PEAK["TF32"][0]:>8.1f} / {AI_PEAK["TF32"][1]:>8.1f} TFLOPs
  INT8   : {AI_PEAK["INT8"][0]:>8.1f} / {AI_PEAK["INT8"][1]:>8.1f} TOPS

Memory:
  Capacity  : {SPECS["memory_capacity_gb"]} GB HBM3
  Bandwidth : {SPECS["memory_bandwidth_tbs"]} TB/s (theoretical), ~{SUSTAINED_BW_TBS} TB/s sustained
  Interface : {SPECS["memory_interface_bits"]}-bit, up to {SPECS["memory_clock_gts"]} GT/s

Topology:
  Compute units : {SPECS["compute_units"]} CUs  ({SPECS["stream_processors"]:,} stream processors)
  Matrix cores  : {SPECS["matrix_cores"]}
  Infinity Cache: {SPECS["infinity_cache_mb"]} MB (last-level)
  Scale-up links: {SPECS["scaleup_links_count"]}× {SPECS["scaleup_link_bandwidth_gbs"]} GB/s Infinity Fabric
  Host I/O      : PCIe Gen5 ×16 ({SPECS["host_io_bandwidth_gbs"]} GB/s)
  TBP           : {SPECS["max_tbp_watts"]} W

Programming model:
  Wavefront size : {WAVEFRONT_SIZE} threads  (≠ NVIDIA warp of 32)
  Max LDS per CU : {MAX_LDS_PER_CU_KB} KB
"""
