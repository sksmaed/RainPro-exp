"""Why is RainPro-8-TW ~8-12x slower per optimizer step than the paper?

THE NUMBER TO BEAT (docs/rainpro_paper.md:267 and Table 9 at :626):
    100k steps @ batch 16 in 13:36 on 1x H100 SXM5 80GB + 26 vCPUs
    == 0.49 s / optimizer step == 30.6 ms / sample

MEASURED on 1x H200 (2026-09-21), batch 4 x accum 4, obs-only:
    fp32 / matmul=highest   3.618 s/step   7.4x the paper   66.1 GiB peak
    fp32 / matmul=high      3.217 s/step   6.6x            (TF32: only 1.12x)
    bf16-mixed              3.245 s/step   6.6x            58.0 GiB peak
    micro-batch 16          OOM on 140 GiB
    parameters              82.62M vs the paper's 36.7M = 2.25x

Read that together: precision buys ~12% and bf16 ties TF32 exactly, so this is
NOT matmul-bound -- it is bandwidth and activation-size bound. 66 GiB for 4
samples is 16.5 GiB/sample, while the paper fit 16 samples in 80 GiB (<5
GiB/sample). Both point at the same place: `dims=(128, 256, 512, 128)` runs the
MaxViT centre at 512 channels and it holds 78% of the parameters, whereas the
paper (docs/rainpro_paper.md:153) describes "halving internal channels" from
MetNet-3 and states "256 channels throughout". `--dims 128 256 256 128` lands
the parameter count near 36.7M; use `--sections model` to check before retiming.

Cross-check on the wall clock: STA_H8 covers ~10.5 months of 2021, so the train
split is ~37k samples == ~2300 optimizer steps per epoch; at 3.6 s that is 2.3 h
of pure compute per epoch against the observed "4+ hours", leaving the rest to
the data pipeline.

So the gap decomposes as:
    ~7.4x in GPU compute      <- dominant; mostly channel width, not precision
    ~2.7x in per-sample CPU   <- real, but overlapped with compute
An infinitely fast DataLoader still leaves you ~7x short. This profiler is
therefore weighted toward the compute side, in this order:

    model    parameter count vs the paper's 36.7M  (CPU-only, seconds)
    gpu      precision x batch-shape matrix        <- the headline experiment
    kernels  torch.profiler top CUDA kernels       <- says *which* op, not just "slow"
    stages   per-sample data-pipeline breakdown    <- the 2.7x
    workers  DataLoader scaling, 2 points          <- confirm the sweet spot only
    fit      real Lightning fit -> T_real          <- ties it all back to wall clock

RUN `--sections model,gpu` FIRST. It is a fork in the road: if GPU-only comes
back near 4 s/step the bottleneck is compute and the data sections can wait;
if it comes back near 0.5 s/step then compute is fine and the whole gap is in
the data pipeline, so run everything else.

Why the precision matrix exists: nothing in this repo calls
`torch.set_float32_matmul_precision`, and PyTorch's default is "highest", so
every matmul runs in true FP32 with no TensorCore path -- on Hopper the
FP32-vs-TF32 matmul peak differs by ~7-8x, which had the right shape for this
gap. Measuring it is what ruled it out (1.12x, see above). Keep the matrix: it
is the cheapest way to re-check the same question after any architecture change,
since a narrower network can shift back toward matmul-bound.

Why the batch-shape comparison exists: the paper fit batch 16 on an 80GB H100.
An H200 has 140GiB. Splitting into 4 micro-batches costs 4x the kernel launches
and lower occupancy for nothing, if 16 fits in one go. It currently OOMs; if
`--dims 128 256 256 128` roughly halves activation memory, retry this -- it is
the next-largest lever after channel width.

Usage: see the `--sections` notes above; every section shares the same
--qpesums / --sta-h8 paths you pass to training.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import os
import random
import statistics
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader  # noqa: F401  (kept for type context)

# `python scripts/profile_training_pipeline.py` puts `scripts/` on sys.path,
# not the repo root, so `import rainpro...` only resolves if the package
# happens to be installed editable. Make it work either way.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import lightning as L  # noqa: E402

import rainpro.data.rainpro8_dataset as dsmod  # noqa: E402
from rainpro.data.rainpro8_datamodule import RainPro8DataModule  # noqa: E402
from rainpro.data.regrid import NearestNeighborRegridder  # noqa: E402
from rainpro.modules.rainpro8 import RainPro8Module  # noqa: E402
from rainpro.modules.utils import EvalRequest  # noqa: E402
from rainpro.network.clt import GroupNorm1VarMean, set_norm_impl  # noqa: E402

# docs/rainpro_paper.md:626 (Table 9) and :267.
PAPER_SECONDS_PER_STEP = 13.6 * 3600 / 100_000  # 0.4896 s @ batch 16
PAPER_BATCH = 16
PAPER_PARAMS = 36.7e6
PAPER_VCPUS = 26


def pct(xs, q):
    return float(np.percentile(xs, q)) if xs else float("nan")


def report_times(name, xs, unit="s"):
    if not xs:
        return
    print(
        f"{name}: n={len(xs)} mean={statistics.mean(xs):.4f}{unit} "
        f"median={statistics.median(xs):.4f}{unit} "
        f"p95={pct(xs,95):.4f}{unit} p99={pct(xs,99):.4f}{unit} max={max(xs):.4f}{unit}"
    )


def make_dm(args, num_workers=None, batch_size=None):
    return RainPro8DataModule(
        data_root={"qpesums": args.qpesums, "sta_h8": args.sta_h8},
        start_date=args.start_date,
        end_date=args.end_date,
        include_satellite=True,
        include_gfs=False,
        cycle_train_days=args.cycle_train_days,
        cycle_val_days=args.cycle_val_days,
        cycle_test_days=args.cycle_test_days,
        cycle_blackout_hours=args.cycle_blackout_hours,
        center_lat=args.center_lat,
        center_lon=args.center_lon,
        train_jitter_km=args.train_jitter_km,
        batch_size=args.batch_size if batch_size is None else batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers if num_workers is None else num_workers,
        persistent_workers=True,
        prefetch_factor=args.prefetch_factor,
        frame_cache_size=args.frame_cache_size,
        variable_aliases={"max_dbz": args.max_dbz_name},
    )


def setup_dm(args, num_workers=None, batch_size=None):
    dm = make_dm(args, num_workers=num_workers, batch_size=batch_size)
    dm.setup("fit")
    return dm


def build_module(dm, args):
    """`--dims` exists so the paper's 36.7M can be *tested*, not argued about.
    `(256, 256, 256, 256)` is the measured match and the paper's stated "256
    channels throughout"; the repo default `(128, 256, 512, 128)` measures
    82.6M.

    `set_norm_impl` must run before the module is built -- `LayerNorm` picks its
    implementation at construction, so flipping it afterwards does nothing."""
    set_norm_impl(args.norm)
    kwargs = {}
    if args.dims is not None:
        kwargs["dims"] = tuple(args.dims)
    if args.center_depth is not None:
        kwargs["center_depth"] = args.center_depth
    module = RainPro8Module(data=dm, max_epochs=1, **kwargs)

    if args.compile:
        # Compile the *network* forward only, and by rebinding the bound method
        # rather than wrapping the module: `RainPro.predict` calls
        # `self.forward(...)`, and `torch.compile(module)` returns an
        # OptimizedModule whose `.predict` would still run eager. This also
        # keeps `self.criterion` (Bucketize/Threshold) outside the graph, where
        # it would most likely break it anyway.
        module.model.forward = torch.compile(
            module.model.forward, mode=args.compile_mode
        )
    return module


# --------------------------------------------------------------------------
# model: is the architecture even the same size as the paper's?
# --------------------------------------------------------------------------

def benchmark_model(args):
    """One cheap check that can invalidate every timing below it.

    The paper is explicit: "We use 256 channels throughout the entire network,
    totaling 36.7 million parameters." If ours is materially larger, an ~8x
    compute gap needs no further explanation and the fix is architectural, not
    a precision flag. Runs on CPU, so it is safe on a login node."""
    print("\n=== Model size vs paper ===")
    dm = setup_dm(args, num_workers=0)
    model = build_module(dm, args)

    total = sum(p.numel() for p in model.model.parameters())
    trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"parameters: total={total/1e6:.2f}M trainable={trainable/1e6:.2f}M")
    print(f"paper:      36.70M  -> ratio {total/PAPER_PARAMS:.2f}x")
    print(f"dims={model.hparams.get('dims')} cond_dim={model.hparams.get('cond_dim')} "
          f"center_depth={model.hparams.get('center_depth')} "
          f"resnet_depth={model.hparams.get('resnet_depth')}")
    print("NOTE: paper says 256 channels throughout; a `dims` tuple containing 512 "
          "means wider stages than the paper, which scales compute ~quadratically "
          "in that stage's channel count.")

    print("\nper-submodule parameter counts:")
    rows = sorted(
        ((n, sum(p.numel() for p in m.parameters(recurse=True)))
         for n, m in model.model.named_children()),
        key=lambda kv: -kv[1],
    )
    for name, count in rows:
        print(f"  {name:30s} {count/1e6:8.3f}M  {100*count/max(total,1):5.1f}%")
    del model, dm


# --------------------------------------------------------------------------
# checknorm: prove the replacement is the same function before timing it
# --------------------------------------------------------------------------

def _norm_pair(num_channels, device, dtype):
    import torch.nn as nn

    native = nn.GroupNorm(1, num_channels, affine=True).to(device)
    with torch.no_grad():
        native.weight.normal_(1.0, 0.1)
        native.bias.normal_(0.0, 0.1)
    custom = GroupNorm1VarMean(num_channels, affine=True).to(device)
    # Same parameter names and shapes, so this is also the check that a
    # checkpoint moves between the two implementations unchanged.
    custom.load_state_dict(native.state_dict())
    return native, custom


def benchmark_check_norm(args):
    """`var_mean` is only worth timing if it is the *same* normalization.

    Two things could silently differ: `torch.var_mean` defaults to the unbiased
    `correction=1` while GroupNorm uses the population variance, and the affine
    has to land per channel rather than per element. Both would still train --
    just not the paper's model. So compare forward and every gradient path
    before trusting any speed number."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Norm equivalence check (device={device.type}) ===")
    C = 256
    ok = True
    for dtype, label in ((torch.float32, "fp32"), (torch.bfloat16, "bf16-autocast")):
        if dtype is torch.bfloat16 and device.type != "cuda":
            continue
        torch.manual_seed(args.seed)
        native, custom = _norm_pair(C, device, dtype)
        x = torch.randn(2, C, 64, 64, device=device)

        outs, grads, out_dtypes = [], [], []
        for mod in (native, custom):
            xi = x.clone().requires_grad_(True)
            mod.zero_grad(set_to_none=True)
            ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
                   if dtype is torch.bfloat16 else contextlib.nullcontext())
            with ctx:
                y = mod(xi)
            out_dtypes.append(y.dtype)
            # A non-uniform scalar, so every element contributes distinctly --
            # `y.sum()` would hide errors that cancel across the tensor.
            loss = (y.float() * torch.linspace(0.5, 1.5, C, device=device)
                    .view(1, -1, 1, 1)).square().mean()
            loss.backward()
            outs.append((y.float().detach(), loss.detach()))
            grads.append((xi.grad.float(), mod.weight.grad.float(), mod.bias.grad.float()))

        # Autocast keeps some normalizations in fp32 regardless of the ambient
        # dtype. If native comes back fp32 here while the custom one comes back
        # bf16, that is worth knowing twice over: the bf16 comparison below is
        # then across two precisions (so expect it to sit near tolerance), and
        # it would also explain why bf16-mixed bought ~0% speed earlier -- the
        # kernel holding 59% of the time would have been running fp32 anyway.
        if dtype is torch.bfloat16:
            differ = "   <- DIFFER, see note below" if out_dtypes[0] != out_dtypes[1] else ""
            print(f"    autocast output dtype: native={out_dtypes[0]} "
                  f"custom={out_dtypes[1]}{differ}")

        tol = 2e-3 if dtype is torch.bfloat16 else 2e-5
        rows = [
            ("forward", outs[0][0], outs[1][0]),
            ("loss", outs[0][1], outs[1][1]),
            ("grad input", grads[0][0], grads[1][0]),
            ("grad weight", grads[0][1], grads[1][1]),
            ("grad bias", grads[0][2], grads[1][2]),
        ]
        print(f"  {label} (tolerance {tol:g}):")
        for name, a, b in rows:
            diff = (a - b).abs().max().item()
            scale = max(a.abs().max().item(), 1e-12)
            rel = diff / scale
            flag = "ok " if rel <= tol else "FAIL"
            ok &= rel <= tol
            print(f"    {flag} {name:12s} max|diff|={diff:.3e}  rel={rel:.3e}")
    print("  => equivalent" if ok else
          "  => NOT equivalent; do not trust var_mean timings until this passes")
    print("  NOTE: if the bf16 row shows native=float32 and custom=bfloat16, autocast is\n"
          "  keeping nn.GroupNorm in fp32. That is a real finding, not a bug in the check:\n"
          "  it would mean bf16-mixed never touched the kernel holding 59% of CUDA time,\n"
          "  which is exactly why bf16 measured no faster than TF32. The fp32 rows above\n"
          "  are then the authoritative equivalence evidence.")


# --------------------------------------------------------------------------
# gpu: the headline experiment -- precision x batch shape
# --------------------------------------------------------------------------

# (label, float32_matmul_precision, autocast dtype or None, cudnn.benchmark)
GPU_VARIANTS = [
    ("fp32 / matmul=highest (CURRENT)", "highest", None, False),
    ("fp32 / matmul=high (TF32)", "high", None, False),
    ("bf16-mixed", "high", torch.bfloat16, False),
    ("bf16-mixed + cudnn.benchmark", "high", torch.bfloat16, True),
]


def slice_batch(batch, lo, hi):
    return {k: v[lo:hi] for k, v in batch.items()}


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def time_variant(model, optimizer, microbatches, amp_dtype, warmup, steps):
    """GPU-only optimizer-step wall time. Synchronizing around the *whole*
    step is correct here -- unlike the data-overlap question, there is nothing
    to overlap with when the batch already lives on the device."""
    accum = len(microbatches)

    def one_step():
        optimizer.zero_grad(set_to_none=True)
        for mb in microbatches:
            ctx = (torch.autocast("cuda", dtype=amp_dtype)
                   if amp_dtype is not None else contextlib.nullcontext())
            with ctx:
                loss = model(mb, EvalRequest(only_loss=True, need_loss=True))
            # bf16 autocast needs no GradScaler (unlike fp16); the loss comes
            # back fp32 from the criterion either way.
            (loss / accum).backward()
        optimizer.step()

    for _ in range(warmup):
        one_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        one_step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times, torch.cuda.max_memory_allocated()


def benchmark_gpu(args):
    if not torch.cuda.is_available():
        print("\n=== GPU benchmark skipped: CUDA unavailable ===")
        return
    print("\n=== GPU-only: precision x batch shape ===")
    print(f"device: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory/2**30:.0f} GiB)")
    print(f"paper target: {PAPER_SECONDS_PER_STEP:.3f} s / optimizer step "
          f"@ batch {PAPER_BATCH} on H100 80GB\n")

    eff = args.effective_batch
    dm = setup_dm(args, batch_size=eff)
    device = torch.device("cuda:0")
    # One real batch of `effective_batch`, reused for every variant so the
    # comparison is exactly apples-to-apples (and so this section never waits
    # on the DataLoader again).
    cpu_batch = next(iter(dm.train_dataloader()))
    gpu_batch = to_device(cpu_batch, device)
    del cpu_batch

    model = build_module(dm, args).to(device)
    model.train()
    opt_cfg = model.configure_optimizers()
    optimizer = opt_cfg["optimizer"] if isinstance(opt_cfg, dict) else opt_cfg

    # Same effective batch, different shapes: 4x4 (what we run now) vs 16x1
    # (what the paper ran). Identical data, so any delta is pure launch
    # overhead / occupancy.
    shapes = []
    for micro in args.micro_batches:
        if eff % micro:
            print(f"skipping micro-batch {micro}: does not divide effective batch {eff}")
            continue
        shapes.append(micro)

    header = f"{'variant':34s}{'micro':>7s}{'accum':>7s}{'s/step':>10s}{'vs paper':>10s}{'peak GiB':>10s}"
    print(header)
    print("-" * len(header))
    results = {}
    variants = ([GPU_VARIANTS[i] for i in args.variants]
                if args.variants is not None else GPU_VARIANTS)
    for label, matmul, amp_dtype, cudnn_bench in variants:
        torch.set_float32_matmul_precision(matmul)
        torch.backends.cudnn.benchmark = cudnn_bench
        for micro in shapes:
            accum = eff // micro
            mbs = [slice_batch(gpu_batch, i * micro, (i + 1) * micro) for i in range(accum)]
            try:
                times, peak = time_variant(
                    model, optimizer, mbs, amp_dtype, args.gpu_warmup, args.gpu_steps
                )
            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                print(f"{label:34s}{micro:>7d}{accum:>7d}{'OOM':>10s}{'':>10s}{'':>10s}")
                continue
            mean = statistics.mean(times)
            results[(label, micro)] = mean
            print(f"{label:34s}{micro:>7d}{accum:>7d}{mean:>10.3f}"
                  f"{mean/PAPER_SECONDS_PER_STEP:>9.1f}x{peak/2**30:>10.1f}")
            torch.cuda.empty_cache()

    if results:
        base = results.get((GPU_VARIANTS[0][0], args.micro_batches[0]))
        best_key = min(results, key=results.get)
        print(f"\nbest: {best_key[0]} @ micro-batch {best_key[1]} "
              f"-> {results[best_key]:.3f} s/step")
        if base:
            print(f"speedup over current config: {base/results[best_key]:.2f}x "
                  f"(would still be {results[best_key]/PAPER_SECONDS_PER_STEP:.1f}x "
                  f"the paper's 0.490 s)")
    # Leave the process in the default state for any later section.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    del model, optimizer, gpu_batch, dm
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# kernels: which op is actually eating the time
# --------------------------------------------------------------------------

def benchmark_kernels(args):
    """`gpu` says compute costs N seconds; this says what those seconds are.

    If the top `self_cuda_time_total` entries are FP32 GEMM kernels (sgemm /
    *_nn_ / cutlass_simt_*), the TF32 hypothesis is confirmed on the spot --
    `simt` in a kernel name means the CUDA cores, i.e. no TensorCore at all."""
    if not torch.cuda.is_available():
        print("\n=== Kernel profile skipped: CUDA unavailable ===")
        return
    from torch.profiler import ProfilerActivity, profile

    print("\n=== Top CUDA kernels (current fp32/highest config) ===")
    eff = args.effective_batch
    micro = args.micro_batches[0]
    accum = eff // micro
    dm = setup_dm(args, batch_size=eff)
    device = torch.device("cuda:0")
    gpu_batch = to_device(next(iter(dm.train_dataloader())), device)
    model = build_module(dm, args).to(device)
    model.train()
    opt_cfg = model.configure_optimizers()
    optimizer = opt_cfg["optimizer"] if isinstance(opt_cfg, dict) else opt_cfg
    mbs = [slice_batch(gpu_batch, i * micro, (i + 1) * micro) for i in range(accum)]

    torch.set_float32_matmul_precision(args.kernel_matmul)
    print(f"matmul_precision={args.kernel_matmul} micro={micro} accum={accum}")

    def one_step():
        optimizer.zero_grad(set_to_none=True)
        for mb in mbs:
            loss = model(mb, EvalRequest(only_loss=True, need_loss=True))
            (loss / accum).backward()
        optimizer.step()

    for _ in range(args.gpu_warmup):
        one_step()
    torch.cuda.synchronize()

    # No `schedule=` here. With one, the profiler clears its event buffer at the
    # end of each cycle, so `key_averages()` after the `with` block sees only
    # whatever fragment survived the last cycle -- which is how the first run of
    # this section reported a single 18us `cudaDeviceSynchronize` and no CUDA
    # kernels at all. Profiling a fixed number of steps with no cycling keeps
    # every event.
    # Profiling compiled code can fail inside Triton's own teardown; that must
    # not take the whole run's results with it.
    try:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
        ) as prof:
            for _ in range(args.kernel_steps):
                one_step()
            torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by="self_cuda_time_total",
                                        row_limit=args.kernel_rows))
    except Exception as exc:  # noqa: BLE001 -- diagnostic script, never fatal
        print(f"kernel profile failed ({type(exc).__name__}: {exc}); "
              "with --compile, try running `--sections kernels` on its own")
    torch.set_float32_matmul_precision("highest")
    del model, optimizer, gpu_batch, dm
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# stages: the data pipeline (the ~2.7x, secondary to compute)
# --------------------------------------------------------------------------

@contextlib.contextmanager
def stage_instrumentation():
    totals = defaultdict(float)
    counts = Counter()
    originals = []

    def patch(obj, name, key):
        orig = getattr(obj, name)
        originals.append((obj, name, orig))

        def wrapped(*a, **kw):
            t0 = time.perf_counter()
            try:
                return orig(*a, **kw)
            finally:
                totals[key] += time.perf_counter() - t0
                counts[key] += 1

        setattr(obj, name, wrapped)

    # `_load_frames` gets its own wrapper so the cost can be attributed per
    # variable *and* per frame. The 12 calls a sample makes differ by 36x in
    # how many frames they fetch (`target_2km` asks for 36 offsets in one
    # `isel`, `radar_8km` for 1), so whether the time tracks frames or tracks
    # calls is what separates "reading/decompressing bytes" from "per-call
    # xarray/dask overhead" -- and those have completely different fixes.
    load_detail: dict[str, list[float]] = defaultdict(lambda: [0.0, 0, 0])
    _orig_load = dsmod.RainPro8Dataset._load_frames
    originals.append((dsmod.RainPro8Dataset, "_load_frames", _orig_load))

    def _wrapped_load(self, store_key, ds, raw_name, positions, *a, **kw):
        n_frames = 1 if positions is None else len(positions)
        start = time.perf_counter()
        try:
            return _orig_load(self, store_key, ds, raw_name, positions, *a, **kw)
        finally:
            elapsed = time.perf_counter() - start
            totals["load_frames_io_decode_cache"] += elapsed
            counts["load_frames_io_decode_cache"] += 1
            entry = load_detail[raw_name]
            entry[0] += elapsed
            entry[1] += 1
            entry[2] += n_frames

    dsmod.RainPro8Dataset._load_frames = _wrapped_load

    patch(NearestNeighborRegridder, "prepare", "regrid_prepare_kdtree")
    patch(NearestNeighborRegridder, "apply", "regrid_apply_gather")
    patch(dsmod, "_fill_no_echo", "fill_no_echo")
    patch(dsmod, "_mask_missing", "mask_missing")
    patch(dsmod, "minmax_normalize", "normalize")

    try:
        yield totals, counts, load_detail
    finally:
        for obj, name, orig in reversed(originals):
            setattr(obj, name, orig)


def benchmark_stages(args):
    print("\n=== Single-process stage breakdown ===")
    budget = PAPER_SECONDS_PER_STEP * PAPER_VCPUS / PAPER_BATCH
    print(f"paper's implied data budget: {budget:.2f} core-s/sample "
          f"({PAPER_VCPUS} vCPUs x {PAPER_SECONDS_PER_STEP:.3f}s / {PAPER_BATCH} samples)")
    dm = setup_dm(args, num_workers=0)
    dataset = dm.train_dataloader().dataset
    rng = random.Random(args.seed)
    indices = [rng.randrange(len(dataset)) for _ in range(args.stage_samples)]

    # One warm-up sample absorbs the one-off store-open / cKDTree-build cost a
    # persistent worker pays only once; timing it would slander the steady state.
    dataset[indices[0]]

    with stage_instrumentation() as (totals, counts, load_detail):
        per_sample = []
        for idx in indices:
            if args.clear_frame_cache:
                dataset._frame_cache.clear()
            t0 = time.perf_counter()
            dataset[idx]
            per_sample.append(time.perf_counter() - t0)

    total_wall = sum(per_sample)
    n = len(indices)
    report_times("dataset[random_idx]", per_sample)
    print(f"mean={total_wall/n:.3f} s/sample -> {total_wall/n/budget:.1f}x the paper's budget")
    # These six wrappers do not nest in the current code -- `_load_frames` calls
    # none of the others, and the rest are siblings inside `_read_source` -- so
    # the sum really is close to a partition of the per-sample cost.
    for key, total in sorted(totals.items(), key=lambda kv: kv[1], reverse=True):
        count = counts[key]
        print(f"  {key:30s} total={total:8.3f}s per-sample={total/n:7.4f}s "
              f"{100*total/total_wall:5.1f}%  calls={count:5d} per-call={total/count:8.5f}s")
    accounted = sum(totals.values())
    print(f"  {'(unaccounted: alloc/xarray/stack)':30s} total={total_wall-accounted:8.3f}s "
          f"per-sample={(total_wall-accounted)/n:7.4f}s {100*(total_wall-accounted)/total_wall:5.1f}%")

    # The decisive table. Each row is one `_load_frames` call site (one variable
    # of one source). If s/call is roughly flat across rows while frames/call
    # ranges over 36x, the cost is per-call overhead -- xarray indexing and dask
    # graph construction -- and the fix is to read the zarr arrays directly or
    # to batch the calls. If instead s/frame is flat, the cost really is bytes,
    # and the fix is chunking and the compression level.
    if load_detail:
        print("\n  _load_frames by variable (does cost track FRAMES or CALLS?):")
        print(f"    {'variable':<18}{'s total':>9}{'calls':>7}{'frames':>8}"
              f"{'s/call':>10}{'s/frame':>10}")
        for name, (secs, calls, frames) in sorted(
            load_detail.items(), key=lambda kv: -kv[1][0]
        ):
            print(f"    {name:<18}{secs:>9.3f}{calls:>7d}{frames:>8d}"
                  f"{secs/max(calls,1):>10.4f}{secs/max(frames,1):>10.4f}")
        per_call = [s / max(c, 1) for s, c, _ in load_detail.values()]
        per_frame = [s / max(f, 1) for s, _, f in load_detail.values()]

        def spread(xs):
            return max(xs) / max(min(xs), 1e-12)

        print(f"    spread across variables: s/call {spread(per_call):.1f}x, "
              f"s/frame {spread(per_frame):.1f}x  <- the flatter one is the real unit")

    # Repeating one index serves all 62 frames from `_frame_cache`, so it is a
    # de-facto CPU-only measurement: random - same == I/O + decompress. This
    # only holds while frame_cache_size >= 62 (one obs-only sample's frames).
    same_idx = indices[0]
    same_times = []
    for _ in range(args.stage_samples):
        t0 = time.perf_counter()
        dataset[same_idx]
        same_times.append(time.perf_counter() - t0)
    same_mean = statistics.mean(same_times)
    report_times("dataset[same_idx] (all frames cached == CPU-only)", same_times)
    if args.frame_cache_size < 62:
        print(f"  WARNING: --frame-cache-size {args.frame_cache_size} < 62 frames/sample, "
              "so this is NOT a clean CPU-only number")
    print(f"  => CPU-only ~{same_mean:.3f} s/sample, "
          f"I/O+decompress ~{total_wall/n - same_mean:.3f} s/sample")
    del dm


# --------------------------------------------------------------------------
# readpath: is the per-chunk cost the store, or the layer on top of it?
# --------------------------------------------------------------------------

def _first_time_var(ds):
    for name, da in ds.data_vars.items():
        if "time" in da.dims:
            return name
    raise SystemExit(f"no time-dimensioned data_var in {list(ds.data_vars)}")


def benchmark_readpath(args):
    """`stages` showed the cost tracks *chunks*, not bytes and not calls.

    A chunk is one frame here, and the measured 27-40 ms per chunk is far too
    slow for reading and decompressing ~1-6 MiB that the page cache already
    holds (re-running `stages` on identical indices changed nothing). That
    points above the store rather than at it: `_get_store` calls
    `xr.open_zarr(path, consolidated=True)` with no `chunks=`, and xarray
    defaults to `chunks='auto'`, so every `.isel(time=[...]).values` builds and
    runs a dask graph with a task per chunk.

    This compares that path against two that skip dask, on the same stores and
    the same positions, so the fix is measured rather than assumed."""
    import xarray as xr
    import zarr

    print("\n=== Read path comparison ===")
    stores = [("qpesums", args.qpesums, args.readpath_frames)]
    sta = [p.strip() for p in args.sta_h8.split(",") if p.strip()]
    if sta:
        # Satellite asks for 2 offsets x 9 bands per sample, so 2 is the real
        # per-call shape there; QPESUMS' target_2km asks for 36 at once.
        stores.append(("sta_h8[0]", sta[0], 2))

    rng = random.Random(args.seed)
    for label, path, n_frames in stores:
        try:
            probe = xr.open_zarr(path, consolidated=True)
        except Exception:
            probe = xr.open_zarr(path, consolidated=False)
        var = _first_time_var(probe)
        n_times = probe.sizes["time"]
        probe.close()
        print(f"\n  {label}: {path}")
        print(f"  variable={var} times={n_times} frames/read={n_frames}")

        # Drawn fresh per repeat, contiguous like the real offset windows.
        position_sets = [
            sorted(rng.sample(range(n_times), n_frames)) if n_frames > 1
            else [rng.randrange(n_times)]
            for _ in range(args.readpath_repeats)
        ]

        def run(make_reader, name):
            try:
                reader, closer = make_reader()
            except Exception as exc:  # noqa: BLE001
                print(f"    {name:<34} unavailable ({type(exc).__name__}: {exc})")
                return
            try:
                reader(position_sets[0])  # warm up open/metadata, not timed
                times = []
                for pos in position_sets:
                    t = time.perf_counter()
                    out = reader(pos)
                    times.append(time.perf_counter() - t)
                    del out
                mean = statistics.mean(times)
                print(f"    {name:<34}{mean:>9.4f}s{mean/n_frames:>11.4f}s/frame")
                return mean
            finally:
                closer()

        print(f"    {'method':<34}{'s/read':>10}{'per frame':>12}")

        def xr_dask():
            ds = xr.open_zarr(path, consolidated=True)
            return (lambda pos: ds[var].isel(time=pos).values), ds.close

        def xr_nodask():
            ds = xr.open_zarr(path, consolidated=True, chunks=None)
            return (lambda pos: ds[var].isel(time=pos).values), ds.close

        def zarr_oindex():
            arr = zarr.open(path, mode="r")[var]
            return (lambda pos: arr.oindex[pos]), (lambda: None)

        def zarr_loop():
            arr = zarr.open(path, mode="r")[var]
            return (lambda pos: np.stack([arr[p] for p in pos])), (lambda: None)

        base = run(xr_dask, "xarray + dask (current)")
        for maker, name in (
            (xr_nodask, "xarray, chunks=None (no dask)"),
            (zarr_oindex, "zarr direct, oindex"),
            (zarr_loop, "zarr direct, per-frame loop"),
        ):
            got = run(maker, name)
            if base and got:
                print(f"    {'':<34}{base/got:>9.1f}x faster than current")


# --------------------------------------------------------------------------
# workers: confirm the sweet spot; the answer is already roughly known
# --------------------------------------------------------------------------

def benchmark_workers(args):
    """Does loader throughput scale with worker count, or plateau?

    That is the question that separates a latency-bound pipeline (more
    concurrent readers help) from a saturated shared filesystem (they do not,
    and the fix is to move the bytes instead)."""
    print("\n=== DataLoader worker scaling ===", flush=True)
    # One datamodule for the whole sweep. `_dataloader` reads `self.num_workers`
    # at call time, so the worker count can be varied between points -- and the
    # split computation (a scan over QPESUMS' ~610k timestamps plus the STA_H8
    # coverage union across every store) is identical for every point. Paying it
    # once instead of once per point is most of this section's wall time on a
    # busy filesystem, and it was why the section could sit silent for many
    # minutes before the first row appeared.
    t0 = time.perf_counter()
    dm = setup_dm(args)
    print(f"splits ready in {time.perf_counter() - t0:.1f}s; sweeping "
          f"{args.worker_sweep} x ({args.loader_warmup} warmup + "
          f"{args.worker_batches} timed) batches of {args.batch_size}", flush=True)
    for workers in args.worker_sweep:
        dm.num_workers = workers
        dm.persistent_workers = workers > 0
        dm.prefetch_factor = args.prefetch_factor if workers > 0 else None
        loader = dm.train_dataloader()
        it = iter(loader)
        warmup = 0 if workers == 0 else min(args.loader_warmup, 10)
        for _ in range(warmup):
            next(it)
        times = []
        for _ in range(args.worker_batches):
            t = time.perf_counter()
            next(it)
            times.append(time.perf_counter() - t)
        mean = statistics.mean(times)
        print(
            f"workers={workers:2d}: mean={mean:.4f}s/batch "
            f"p95={pct(times,95):.4f}s throughput={args.batch_size/mean:.2f} samples/s",
            flush=True,
        )
        # Drop the iterator explicitly: it owns the worker processes, and
        # relying on rebinding at the top of the next loop would keep the old
        # pool alive while the next one spawns.
        del it, loader
        gc.collect()
    del dm


# --------------------------------------------------------------------------
# fit: the real number, with real overlap
# --------------------------------------------------------------------------

class StepTimer(L.Callback):
    """Times optimizer steps (not micro-batches), skipping the first few."""

    def __init__(self, skip: int):
        self.skip = skip
        self.times: list[float] = []
        self._prev = None
        self._last_gs = -1

    def on_train_batch_end(self, trainer, *_):
        gs = trainer.global_step
        if gs == self._last_gs:
            return
        self._last_gs = gs
        now = time.perf_counter()
        if self._prev is not None and gs > self.skip:
            self.times.append(now - self._prev)
        self._prev = now


def benchmark_fit(args):
    """The only section that measures what you actually pay: a real Lightning
    fit, real DataLoader overlap, no synchronization games. Everything above
    explains this number; this number is the one to compare with 0.490 s."""
    if not torch.cuda.is_available():
        print("\n=== Real-fit benchmark skipped: CUDA unavailable ===")
        return
    print("\n=== Real Lightning fit (T_real) ===")
    micro = args.micro_batches[0]
    accum = args.effective_batch // micro
    torch.set_float32_matmul_precision(args.fit_matmul)
    print(f"precision={args.fit_precision} matmul={args.fit_matmul} "
          f"micro={micro} accum={accum} workers={args.num_workers}")

    dm = setup_dm(args, batch_size=micro)
    model = build_module(dm, args)
    timer = StepTimer(skip=args.fit_warmup_steps)
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        precision=args.fit_precision,
        max_steps=args.fit_steps,
        accumulate_grad_batches=accum,
        gradient_clip_val=1.0,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[timer],
    )
    t0 = time.perf_counter()
    trainer.fit(model, datamodule=dm)
    wall = time.perf_counter() - t0

    report_times("real optimizer step", timer.times)
    if timer.times:
        mean = statistics.mean(timer.times)
        print(f"T_real = {mean:.3f} s/step vs paper {PAPER_SECONDS_PER_STEP:.3f} s "
              f"-> {mean/PAPER_SECONDS_PER_STEP:.1f}x")
        print(f"implied: {args.effective_batch/mean:.2f} samples/s, "
              f"{mean*2300/3600:.2f} h/epoch at ~2300 steps/epoch")
    print(f"(total wall incl. setup/teardown: {wall:.1f}s)")
    torch.set_float32_matmul_precision("highest")
    # `persistent_workers=True` keeps this fit's worker processes alive for as
    # long as the DataLoader is reachable, so without this they would sit on
    # the allocation's CPUs and page cache through every later section.
    del trainer, model, timer, dm
    gc.collect()


# --------------------------------------------------------------------------

SECTIONS = {
    "model": benchmark_model,
    "checknorm": benchmark_check_norm,
    "gpu": benchmark_gpu,
    "kernels": benchmark_kernels,
    "stages": benchmark_stages,
    "readpath": benchmark_readpath,
    "workers": benchmark_workers,
    "fit": benchmark_fit,
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--qpesums", required=True)
    ap.add_argument("--sta-h8", required=True, help="comma-separated for multiple zarr stores")
    ap.add_argument("--max-dbz-name", default="MaxDBZ")
    ap.add_argument("--start-date", default="2021-01-01")
    ap.add_argument("--end-date", default="2022-01-01")
    ap.add_argument("--cycle-train-days", type=int, default=17)
    ap.add_argument("--cycle-val-days", type=int, default=1)
    ap.add_argument("--cycle-test-days", type=int, default=2)
    ap.add_argument("--cycle-blackout-hours", type=float, default=12)
    ap.add_argument("--center-lat", type=float, default=23.7)
    ap.add_argument("--center-lon", type=float, default=121.0)
    ap.add_argument("--train-jitter-km", type=float, default=256)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="batch size for the data-pipeline sections (stages/workers)")
    ap.add_argument("--eval-batch-size", type=int, default=2)
    ap.add_argument("--effective-batch", type=int, default=16,
                    help="samples per optimizer step; 16 == the paper's batch size")
    ap.add_argument("--micro-batches", type=int, nargs="+", default=[4, 16],
                    help="micro-batch sizes to compare at the same effective batch; "
                         "4 == what we run now (accum 4), 16 == what the paper ran (accum 1)")
    ap.add_argument("--num-workers", type=int, default=11,
                    help="11, not 15: --cpus-per-task=12 leaves 11 for workers")
    ap.add_argument("--prefetch-factor", type=int, default=4)
    ap.add_argument("--frame-cache-size", type=int, default=128,
                    help="matches rainpro8.yml; must be >= 62 for the CPU-only split")
    ap.add_argument("--dims", type=int, nargs=4, default=None,
                    metavar=("DIM_4KM", "DIM_8KM", "DIM_16KM", "DIM_2KM"),
                    help="override RainPro's channel widths. The repo default is "
                         "(128, 256, 512, 128), which measures 82.6M parameters "
                         "against the paper's 36.7M; the MaxViT centre runs at "
                         "DIM_16KM and holds 78%% of them. Try `--dims 128 256 256 128`.")
    ap.add_argument("--center-depth", type=int, default=None,
                    help="number of MaxViT blocks (repo and paper both use 12)")
    ap.add_argument("--norm", default="native", choices=["native", "var_mean"],
                    help="NCHW normalization implementation. 'native' is upstream's "
                         "nn.GroupNorm(num_groups=1), whose moments kernel gets a grid "
                         "of only N blocks (4 at micro-batch 4, on 132 SMs) and ate 59%% "
                         "of CUDA time when measured. 'var_mean' is the same function "
                         "through a multi-block reduction; run --sections checknorm first.")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the network forward. Inductor decomposes "
                         "aten.native_group_norm into var_mean + elementwise and can fuse "
                         "the result, so this is the other route past the same kernel. "
                         "Expect minutes of compilation on the first step; --gpu-warmup "
                         "covers it, and set TORCH_LOGS=graph_breaks to see fragmentation.")
    ap.add_argument("--compile-mode", default="default",
                    choices=["default", "reduce-overhead", "max-autotune"])
    ap.add_argument("--variants", type=int, nargs="+", default=None,
                    metavar="I",
                    help="indices into the precision matrix, to skip rows you do not need: "
                         + "; ".join(f"{i}={v[0]}" for i, v in enumerate(GPU_VARIANTS))
                         + ". Worth using with --compile, where each row is a separate "
                           "recompilation.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sections", default="model,gpu",
                    help="comma-separated: " + ",".join(SECTIONS) +
                         " (default runs the fork-in-the-road pair first)")
    ap.add_argument("--loader-warmup", type=int, default=10)
    ap.add_argument("--stage-samples", type=int, default=30,
                    help="30, not 10: per-sample variance was +-20%% in earlier runs")
    ap.add_argument("--clear-frame-cache", action="store_true")
    ap.add_argument("--readpath-frames", type=int, default=36,
                    help="frames per read for the QPESUMS store; 36 == target_2km's "
                         "real per-call shape")
    ap.add_argument("--readpath-repeats", type=int, default=10)
    ap.add_argument("--worker-sweep", type=int, nargs="+", default=[4, 11])
    ap.add_argument("--worker-batches", type=int, default=20)
    ap.add_argument("--gpu-warmup", type=int, default=5)
    ap.add_argument("--gpu-steps", type=int, default=20)
    ap.add_argument("--kernel-steps", type=int, default=3)
    ap.add_argument("--kernel-rows", type=int, default=25)
    ap.add_argument("--kernel-matmul", default="highest",
                    choices=["highest", "high", "medium"])
    ap.add_argument("--fit-steps", type=int, default=60)
    ap.add_argument("--fit-warmup-steps", type=int, default=10)
    ap.add_argument("--fit-precision", default="32")
    ap.add_argument("--fit-matmul", default="highest",
                    choices=["highest", "high", "medium"])
    args = ap.parse_args()

    requested = [x.strip() for x in args.sections.split(",") if x.strip()]
    unknown = [s for s in requested if s not in SECTIONS]
    if unknown:
        raise SystemExit(f"unknown section(s) {unknown}; available: {sorted(SECTIONS)}")

    # Fail in a second with a readable message, not forty lines into zarr.
    # `salloc` hands you a fresh shell, so an exported $QPESUMS/$STA_H8 from an
    # earlier allocation is gone and `--qpesums "$QPESUMS"` silently passes an
    # empty string -- which zarr resolves as a relative path and reports as a
    # missing group in the current working directory.
    if any(s != "checknorm" for s in requested):
        missing = []
        for flag, value in (("--qpesums", args.qpesums), ("--sta-h8", args.sta_h8)):
            for path in (p.strip() for p in value.split(",")):
                if not path:
                    missing.append(f"{flag}: empty (is the shell variable exported?)")
                elif not os.path.exists(path):
                    missing.append(f"{flag}: {path} does not exist")
        if missing:
            raise SystemExit("cannot start:\n  " + "\n  ".join(missing))

    print("RainPro-8-TW training-pipeline profiler")
    print(f"effective_batch={args.effective_batch} micro_batches={args.micro_batches} "
          f"workers={args.num_workers} frame_cache={args.frame_cache_size}")
    print(f"dims={args.dims} norm={args.norm} compile={args.compile}"
          + (f" (mode={args.compile_mode})" if args.compile else ""))
    if args.compile and args.gpu_warmup < 3:
        print("WARNING: --gpu-warmup < 3 with --compile; the first step pays compilation")
    print(f"paper baseline: {PAPER_SECONDS_PER_STEP:.3f} s/optimizer step "
          f"@ batch {PAPER_BATCH}, 1x H100 80GB + {PAPER_VCPUS} vCPUs "
          "(docs/rainpro_paper.md:267,626)")
    print("Tip: run alongside `mpstat -P ALL 1` to tell CPU-busy from I/O-wait.")

    for name in requested:
        SECTIONS[name](args)


if __name__ == "__main__":
    main()
