import os
import time
import torch
import pandas as pd
import numpy as np
from thop import profile  # pip install thop

# === Import your models ===
from models.RetinaLite_Net import RetinaLiteNet
from models.UNeXt import UNeXt
from models.LTMSegNet import LTMSegNet
from models.ESDMR_Net import ESRMNet
from models.MLU_Net import MLUNet
from models.LV_Net import LV_UNet
from models.LEA_Net import LeaNet
from models.LAMFF_Net import LAMFFNet
from models.LWBNA_Net import LWBNA_Unet
from models.PMFS_Net import PMFSNet

# === Instantiate models ===
models = [
    UNeXt(num_classes=1, input_channels=3),
    LTMSegNet(in_channels=3, base_ch=(16,32,64,128), num_classes=1),
    ESRMNet(in_channels=3, num_classes=1),
    RetinaLiteNet(in_channels=3, num_classes=1, num_heads=4, d_k=32),
    MLUNet(in_channels=3, out_channels=1),
    LV_UNet(),
    LeaNet(in_channels=3, out_channels=1),
    LAMFFNet(in_channels=3, num_classes=1, base_c=32),
    LWBNA_Unet(in_channels=3, num_classes=1),
    PMFSNet(in_channels=3, n_classes=1)
]

# === Utils ===
def count_params(model):
    return sum(p.numel() for p in model.parameters())

def compute_flops_params(model, input_size=(1,3,224,224), device="cpu"):
    """
    Compute FLOPs and Params using a fixed dummy input.
    Ensures consistency across datasets.
    """
    model = model.to(device)
    model.eval()
    dummy = torch.randn(input_size).to(device)  ### FIXED: Always same input size
    macs, params = profile(model, inputs=(dummy,), verbose=False)
    flops = macs * 2  # FLOPs convention (2*MACs)
    return flops, params

def measure_inference_time(model, device='cuda', input_size=(1,3,224,224),
                           warmup=50, runs=200):
    model = model.to(device)
    model.eval()
    dummy = torch.randn(input_size).to(device)  ### FIXED: Always same input size

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)

    # Measure
    times = []
    with torch.no_grad():
        for _ in range(runs):
            if device.startswith("cuda"):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(dummy)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
            else:  # CPU or MPS
                t0 = time.perf_counter()
                _ = model(dummy)
                t1 = time.perf_counter()
            times.append(t1 - t0)

    times = np.array(times)
    return times.mean(), times.std(), times  # seconds

def measure_peak_memory(model, device='cuda', input_size=(1,3,224,224), warmup=10):
    model = model.to(device)
    model.eval()
    dummy = torch.randn(input_size).to(device)  ### FIXED: Always same input size

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy)
            torch.cuda.synchronize()
            _ = model(dummy)
            torch.cuda.synchronize()
        peak_bytes = torch.cuda.max_memory_allocated(device=device)
        peak_mb = peak_bytes / (1024 ** 2)
        return peak_mb
    else:
        # For CPU: heuristic
        import psutil
        proc = psutil.Process(os.getpid())
        before = proc.memory_info().rss
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy)
        after = proc.memory_info().rss
        used_mb = (after - before) / (1024 ** 2)
        return used_mb

# === Main evaluation ===
if __name__ == "__main__":
    # Select device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    Results = {}
    fixed_input_size = (1, 3, 224, 224)  ### FIXED

    for model in models:
        model_name = model.__class__.__name__
        print(f"\n=== Evaluating {model_name} ===")

        # Compute FLOPs and Params
        flops, params = compute_flops_params(model, input_size=fixed_input_size, device=device)
        print(f"Params (count): {params:,}")
        print(f"FLOPs (approx): {flops:,}")

        # Measure inference time
        mean_t, std_t, all_times = measure_inference_time(model, device=device,
                                                          input_size=fixed_input_size,
                                                          warmup=50, runs=200)
        print(f"Inference time: {mean_t*1000:.4f} ms ± {std_t*1000:.4f} ms")

        # Measure peak memory
        peak_mb = measure_peak_memory(model, device=device,
                                      input_size=fixed_input_size, warmup=10)
        print(f"Peak memory during forward: {peak_mb:.2f} MB")

        results = {
            "model": model_name,
            "params": int(params),
            "flops": float(flops),
            "infer_time_mean_s": float(mean_t),
            "infer_time_std_s": float(std_t),
            "peak_memory_mb": float(peak_mb),
            "device": device,
            "input_size": fixed_input_size,
            "batch_size": 1,
            "precision": "fp32"
        }
        Results[model_name] = results

    # Save all results to CSV
    df = pd.DataFrame.from_dict(Results, orient='index')
    df.to_csv("model_performance_summary.csv")
