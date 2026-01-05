import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import time
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

# List available devices
devices = []
if torch.backends.mps.is_available():
    devices.append(("MPS", torch.device("mps")))
devices.append(("CPU", torch.device("cpu")))
if torch.cuda.is_available():
    devices.append(("CUDA"), torch.device("cuda"))

if not devices:
    raise RuntimeError("No available devices")

print("Available devices :", [name for name, _ in devices])


torch.manual_seed(42)
BATCH_SIZE = 512
N_SAMPLES = 500
INPUT_SIZE = 512
HIDDEN_SIZE = 2048
OUTPUT_SIZE = 1

X = torch.randn(N_SAMPLES, INPUT_SIZE)
y = torch.randn(N_SAMPLES, OUTPUT_SIZE)

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

def benchmark_training(device_name, device, X, y, batch_size=64, epochs=3):
    X_dev = X.to(device)
    y_dev = y.to(device)
    model = SimpleNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_dataset = TensorDataset(X_dev, y_dev)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    total_batches = len(train_loader) * epochs

    # Progress: total batches
    pbar = tqdm(
        total=total_batches,
        desc=f"Learn using {device_name}",
        unit="batch",
        leave=False,
        dynamic_ncols=True
    )

    start_time = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            pbar.update(1)
    pbar.close()

    total_time = time.perf_counter() - start_time
    return total_time / epochs


def benchmark_inference(device_name, device, input_size, hidden_size, output_size, batch_size=64, num_runs=200):
    model = SimpleNet(input_size, hidden_size, output_size).to(device).eval()
    dummy_input = torch.randn(batch_size, input_size, device=device)

    def infer():
        with torch.no_grad():
            return model(dummy_input)

    # Warm-up
    _ = infer()

    # Inference progress
    pbar = tqdm(
        total=num_runs,
        desc=f"Inference using {device_name}",
        unit="run",
        leave=False,
        dynamic_ncols=True
    )

    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = infer()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        pbar.update(1)
    pbar.close()

    return sum(times) / len(times)


results = {}
for name, device in devices:
    print(f"\n Run benchmark using {name}...")
    train_time = benchmark_training(name, device, X, y, batch_size=BATCH_SIZE, epochs=50)
    inf_time = benchmark_inference(name, device, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, batch_size=BATCH_SIZE, num_runs=200)
    results[name] = {
        "train_time_per_epoch_sec": train_time,
        "inference_time_per_batch_sec": inf_time,
        "inference_time_per_sample_ms": inf_time / BATCH_SIZE * 1000
    }

print("\n Result \n".center(80, "="))
print(f"{'Device':<10} | {'Epoch (sec)':<12} | {'Inference/batch (ms)':<20} | {'Sample (mcs)':<15}")
print("-" * 80)

base_train = results["CPU"]["train_time_per_epoch_sec"]
base_inf_batch = results["CPU"]["inference_time_per_batch_sec"]

for name, _ in devices:
    if name not in results:
        continue

    result = results[name]
    train_sp = base_train / result["train_time_per_epoch_sec"] if name != "CPU" else 1.0
    inf_sp = base_inf_batch / result["inference_time_per_batch_sec"] if name != "CPU" else 1.0

    train_str = f"{result['train_time_per_epoch_sec']:.3f}" + (f" (×{train_sp:.1f})" if name != "CPU" else "")
    inf_str = f"{result['inference_time_per_batch_sec']*1000:.3f}" + (f" (×{inf_sp:.1f})" if name != "CPU" else "")

    print(
        f"{name:<10} | {train_str:<18} | {inf_str:<24} | {result['inference_time_per_sample_ms']*1000:<15.1f}"
    )

print("=" * 80)