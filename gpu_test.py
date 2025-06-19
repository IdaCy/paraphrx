# gpu_test.py — 30-line smoke test that really uses the GPU
import torch, time, socket, os

print("node :", socket.gethostname())
print("cwd  :", os.getcwd())
print("cuda :", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA NOT AVAILABLE")

print("gpu  :", torch.cuda.get_device_name(0))
# do a tiny workload so you can see the GPU is active
a = torch.randn(4096, 4096, device="cuda")
b = torch.randn(4096, 4096, device="cuda")
t0 = time.time()
c = a @ b
torch.cuda.synchronize()
print("matmul OK; elapsed:", round(time.time() - t0, 3), "s")
print("checksum :", c.float().mean().item())
