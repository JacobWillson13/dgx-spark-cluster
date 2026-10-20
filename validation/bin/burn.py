import torch, time
a = torch.randn(8192, 8192, device='cuda', dtype=torch.float16)
b = torch.randn(8192, 8192, device='cuda', dtype=torch.float16)
for _ in range(5): a @ b
torch.cuda.synchronize()
t0 = time.time(); n = 0
while time.time() - t0 < 15:
    for _ in range(10): a @ b
    n += 10
torch.cuda.synchronize()
dt = time.time() - t0
print(f"{2*8192**3*n/dt/1e12:.1f} TFLOP/s fp16")
