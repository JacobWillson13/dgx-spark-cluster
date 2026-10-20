# Setup

Build a two-node DGX Spark cluster with a direct 200 GbE ConnectX-7 link and serve models across both nodes with native vLLM.

Every command block is labeled with the machine it runs on: `spark1`, `spark2`, or `laptop`.

## Addresses

No address in this guide is real. Set these once on every machine, using your own values:

```bash
# laptop, spark1 and spark2
export SPARK1_MGMT=<spark1 management IP>
export SPARK2_MGMT=<spark2 management IP>
export SPARK1_CX7_A=<spark1 address on enp1s0f1np1>
export SPARK2_CX7_A=<spark2 address on enp1s0f1np1>
export SPARK1_CX7_B=<spark1 address on enP2p1s0f1np1>
export SPARK2_CX7_B=<spark2 address on enP2p1s0f1np1>
```

The management addresses are a private subnet you choose. The four ConnectX-7 addresses are assigned by NVIDIA's Cluster Assistant in step 3.

---

## 1. Management network

A private `/24` with static addresses and no router or DHCP. Without a DHCP server, DHCP-configured interfaces fall back to link-local addresses, so set every address by hand. The laptop keeps Wi-Fi for internet access at the same time.

On each Spark, find the wired management interface's NetworkManager profile and set its address (`SPARK1_MGMT` on spark1, `SPARK2_MGMT` on spark2):

```bash
# spark1 (spark2 identical with its own address)
nmcli -f NAME,DEVICE,TYPE connection show
sudo nmcli connection modify "<profile>" ipv4.method manual \
  ipv4.addresses "$SPARK1_MGMT/24" ipv4.gateway "" ipv4.dns "" connection.autoconnect yes
sudo nmcli connection up "<profile>"
```

Add SSH aliases on the laptop, then copy your key to both nodes:

```text
# laptop: ~/.ssh/config
Host spark1
  HostName <spark1 management IP>
  User <user>
Host spark2
  HostName <spark2 management IP>
  User <user>
```

```bash
# laptop
ssh-copy-id spark1 && ssh-copy-id spark2
```

## 2. Matching software

Both nodes must run the same release. Mismatched kernels mean mismatched `mlx5` drivers, which makes any interconnect measurement meaningless. Update through the DGX Dashboard, which coordinates firmware with packages, then confirm:

```bash
# spark1 and spark2
uname -r
nvidia-smi --query-gpu=driver_version --format=csv,noheader
id "$USER"
```

Kernel and driver must match exactly. The username, UID and GID must match too: multi-node launches assume one identity across the cluster.

## 3. ConnectX-7 link

Connect the QSFP cable to the same physical port on both nodes.

```bash
# spark1 and spark2
ibdev2netdev
```

```text
rocep1s0f0   port 1 ==> enp1s0f0np0   (Down)
rocep1s0f1   port 1 ==> enp1s0f1np1   (Up)
roceP2p1s0f0 port 1 ==> enP2p1s0f0np0 (Down)
roceP2p1s0f1 port 1 ==> enP2p1s0f1np1 (Up)
```

Each physical port appears as **two interfaces, one per PCIe path**. `enp1s0f1np1` and `enP2p1s0f1np1` are the same wire. Ethernet tools take the interface name; RDMA tools take the left-hand device name.

Add both Sparks to NVIDIA Sync on the laptop, then run **Cluster Assistant → Add New Cluster**. It checks usernames and UIDs, assigns two addresses per node (one per PCIe path, on separate `/24` subnets), exchanges SSH keys between the nodes and runs a link test. Record the four addresses it assigns in the variables above.

## 4. Jumbo frames

RoCE path MTU follows the interface MTU: 1500 gives 1024, 9000 gives 4096. Add `mtu: 9000` to the four ConnectX-7 interfaces in the Assistant's netplan file on both nodes:

```bash
# spark1 and spark2
F=/etc/netplan/99-nvidia-sync-cluster.yaml
sudo cp "$F" "$F.bak"
sudo sed -i '/dhcp4: false/a\      mtu: 9000' "$F"
sudo cat "$F"                       # check all four interfaces before applying
sudo netplan generate && sudo netplan apply
ip link show enp1s0f1np1 | head -1  # shows mtu 9000
```

Netplan reads only `*.yaml`, so the backup is ignored. Re-running the Cluster Assistant may overwrite the file.

## 5. Validate the link

Drive both PCIe paths at once. One path cannot saturate the port.

```bash
# spark2 (server)
sudo apt install -y perftest
ib_write_bw -d rocep1s0f1   -F --report_gbits -q 8 -D 20 -p 18515 > /tmp/bw1.txt 2>&1 &
ib_write_bw -d roceP2p1s0f1 -F --report_gbits -q 8 -D 20 -p 18516 > /tmp/bw2.txt 2>&1 &
```

```bash
# spark1 (client)
sudo apt install -y perftest
ib_write_bw -d rocep1s0f1   -F --report_gbits -q 8 -D 20 -p 18515 "$SPARK2_CX7_A" > /tmp/bw1.txt 2>&1 &
ib_write_bw -d roceP2p1s0f1 -F --report_gbits -q 8 -D 20 -p 18516 "$SPARK2_CX7_B" > /tmp/bw2.txt 2>&1 &
wait; tail -n 3 /tmp/bw1.txt /tmp/bw2.txt
```

Expect about 98 Gb/s per path and 196 Gb/s combined at MTU 9000, and about 112 Gb/s from either path alone. Measure latency with `ib_write_lat` (about 1.7 µs), not `ping`, which goes through the kernel stack and reports milliseconds.

## 6. Native vLLM

vLLM 0.29.0 installs and runs natively on the GB10 (sm_121); no container is needed. Build the same environment on both nodes:

```bash
# spark1 and spark2
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12
uv venv ~/venvs/vllm029 --python 3.12
source ~/venvs/vllm029/bin/activate
uv pip install vllm==0.29.0 --torch-backend=auto
uv pip install "ray[default]"
python -c "import vllm, torch, ray; print(vllm.__version__, torch.__version__, torch.cuda.get_device_capability(), ray.__version__)"
```

Expect `0.29.0`, a cu132 PyTorch build, `(12, 1)` and `2.58.0` (or whatever Ray resolves; it must match on both nodes). Do **not** install `flashinfer-cubin`: its version check hard-fails against FlashInfer 0.6.18.

Prepare memory on both nodes. On unified memory, swap and the page cache compete with the model:

```bash
# spark1 and spark2
sudo swapoff -a                       # and remove the swap entry from /etc/fstab
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
```

[`docs/VALIDATION.md`](VALIDATION.md) section 1.1 sets up a password-free rule for exactly that cache-drop command, so automated runs can do it before every launch.

## 7. Models

Download weights directly on the Sparks, never through the laptop. Split downloads across both nodes, then copy over the ConnectX-7 link:

```bash
# spark1 and spark2
uv tool install "huggingface_hub[cli]"
export HF_XET_HIGH_PERFORMANCE=1
mkdir -p ~/models
nohup hf download <repo> --local-dir ~/models/<name> > ~/dl.log 2>&1 &
```

```bash
# spark1, copying a model it downloaded to spark2
rsync -a --exclude .cache ~/models/<name>/ "$USER@$SPARK2_CX7_A:models/<name>/"
```

Multi-node vLLM needs every model at the **same absolute path** on both nodes. To verify a download, re-run the same `hf download` command: it checks every file and returns at once when nothing is missing. Size and file count can match while a shard is truncated.

## 8. Serve on one node

```bash
# spark1
source ~/venvs/vllm029/bin/activate
vllm serve ~/models/Qwen3.8-27B --served-model-name qwen \
  --max-model-len 8192 --gpu-memory-utilization 0.85 \
  --generation-config vllm --load-format fastsafetensors --port 8000
```

- `--load-format fastsafetensors` loads weights about 44x faster than the default loader.
- `--generation-config vllm` stops each model's own sampling defaults from silently differing between runs.
- `0.85` leaves room for the OS on unified memory; the default 0.92 fails the startup check.

## 9. Serve across both nodes

Tensor parallelism across hosts needs Ray. Export the environment on **both** nodes before `ray start`; Ray workers inherit the environment of the shell that started Ray, not a later one.

```bash
# spark1 (on spark2, use VLLM_HOST_IP=$SPARK2_CX7_A)
source ~/venvs/vllm029/bin/activate
export CUDA_HOME=/usr/local/cuda PATH="/usr/local/cuda/bin:$PATH"
export VLLM_HOST_IP=$SPARK1_CX7_A
export NCCL_SOCKET_IFNAME=enp1s0f1np1,enP2p1s0f1np1
export GLOO_SOCKET_IFNAME=enp1s0f1np1,enP2p1s0f1np1
export NCCL_IB_MERGE_NICS=1
export VLLM_ALLREDUCE_USE_FLASHINFER=0
export MAX_JOBS=2 FLASHINFER_NVCC_THREADS=2
```

- `GLOO_SOCKET_IFNAME` is required. Without it, Gloo binds one rank to loopback and launch fails with `Gloo connectFullMesh failed`.
- `MAX_JOBS` and `FLASHINFER_NVCC_THREADS` bound just-in-time kernel builds, which otherwise can exhaust memory with the model already loaded.

```bash
# spark1
ray start --head --node-ip-address "$SPARK1_CX7_A" --port 6379
```

```bash
# spark2
ray start --address "$SPARK1_CX7_A:6379"
```

```bash
# spark1
ray status                                   # 2 nodes, 2.0 GPU
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
vllm serve ~/models/Qwen3.8-27B --served-model-name qwen \
  --tensor-parallel-size 2 --distributed-executor-backend ray \
  --max-model-len 8192 --gpu-memory-utilization 0.80 \
  --generation-config vllm --load-format fastsafetensors --port 8000
```

Use `0.80` for two-node runs. The server log should list both RoCE devices on the `NET/IB : Using` line, confirming NCCL uses both rails.

Stop Ray on both nodes with `ray stop` when done.

Before trusting any throughput number from a configuration, run the correctness gate in [`docs/VALIDATION.md`](VALIDATION.md).
