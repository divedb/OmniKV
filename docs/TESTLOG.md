# Test Log

Commands, results, failures and fixes from remote validation runs.
Append-only; newest entries last.

## 2026-11-20 — Environment setup

| Step | Command | Result |
|---|---|---|
| SSH key auth | `ssh -p 44740 root@connect.nmb2.seetacloud.com 'hostname'` | OK (`autodl-container-dfepxcx0ax-8e77fec0`) |
| GPU audit | `nvidia-smi --query-gpu=...` | RTX 3080 Ti, 12 GiB, driver 595.71.05, sm_86 |
| Disk audit | `df -h /root/autodl-tmp` | 50 GB disk, 3.5 GB free (cleanup pending) |
| Python audit | `ls /root/miniconda3/bin/python3` | 3.12.3 (no PATH in non-interactive SSH) |
| Disk cleanup | `rm -rf FlexKV/ssd_cache FlexKV/ssd_cache2` | 32 GB freed (owner-approved); 35 GB available |
| uv install (remote) | standalone installer → `~/.local/bin/uv` | uv 0.13.9 |
| Project sync | `scripts/sync_remote.sh` | `/root/autodl-tmp/omnikv` created (69 KB sent) |
| GPU stack install | `uv pip install vllm==0.34.0` (Tsinghua mirror) | in progress → result recorded below when complete |
