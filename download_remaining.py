import os, subprocess, sys

os.environ['HF_TOKEN'] = os.environ.get('HF_TOKEN', '')

local_dir = "/workspace/data/lvbench_lmms"
existing = {f for f in os.listdir(f"{local_dir}/video_chunks") if f.endswith('.zip')}
all_chunks = {f"videos_chunk_{i:03d}.zip" for i in range(1, 15)}
missing = sorted(all_chunks - existing)

print(f"Have: {len(existing)}/14 | Missing: {len(missing)}")

for f in missing:
    path = f"video_chunks/{f}"
    dest = f"{local_dir}/video_chunks/{f}"
    print(f"\n>>> {f}...", end=" ", flush=True)
    r = subprocess.run([
        sys.executable, "-c",
        "import os;"
        "from huggingface_hub import hf_hub_download;"
        f"hf_hub_download('lmms-lab/LVBench','{path}',repo_type='dataset',local_dir='{local_dir}')"
    ], capture_output=True, text=True, timeout=1800)
    if r.returncode == 0:
        print(f"OK ({os.path.getsize(dest)/1e9:.1f} GB)")
    else:
        print(f"FAIL: {r.stderr[-200:]}")
