import subprocess
import re
from statistics import mean
from itertools import product

configs = ["modelnet40", "scanobjectnn"]
seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
shots = [1, 2, 4, 8, 16]

results = {cfg: {shot: [] for shot in shots} for cfg in configs}

pattern = re.compile(r"PointGMDA: (\d+\.\d+)")

for config, shot, seed in product(configs, shots, seeds):
    cmd = f"python main_few_shots.py --config configs/{config}.yaml --seed {seed} --shot {shot} --ckpt_path ckpt/openshape-pointbert-no-lvis.pt"
    print(f"Running: {cmd}")

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")
        print(f"Stderr: {e.stderr}")
        print(f"Stdout: {e.stdout}")
        continue

    match = pattern.search(result.stdout)
    if match:
        acc = float(match.group(1))
        results[config][shot].append(acc)
        print(f"Config: {config}, Shot: {shot}, Seed: {seed}, Accuracy: {acc}")
    else:
        print(f"Error parsing output for {config} {shot} {seed}")
        print("Output was:")
        print(result.stdout)
        print("Stderr:")
        print(result.stderr)

for config in configs:
    print(f"\nResults for {config}:")
    for shot in shots:
        avg = mean(results[config][shot]) if results[config][shot] else 0
        print(f"Shot {shot}: Average Accuracy = {avg:.4f}")
