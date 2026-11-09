import os, shutil
import fiftyone.zoo as foz

ds = foz.load_zoo_dataset(
    "https://github.com/AdonaiVera/bddoia-fiftyone",
    split="train",
)

out = "data_bddoia"
for d in ("MOVE", "STOP"):
    os.makedirs(f"{out}/{d}", exist_ok=True)

limit = 3000
counts = {"MOVE": 0, "STOP": 0}

first = ds.first()
print("Example labels:", [c.label for c in first.ground_truth.classifications])

for s in ds.shuffle(seed=42).iter_samples():
    labels = {c.label for c in s.ground_truth.classifications}

    if "forward" in labels and "stop" not in labels:
        folder = "MOVE"
    elif "stop" in labels and "forward" not in labels:
        folder = "STOP"
    else:
        continue  # turns, mixed, or unclear: skip

    if counts[folder] >= limit:
        continue

    shutil.copy(s.filepath, f"{out}/{folder}/{os.path.basename(s.filepath)}")
    counts[folder] += 1

    if all(c >= limit for c in counts.values()):
        break

print(counts)
