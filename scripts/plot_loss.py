import os
import json
from pathlib import Path
import matplotlib.pyplot as plt

project_root_path = Path(__file__).resolve().parent.parent 

losses_path = project_root_path / "results" / "training_log.json"
with losses_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

plt.figure(figsize=(8, 5))

if data["train"]["steps"]:
    plt.plot(data["train"]["steps"], data["train"]["loss"], label="Train Loss")

if data["eval"]["steps"]:
    plt.plot(data["eval"]["steps"], data["eval"]["loss"], label="Eval Loss", marker='o')

plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training and Evaluation Loss")
plt.legend()
plt.grid(True)
# plt.savefig("loss_plot.png")
plt.show()