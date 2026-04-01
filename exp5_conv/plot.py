import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/20260401_224609_conv_sweep.csv")

# build config label
df["config"] = "Cin=" + df["cin"].astype(str) + ", Cout=" + df["cout"].astype(str)

configs = sorted(df["config"].unique())
n = len(configs)

fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=True)

if n == 1:
    axes = [axes]

for ax, config in zip(axes, configs):
    sub = df[df["config"] == config].sort_values("height")

    naive = sub[sub["impl"] == "naive"]
    tiled = sub[sub["impl"] == "tiled"]

    ax.plot(
        naive["height"], naive["gflops"],
        marker="o", linewidth=2.5, markersize=7,
        label="Naive"
    )
    ax.plot(
        tiled["height"], tiled["gflops"],
        marker="s", linewidth=2.5, markersize=7,
        label="Tiled"
    )

    ax.set_title(config)
    ax.set_xlabel("Spatial Size (H = W)")
    ax.set_xticks([256, 512, 1024])
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("Throughput (GFLOP/s)")
axes[0].legend()

fig.suptitle("2D Convolution Throughput vs Spatial Size", fontsize=14)
fig.tight_layout()
fig.savefig("plot.png", dpi=200, bbox_inches="tight")
plt.show()