import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create plots folder
os.makedirs("plots", exist_ok=True)

# Set style
sns.set_theme(style="darkgrid")
plt.rcParams["figure.figsize"] = (12, 6)

def plot_metrics(csv_path, title_prefix, save_prefix):
    df = pd.read_csv(csv_path)
    
    # 1. Total Loss Plot
    plt.figure()
    sns.lineplot(data=df, x="step", y="loss_total")
    plt.title(f"{title_prefix} - Total Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig(f"plots/{save_prefix}_loss.png")
    plt.close()
    
    # 2. Learning Rate Plot
    plt.figure()
    sns.lineplot(data=df, x="step", y="lr")
    plt.title(f"{title_prefix} - Learning Rate")
    plt.xlabel("Step")
    plt.ylabel("LR")
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.savefig(f"plots/{save_prefix}_lr.png")
    plt.close()
    
    # 3. Component Losses (Think vs Response)
    if df["loss_think"].sum() > 0:
        plt.figure()
        plt.plot(df["step"], df["loss_think"], label="Think Loss", alpha=0.7)
        plt.plot(df["step"], df["loss_response"], label="Response Loss", alpha=0.7)
        plt.title(f"{title_prefix} - Component Losses")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"plots/{save_prefix}_components.png")
        plt.close()

    # 4. Gradient Norm
    plt.figure()
    sns.lineplot(data=df, x="step", y="grad_norm")
    plt.title(f"{title_prefix} - Gradient Norm")
    plt.xlabel("Step")
    plt.ylabel("Norm")
    plt.savefig(f"plots/{save_prefix}_grad_norm.png")
    plt.close()

# Plot Phase 1b (Pretraining)
plot_metrics("logs/log_pretrain_phase1b.csv", "Phase 1b Pretraining", "phase1b")

# Plot SFT
plot_metrics("logs/log_sft_20260501_083611.csv", "Phase 2 SFT", "phase2_sft")

print("Plots saved to the 'plots/' folder:")
print(" - phase1b_loss.png, phase1b_lr.png, phase1b_grad_norm.png")
print(" - phase2_sft_loss.png, phase2_sft_lr.png, phase2_sft_components.png, phase2_sft_grad_norm.png")
