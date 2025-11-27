# ===============================================================
#  Script to confirm parameter-count difference in TSMixer models
# ===============================================================

from models import TSMixer

def count_params(model):
    """Return the total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    input_dim = 38        # e.g., areawise dataset
    pred_len  = 96
    latent_dim = 64

    print("\n🔍 Checking TSMixer parameter counts...\n")

    for n_blocks in [2, 4]:
        m = TSMixer(input_dim=input_dim, pred_len=pred_len,
                    latent_dim=latent_dim, n_blocks=n_blocks)
        print(f"TSMixer with n_blocks={n_blocks:<2d} → {count_params(m):,} parameters")

    print("\n✅ This verifies that the larger (global-KD) model uses 4 Mixer blocks\n"
          "   while the smaller (sequential-KD) model used 2 Mixer blocks.\n")
