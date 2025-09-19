import torch
import fireducks.pandas as pd
from autoencoders.base_ae import AutoEncoder
from autoencoders.m_base import BigAutoEncoder
from autoencoders.denoising_ae import DenoisingAutoEncoder
from autoencoders.binary_mask_dae import MaskedDenoisingAutoEncoder


model_class = {
    # "base": AutoEncoder,
    "big_ae": BigAutoEncoder,
    # "denoising": DenoisingAutoEncoder,
    # "masked_denoising": MaskedDenoisingAutoEncoder,
}

df = pd.read_feather("data/testing_data.feather")
score_cols = [
    "gpn",
    "cadd",
    "remm",
    "dann",
    "fire",
    "gerp",
    "cscape",
    "gwrvis",
    "jarvis",
    "funseq2",
    "linsight",
    "repliseq_g2",
    "repliseq_s1",
    "repliseq_s2",
    "repliseq_s3",
    "repliseq_s4",
    "repliseq_g1b",
    "macie_conserved",
    "macie_regulatory",
    "fathmm_mkl_coding",
    "fathmm_xf_noncoding",
    "fathmm_mkl_noncoding",
    "conservation_30p",
    "conservation_100v",
]
X = df[score_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
X_tensor = torch.tensor(X, dtype=torch.float32)
print("✅ Testing data loaded")

for model_name in model_class.keys():
    sample_size = [1]
    for size in sample_size:
        print(f"Processing model: {model_name} Size:{size}M")
        # ---------------------------
        # Load model
        # ---------------------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model_class[model_name]()
        model.load_state_dict(
            torch.load(
                f"output/training_{size}m/{model_name}/model.pt",
                map_location=device,
            )
        )
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully")

        with torch.no_grad():
            if model_name == "masked_denoising":
                zero_mask = torch.zeros_like(X_tensor).to(device)
                composite_scores = (
                    model.encoder(torch.cat([X_tensor.to(device), zero_mask], dim=1))
                    .squeeze()
                    .cpu()
                    .numpy()
                )
            else:
                # For other models, use standard encoder call
                composite_scores = (
                    model.encoder(X_tensor.to(device)).squeeze().cpu().numpy()
                )

        print(composite_scores, composite_scores.shape)
        df[f"score_{model_name}_{size}M"] = composite_scores
        print(f"✅ Composite scores generated for {model_name}_{size}M")

df.to_feather("output/all_composite_score_test_data.feather")
df.to_csv("output/all_composite_score_test_data.tsv", sep="\t", index=False)
