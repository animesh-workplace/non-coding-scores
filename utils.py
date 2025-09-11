import os
import torch
import fireducks.pandas as pd


def log_cleanup(INPUT_CSV_PATH, OUTPUT_CSV_PATH):
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find the file at '{INPUT_CSV_PATH}'")
        exit()

    df["epoch"].ffill(inplace=True)
    df["lr-NAdam"].ffill(inplace=True)

    df_cleaned = df.groupby("epoch").first()
    df_cleaned["step"] = df.groupby("epoch")["step"].last()
    df_cleaned.to_csv(OUTPUT_CSV_PATH, sep="\t", index=False)
    print(f"\nMetrics have been saved to: {OUTPUT_CSV_PATH}")


def save_model(model, data, X_tensor, output_path):
    os.makedirs("output", exist_ok=True)

    model.eval()
    with torch.no_grad():
        composite_scores = model.encoder(X_tensor).squeeze().numpy()

    result = data[["chr", "pos", "ref", "alt"]].copy()
    result["composite_score"] = composite_scores
    result.to_feather(f"{output_path}/composite_scores.feather")
    torch.save(model.state_dict(), f"{output_path}/model.pt")
    print(f"Saved composite_scores & model params: {output_path}")
