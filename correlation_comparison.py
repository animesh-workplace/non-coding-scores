import fireducks.pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Define File Paths and Feature Names ---

# IMPORTANT: Replace these with the actual paths to your files
composite_scores_file = (
    "output/composite_scores_20250910_191838.feather"  # Example path
)
original_data_file = "data/combined_scores.feather"  # Example path

# List of your 24 original score columns
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

# --- 2. Load and Merge Data ---

print("Loading data...")
# Load your generated composite scores
df_composite = pd.read_feather(composite_scores_file)

# Load the original dataset, but only the columns you need
# We need the key columns ('chr', 'pos', 'ref', 'alt') and the score columns
df_original = pd.read_feather(original_data_file)

# Merge the two dataframes to align composite scores with original features
print("Merging dataframes...")
df_merged = pd.merge(df_composite, df_original, on=["chr", "pos", "ref", "alt"])
print("Data merged successfully.")


# --- 3. Calculate Correlations ---

print("Calculating correlations...")
# Calculate the correlation of every score_col with the 'composite_score'
correlations = (
    df_merged[score_cols + ["composite_score"]]
    .corr()["composite_score"]
    .drop("composite_score")
)

# Sort the correlations for better visualization
correlations_sorted = correlations.sort_values(ascending=False)

print("\n--- Correlation Results ---")
print(correlations_sorted)
print("---------------------------\n")


# --- 4. Visualize the Results ---

print("Generating plot...")
plt.figure(figsize=(12, 10))
sns.barplot(
    x=correlations_sorted.values, y=correlations_sorted.index, palette="viridis"
)
plt.title("Correlation of Original Features with the Composite Score", fontsize=16)
plt.xlabel("Pearson Correlation Coefficient", fontsize=12)
plt.ylabel("Original Features", fontsize=12)
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.tight_layout()

# Save the plot to a file
output_plot_file = "composite_score_correlations.png"
plt.savefig(output_plot_file, dpi=300)

print(f"Correlation plot saved to '{output_plot_file}'")
# To display the plot in a notebook, you can use:
# plt.show()
