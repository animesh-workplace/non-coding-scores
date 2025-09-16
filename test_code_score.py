import numpy as np
import fireducks.pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chisquare, ks_2samp, wasserstein_distance


def check_sampling_distribution(population_file, sample_files, chrom_col="chrom"):
    pop_counts = pd.read_feather(population_file)[chrom_col].value_counts().sort_index()
    pop_dist = pop_counts / pop_counts.sum()
    results = {}
    print("Population chromosome distribution:", pop_dist.to_dict())
    for name, f in sample_files.items():
        print(f"Analyzing sample: {name}")
        sample_counts = pd.read_feather(f)[chrom_col].value_counts().sort_index()
        sample_dist = sample_counts / sample_counts.sum()
        # Align indices
        aligned = pd.concat([pop_dist, sample_dist], axis=1).fillna(0)
        aligned.columns = ["Population", "Sample"]
        # Chi-square goodness of fit
        chi2, pval = chisquare(
            f_obs=aligned["Sample"] * len(sample_counts),
            f_exp=aligned["Population"] * len(sample_counts),
        )
        results[name] = {"distribution": aligned, "chi2": chi2, "pval": pval}
        # Plot comparison
        aligned.plot(kind="bar", figsize=(10, 5), alpha=0.7)
        plt.title(f"Chromosome distribution: {name} vs Population\nChi2 p={pval:.3e}")
        plt.ylabel("Proportion")
        plt.show()
    return results


results = check_sampling_distribution(
    population_file="combined_scores.feather",
    sample_files={
        "1M": "sampled_dataset_1M.feather",
        "10M": "sampled_dataset_10M.feather",
        "25M": "sampled_dataset_25M.feather",
    },
    chrom_col="chr",  # change if your column name differs
)


def compare_score_distributions(name, df_pop, df_samp, score_cols):
    results = {}
    for col in score_cols:
        pop_values = df_pop[col].dropna()
        samp_values = df_samp[col].dropna()
        # KS test
        ks_stat, ks_p = ks_2samp(pop_values, samp_values)
        # Wasserstein distance
        wd = wasserstein_distance(pop_values, samp_values)
        results[col] = {"KS_stat": ks_stat, "KS_p": ks_p, "Wasserstein": wd}
        # Histogram comparison
        plt.figure(figsize=(8, 4))
        plt.hist(pop_values, bins=100, alpha=0.5, label="Population", density=True)
        plt.hist(samp_values, bins=100, alpha=0.5, label="Sample", density=True)
        plt.title(f"Distribution of {col}")
        plt.xlabel("Score")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        hist_path = f"{name}_{col}_hist.png"
        plt.savefig(hist_path, dpi=300)
        plt.close()
        # QQ-plot: population vs sample quantiles
        plt.figure(figsize=(5, 5))
        q_pop = pd.Series(pop_values).quantile(q=np.linspace(0, 1, 1000))
        q_samp = pd.Series(samp_values).quantile(q=np.linspace(0, 1, 1000))
        plt.plot(q_pop, q_samp, "o", alpha=0.5)
        max_val = max(q_pop.max(), q_samp.max())
        min_val = min(q_pop.min(), q_samp.min())
        plt.plot([min_val, max_val], [min_val, max_val], "r--")  # diagonal reference
        plt.title(f"QQ-plot of {col} (Population vs Sample)")
        plt.xlabel("Population Quantiles")
        plt.ylabel("Sample Quantiles")
        plt.tight_layout()
        qq_path = f"{name}_{col}_qq.png"
        plt.savefig(qq_path, dpi=300)
        plt.close()
    return pd.DataFrame(results).T


sample_files = {
    "1M": "sampled_dataset_1M.feather",
    "10M": "sampled_dataset_10M.feather",
    "25M": "sampled_dataset_25M.feather",
}

df_pop = pd.read_feather("combined_scores.feather")
for sample in sample_files.values():
    df_samp = pd.read_feather(sample)
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
    compare_score_distributions(
        df_pop=df_pop,
        df_samp=df_samp,
        score_cols=score_cols,
        name=sample.split(".")[0],
    )
