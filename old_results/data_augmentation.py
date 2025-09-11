import random
import numpy as np
import pandas as pd

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)


def generate_genomic_data(n_rows=100000):
    """Generate synthetic genomic data with 29 columns"""

    # Chromosome options (1-22, X, Y)
    chromosomes = [str(i) for i in range(1, 23)] + ["X", "Y"]

    # Nucleotide bases
    bases = ["A", "T", "G", "C"]

    data = []

    for _ in range(n_rows):
        # Column 1: Chromosome
        chromosome = random.choice(chromosomes)

        # Column 2: Position (genomic coordinate)
        # Use realistic ranges for different chromosomes
        if chromosome == "X":
            position = random.randint(1, 155000000)
        elif chromosome == "Y":
            position = random.randint(1, 59000000)
        else:
            # Autosomal chromosomes - approximate lengths
            chr_lengths = {
                "1": 249000000,
                "2": 243000000,
                "3": 198000000,
                "4": 191000000,
                "5": 181000000,
                "6": 171000000,
                "7": 159000000,
                "8": 146000000,
                "9": 141000000,
                "10": 134000000,
                "11": 135000000,
                "12": 133000000,
                "13": 115000000,
                "14": 107000000,
                "15": 102000000,
                "16": 90000000,
                "17": 81000000,
                "18": 78000000,
                "19": 59000000,
                "20": 63000000,
                "21": 48000000,
                "22": 51000000,
            }
            max_pos = chr_lengths.get(chromosome, 150000000)
            position = random.randint(1, max_pos)

        # Column 3: Reference allele
        ref = random.choice(bases)

        # Column 4: Alternative allele (different from ref)
        alt_options = [b for b in bases if b != ref]
        alt = random.choice(alt_options)

        # Columns 5-29: 25 score columns
        # Generate realistic genomic scores with different distributions
        scores = []

        # Some scores follow normal distribution (e.g., conservation scores)
        for i in range(8):
            score = round(np.random.normal(0.5, 0.2), 4)
            score = max(0, min(1, score))  # Clamp between 0 and 1
            scores.append(score)

        # Some scores follow beta distribution (e.g., allele frequencies)
        for i in range(5):
            score = round(np.random.beta(2, 5), 4)
            scores.append(score)

        # Some scores follow exponential-like distribution (e.g., CADD scores)
        for i in range(4):
            score = round(np.random.exponential(2), 4)
            score = min(score, 50)  # Cap at 50
            scores.append(score)

        # Some scores with occasional NA values (missing data)
        for i in range(5):
            if random.random() < 0.1:  # 10% chance of NA
                scores.append("NA")
            else:
                score = round(np.random.uniform(0, 10), 4)
                scores.append(score)

        # Some integer scores (e.g., read depth)
        for i in range(3):
            score = random.randint(0, 100)
            scores.append(score)

        # Combine all columns
        row = [chromosome, position, ref, alt] + scores
        data.append(row)

    return data


# Generate the data
print("Generating 1000 rows of genomic data...")
genomic_data = generate_genomic_data()

# Create column names
columns = ["chromosome", "position", "ref", "alt"] + [
    f"score_{i + 1}" for i in range(25)
]

# Create DataFrame
df = pd.DataFrame(genomic_data, columns=columns)

# Display first few rows
print("\nFirst 5 rows:")
print(df.head())

print(f"\nDataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Save to TSV file
output_file = "synthetic_genomic_data.tsv"
df.to_csv(output_file, sep="\t", index=False)
print(f"\nData saved to: {output_file}")

# Show some statistics
print("\nData Summary:")
print("=" * 50)
print(f"Chromosomes represented: {sorted(df['chromosome'].unique())}")
print(f"Position range: {df['position'].min()} - {df['position'].max()}")
print(f"Reference alleles: {sorted(df['ref'].unique())}")
print(f"Alternative alleles: {sorted(df['alt'].unique())}")

# Count NA values per column
na_counts = []
for col in df.columns:
    na_count = (df[col] == "NA").sum()
    if na_count > 0:
        na_counts.append((col, na_count))

if na_counts:
    print("\nNA counts by column:")
    for col, count in na_counts:
        print(f"  {col}: {count}")

# Sample of the generated data
print("\nSample rows (formatted):")
print("=" * 100)
for i in range(3):
    row = df.iloc[i]
    print(f"Row {i + 1}:")
    print(f"  Chromosome: {row['chromosome']}")
    print(f"  Position: {row['position']}")
    print(f"  Ref: {row['ref']} -> Alt: {row['alt']}")
    print(f"  Scores: {list(row[4:9])}... (showing first 5 of 25)")
    print()
