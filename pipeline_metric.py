from evaluate import process_csv, calculate_overall_stats
import pandas as pd
import sys
import os

if len(sys.argv) < 2:
    print("Usage: python script.py <folder>")
    sys.exit(1)

folder = sys.argv[1]
if not os.path.isdir(folder):
    print(f"Error: Folder '{folder}' not found.")
    sys.exit(1)

responded_path = os.path.join(folder, 'responded.csv')
correctness_path = os.path.join(folder, 'correctness.csv')

# Load responded.csv
df = pd.read_csv(responded_path)
results_dfs, matrix = process_csv(df)

for n in results_dfs:
    output_path = os.path.join(folder, f"mean.{n}.csv")
    results_dfs[n].to_csv(output_path, index=False)

matrix_path = os.path.join(folder, 'matrix.csv')
matrix.to_csv(matrix_path, index=False, header=True)

# Load correctness.csv
correctness_df = pd.read_csv(correctness_path)
for n in results_dfs:
    results_df = results_dfs[n]
    df_final = calculate_overall_stats(results_df, correctness_df)
    final_path = os.path.join(folder, f'final.{n}.csv')
    df_final.to_csv(final_path)