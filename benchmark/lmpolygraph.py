import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.isotonic import IsotonicRegression

from lm_polygraph.utils.generation_parameters import GenerationParameters
from lm_polygraph.utils.model import BlackboxModel
from lm_polygraph.estimators import PTrueEmpirical, LexicalSimilarity, DegMat, LUQ, KernelLanguageEntropy
from lm_polygraph import estimate_uncertainty


# ----- Estimation + Save Uncertainty -----
def run_estimation(api_key, model_name, temperature, estimator_name, input_path, out_path):
    estimator_map = {
        "PTrueEmpirical": PTrueEmpirical,
        "LexicalSimilarity": LexicalSimilarity,
        "DegMat": DegMat,
        "LUQ": LUQ,
        "KernelLanguageEntropy": KernelLanguageEntropy
    }
    assert estimator_name in estimator_map, f"Estimator '{estimator_name}' not supported."

    model = BlackboxModel(
        openai_api_key=api_key,
        model_path=model_name,
        generation_parameters=GenerationParameters(temperature=temperature)
    )
    estimator = estimator_map[estimator_name]()

    df = pd.read_csv(input_path)
    if os.path.exists(out_path):
        df_out = pd.read_csv(out_path)
        start = len(df_out)
    else:
        df_out = pd.DataFrame(columns=["uncertainty"])
        start = 0

    for i in range(start, len(df)):
        q = df.loc[i, "question"]
        prompt = f"{q} Answer concisely and return only the name."
        ue = estimate_uncertainty(model, estimator, input_text=prompt)
        df_out.loc[i, "uncertainty"] = ue.uncertainty
        df_out.to_csv(out_path, index=False)
        print(f"[{i+1}/{len(df)}] '{q}' → {ue.uncertainty:.4f}")

    print("Finished – results saved to", out_path)


# ----- Evaluation -----
def brier_score(y_true, y_prob):
    return np.mean((y_prob - y_true) ** 2)




def compute_prr(y_true, uncertainty):
    aucpr_unc = average_precision_score(y_true, 1 - uncertainty)
    aucpr_oracle = average_precision_score(y_true, y_true)
    return np.nan if aucpr_oracle == 0 else aucpr_unc / aucpr_oracle



def compute_metrics(y_true, x):
    auroc = roc_auc_score(y_true, x)
    prr = compute_prr(y_true, -x)

    correctness = y_true
   

    iso_model = IsotonicRegression(out_of_bounds='clip')
    x_isotonic = iso_model.fit_transform(x, correctness)
    brier_isotonic = brier_score(correctness, x_isotonic)

    return (
        auroc, prr,  brier_isotonic,
    
    )

def calculate_overall_stats(results_df, correctness_df):
    correctness = correctness_df["Correctness"].values
    n_bootstraps = 100
    results_list = []

    for column in results_df.columns:
        x = -results_df[column].values
        metrics = list(zip(*[
            compute_metrics(correctness[np.random.choice(len(x), len(x), True)],
                            x[np.random.choice(len(x), len(x), True)])
            for _ in range(n_bootstraps)
        ]))

        results_list.append({
            "Uncertainty Measure": column,
            "AUROC": f"{np.mean(metrics[0]):.3f} ± {np.std(metrics[0]):.3f}",
            "PRR": f"{np.mean(metrics[1]):.3f} ± {np.std(metrics[1]):.3f}",
            "Brier_isotonic": f"{np.mean(metrics[4]):.3f} ± {np.std(metrics[4]):.3f}",
            
        })

    final_df = pd.DataFrame(results_list)
    print(final_df)
    return final_df


# ----- Main -----
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", default="your_openai_key")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--model", default="gpt-3.5-turbo")
    parser.add_argument("--estimator", default="LexicalSimilarity", choices=[
        "PTrueEmpirical", "LexicalSimilarity", "DegMat", "LUQ", "KernelLanguageEntropy"
    ])
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    run_estimation(args.api_key, args.model, args.temperature, args.estimator, args.input_path, args.output_path)

