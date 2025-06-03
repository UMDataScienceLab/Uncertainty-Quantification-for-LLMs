# Inv-Entropy: Uncertainty-Quantification-for-LLMs
A Fully Probabilistic Framework for Uncertainty Quantification in LLMs.
## Abstract
Large language models (LLMs) have transformed natural language processing, but their reliable deployment requires effective uncertainty quantification (UQ). Existing UQ methods are often heuristic and lack a fully probabilistic foundation. This paper begins by providing a theoretical justification for the role of perturbations in UQ for LLMs. We then introduce a dual random walk perspective, modeling input–output pairs as two Markov chains with transition probabilities defined by semantic similarity. Building on this, we propose a fully probabilistic framework based on an inverse model, which quantifies uncertainty by evaluating the diversity of the input space conditioned on a given output through systematic perturbations. Within this framework, we define a new uncertainty measure, Inv-Entropy. A key strength of our framework is its flexibility: it supports various definitions of uncertainty measures, embeddings, perturbation strategies, and similarity metrics. We also propose GAAP, a perturbation algorithm based on genetic algorithms, which enhances the diversity of sampled inputs. In addition, we introduce a new evaluation metric, Temperature Sensitivity of Uncertainty (TSU), which directly assesses uncertainty without relying on correctness as a proxy. Extensive experiments demonstrate that Inv-Entropy outperforms existing semantic UQ methods.

## How to run Inv-Entropy
  - Complete the paraphrasing, response generation, and correctness evaluation by running
    ```
    python pipeline_agent.py inputfile.csv
    ```
  - After running the command above, a folder named inputfile will be created. To compute the inverse entropy and evaluation metrics including AUROC, PRR, and Brier Score, run:
    ```
    python pipeline_metric.py inputfile
    ```

