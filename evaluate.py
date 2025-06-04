import random
import numpy as np
import pandas as pd
import nltk
from sentence_transformers import SentenceTransformer, util
import ast
from scipy.stats import wasserstein_distance
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from sklearn.isotonic import IsotonicRegression




nltk.download('punkt')

sbert_device = "cuda"
sbert_model = SentenceTransformer('all-mpnet-base-v2', device=sbert_device)


def row_normalize(matrix):
    row_sums = matrix.sum(axis=1, keepdims=True)
    return matrix / row_sums
def compute_entropy(prob_vector):
    prob_vector = prob_vector[prob_vector > 0]  
    return -np.sum(prob_vector * np.log2(prob_vector))

def compute_conditional_entropy(prob_matrix, axis):
    entropy_values = np.zeros(prob_matrix.shape[axis])
    
    if axis == 1:  
        for i in range(prob_matrix.shape[0]):
            row = prob_matrix[i, :]
            entropy_values[i] = compute_entropy(row)
    else:  
        for j in range(prob_matrix.shape[1]):
            col = prob_matrix[:, j]
            entropy_values[j] = compute_entropy(col)
    
    return np.sum(entropy_values)  

def compute_joint_entropy(pxy):
    
    pxy_nonzero = pxy[pxy > 0] 
    return -np.sum(pxy_nonzero * np.log2(pxy_nonzero))

def kl_divergence(p, q):
    p = p.flatten()
    q = q.flatten()
    mask = (p > 0) & (q > 0)  
    return np.sum(p[mask] * np.log2(p[mask] / q[mask]))

def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))


def parse_matrix(matrix_str):
    """将字符串转换为 numpy 矩阵"""
    return np.array(ast.literal_eval(matrix_str))

def sbert_embeddings(texts):
    unique_texts = list(dict.fromkeys(texts)) 
    embeddings = {text: sbert_model.encode(text, convert_to_tensor=True) for text in unique_texts}
    return embeddings

def compute_similarity_matrix(embeddings, texts):
    n = len(texts)
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):  
            similarity = abs(util.cos_sim(embeddings[texts[i]], embeddings[texts[j]]).item())
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity  
    return similarity_matrix

def row_normalize(matrix):
    row_sums = matrix.sum(axis=1, keepdims=True)
    return matrix / row_sums

def compute_entropy(prob_vector):
    prob_vector = prob_vector[prob_vector > 0]  
    return -np.sum(prob_vector * np.log2(prob_vector))

def compute_conditional_entropy(prob_matrix, axis):
    entropy_values = np.zeros(prob_matrix.shape[axis])
    
    if axis == 1:  
        for i in range(prob_matrix.shape[0]):
            row = prob_matrix[i, :]
            entropy_values[i] = compute_entropy(row)
    else:  
        for j in range(prob_matrix.shape[1]):
            col = prob_matrix[:, j]
            entropy_values[j] = compute_entropy(col)
    
    return np.sum(entropy_values)  

def compute_joint_entropy(pxy):
    """ 计算联合熵 H(X, Y) = -sum p(x,y) log p(x,y) """
    pxy_nonzero = pxy[pxy > 0]  
    return -np.sum(pxy_nonzero * np.log2(pxy_nonzero))

def kl_divergence(p, q):
    p = p.flatten()
    q = q.flatten()
    mask = (p > 0) & (q > 0)  
    return np.sum(p[mask] * np.log2(p[mask] / q[mask]))

def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))

def process_csv(df):
    df = df.drop(columns=['question', 'value'])
    df = df.iloc[:,1:]

    results = []
    matrices = []
    
    for idx, row in df.iterrows():
        
        parsed_row = [ast.literal_eval(cell) for cell in row]
        
        # Get questions and answers
        x = []  # question
        y = []  # answer
        
        for entry in parsed_row:
            question = entry[0]
            answers = entry[1:]
            for answer in answers:
                x.append(question)
                y.append(answer)
        
        # compute embedding
        x_embeddings = sbert_embeddings(x)
        y_embeddings = sbert_embeddings(y)
        
        # compute similarity matrix
        simX = compute_similarity_matrix(x_embeddings, x)
        simY = compute_similarity_matrix(y_embeddings, y)

        matrices.append([simX.tolist(), simY.tolist()])
        
        # Calculate row normalization matrix
        px = row_normalize(simX)
        py = row_normalize(simY)
        
        # calculate n
        n = px.shape[0]
        
        # construct pi_uniform
        pi_uniform = np.full((1, n), 1/n)
        
        # compute px_marginal, px_y, py_marginal
        py_marginal_I = (pi_uniform @ py).T
        px_y_I = py @ px
        px_marginal_I = (pi_uniform @ py @ py @ px).T

        # compute py_x by baye's rule
        py_x_I = (px_y_I * py_marginal_I) / px_marginal_I.T

        # compute pxy
        pxy_I  = px_y_I * py_marginal_I

        #The index of x determines the row, and the index of y determines the column.
        px_y_I = px_y_I.T
        py_x_I = py_x_I.T
        pxy_I = pxy_I.T

       
        WD_px_py_I = wasserstein_distance(px_marginal_I.flatten(), py_marginal_I.flatten())

        entropy_y_x_I = -np.sum(np.diag(py_x_I) * np.log(np.diag(py_x_I)))
        
        entropy_x_y_I = -np.sum(np.diag(px_y_I) * np.log(np.diag(px_y_I)))
       
        max_y_I = max(py_marginal_I).item()



        results.append([WD_px_py_I,entropy_y_x_I,entropy_x_y_I,max_y_I ])
        #idx = len(results)
        print(f"Finished processing row {idx+1}")
    

    output_df1 = pd.DataFrame(results, columns=["WD_px_py_I","pseduo_entropy_y_x_I","pseduo_entropy_x_y_I","max_y_I"])
    # output_df1.to_csv('matrix1.csv', index=False, header=True)
    output_df2 = pd.DataFrame(matrices, columns=["sim_X", "sim_Y"])
    output_df2.to_csv('.matrix.tmp.csv', index=False, header=True)
    
    
    df = output_df2
    df = pd.read_csv(".matrix.tmp.csv")


    mean_dfs = {}
    for N in [20]:
        # Used to record the (number of rows × number of columns) result matrix generated each cycle.
        results_list = []

        # The order of the column names in the table (the order in which the calculation results are appended below must be consistent with this order)
        columns = [
            "WD_px_py_I","pseduo_entropy_y_x_I","pseduo_entropy_x_y_I","max_y_I"
        ]

        # Used to store used keep_indices to prevent duplication.
        used_indices = set()
        np.random.seed(333)
        for i in range(N):
            # Generate unique keep_indices
            while True:
                keep_indices = [
                    random.randint(0, 4),
                    random.randint(5, 9),
                    random.randint(10, 14),
                    random.randint(15, 19),
                    random.randint(20, 24),
                    random.randint(25, 29),
                    random.randint(30, 34),
                    random.randint(35, 39),
                    random.randint(40, 44),
                    random.randint(45, 49),
                ]
                keep_indices_tuple = tuple(keep_indices)

                if keep_indices_tuple not in used_indices:
                    used_indices.add(keep_indices_tuple)
                    break
                else:
                    continue

            # The results of this iteration are first placed in a temporary list.
            current_iteration_results = []

            # Internal loop: Calculate each row of df.
            for row_idx, row in df.iterrows():
                simX_0 = parse_matrix(row[0])  # derive simX
                simY_0 = parse_matrix(row[1])  # derive simY

                # Slice according to the randomly selected keep_indices
                simX = simX_0[keep_indices, :][:, keep_indices]
                simY = simY_0[keep_indices, :][:, keep_indices]

                
                px = row_normalize(simX)
                py = row_normalize(simY)

    
                n = px.shape[0]

                
                pi_uniform = np.full((1, n), 1/n)
                py_marginal_I = (pi_uniform @ py).T
                px_y_I = py @ px
                px_marginal_I = (pi_uniform @ py @ py @ px).T
                py_x_I = (px_y_I * py_marginal_I) / px_marginal_I.T
                pxy_I  = px_y_I * py_marginal_I

                # 转换维度（让行= x，列= y）
                px_y_I = px_y_I.T
                py_x_I = py_x_I.T
                pxy_I = pxy_I.T

                

                WD_px_py_I = wasserstein_distance(px_marginal_I.flatten(), py_marginal_I.flatten())

                
                
                entropy_y_x_I = -np.sum(np.diag(py_x_I) * np.log(np.diag(py_x_I)))
                
                
                entropy_x_y_I = -np.sum(np.diag(px_y_I) * np.log(np.diag(px_y_I)))
                
                max_y_I = max(py_marginal_I).item()

                
                current_iteration_results.append([
                    WD_px_py_I,entropy_y_x_I,entropy_x_y_I,max_y_I
                ])

            
            iteration_df = pd.DataFrame(current_iteration_results, columns=columns)
            results_list.append(iteration_df)

            print(f"===== Finished iteration {i+1}/{N}, keep_indices={keep_indices_tuple} =====")

      
        sum_df = results_list[0].copy()
        for j in range(1, N):
            sum_df += results_list[j]

        mean_df = sum_df / N
        mean_dfs[N] = mean_df

    return mean_dfs, output_df2


#evaluation

def brier_score(y_true, y_prob):
    """compute Brier Score"""
    return np.mean((y_prob - y_true) ** 2)



def compute_prr(y_true, uncertainty):
    """
    compute Prediction Rejection Ratio (PRR) = AUCPR_unc / AUCPR_oracle
      - AUCPR_unc = average_precision_score(y_true, 1 - uncertainty)
      - AUCPR_oracle = average_precision_score(y_true, y_true)  
    """
    
    aucpr_unc = average_precision_score(y_true, 1 - uncertainty)
    
  
    aucpr_oracle = average_precision_score(y_true, y_true)  
    

    if aucpr_oracle == 0:
        return np.nan
    else:
        return aucpr_unc / aucpr_oracle
    

def compute_metrics(y_true, x):
    # 1) AUROC
    auroc = roc_auc_score(y_true, x)
    # 2) PRR
    prr = compute_prr(y_true, -x)
    correctness = y_true

   
    iso_model = IsotonicRegression(out_of_bounds='clip')
    x_isotonic = iso_model.fit_transform(x, correctness)
    brier_isotonic = brier_score(correctness, x_isotonic)


    return (
        auroc,
        prr,
        brier_isotonic,
        
    )

def calculate_overall_stats(results_df, correctness_df):
    
    correctness = correctness_df["Correctness"].values

    n_bootstraps = 100
    results_list = []

    columns = results_df.columns

    for column in columns:
        
        x = -results_df[column].values
        
        # 初始化存储列表
        auroc_vals = []
        prr_vals = []
        brier_isotonic_vals = []
        
     
        np.random.seed(42)

        for _ in range(n_bootstraps):
        
            indices = np.random.choice(len(x), size=len(x), replace=True)
         
            correctness_boot = correctness[indices]
            x_boot = x[indices]
  
            (auroc_b,
            prr_b,
            brier_isotonic_b,
            ) = compute_metrics(correctness_boot, x_boot)
            
            auroc_vals.append(auroc_b)
            prr_vals.append(prr_b)
           
            brier_isotonic_vals.append(brier_isotonic_b)
            
        
        auroc_mean, auroc_std = np.mean(auroc_vals), np.std(auroc_vals)
        prr_mean, prr_std = np.mean(prr_vals), np.std(prr_vals)
        brier_isotonic_mean, brier_isotonic_std = np.mean(brier_isotonic_vals), np.std(brier_isotonic_vals)
        
   
        results_list.append({
            "Uncertainty Measure": column,
            "AUROC": f"{auroc_mean:.3f} ± {auroc_std:.3f}",
            "PRR": f"{prr_mean:.3f} ± {prr_std:.3f}",
            "Brier_isotonic": f"{brier_isotonic_mean:.3f} ± {brier_isotonic_std:.3f}",
        })

    results_df_final = pd.DataFrame(results_list)
    return results_df_final