import openai
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

openai.api_key = "your_openai_key"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def ask_chatgpt(question, num_responses=5, temperature=0.7):
    responses = []
    for _ in range(num_responses):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": f"{question} Answer concisely and return only the name."}],
                temperature=temperature
            )
            responses.append(response['choices'][0]['message']['content'].strip())
        except Exception as e:
            print(f"[ERROR] OpenAI API error: {e}")
            responses.append("ERROR")
    return responses

class EntailmentDeberta:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge-mnli")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v2-xlarge-mnli").to(DEVICE)

    def check_implication(self, text1, text2):
        inputs = self.tokenizer(text1, text2, return_tensors="pt", truncation=True).to(DEVICE)
        outputs = self.model(**inputs)
        logits = outputs.logits
        return torch.argmax(F.softmax(logits, dim=1)).item()

def get_semantic_ids(responses, model):
    def are_equivalent(t1, t2):
        imp1 = model.check_implication(t1, t2)
        imp2 = model.check_implication(t2, t1)
        return (0 not in [imp1, imp2]) and ([imp1, imp2] != [1, 1])

    semantic_ids = [-1] * len(responses)
    next_id = 0
    for i in range(len(responses)):
        if semantic_ids[i] == -1:
            semantic_ids[i] = next_id
            for j in range(i + 1, len(responses)):
                if semantic_ids[j] == -1 and are_equivalent(responses[i], responses[j]):
                    semantic_ids[j] = next_id
            next_id += 1
    return semantic_ids

def cluster_assignment_entropy(semantic_ids):
    n = len(semantic_ids)
    counts = np.bincount(semantic_ids)
    p = counts / n
    return -np.sum(p * np.log(p))

def compute_semantic_entropy(input_csv, output_csv, num_responses=5, temperature=0.7):
    df = pd.read_csv(input_csv)
    model = EntailmentDeberta()
    entropies = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        question = row['question']
        responses = ask_chatgpt(question, num_responses=num_responses, temperature=temperature)
        if all(r == "ERROR" for r in responses):
            entropy = np.nan
        else:
            semantic_ids = get_semantic_ids(responses, model)
            entropy = cluster_assignment_entropy(semantic_ids)
        entropies.append(entropy)

    df["semantic_entropy"] = entropies
    df.to_csv(output_csv, index=False)
    print(f"Saved results to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute semantic entropy using GPT and DeBERTa.")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file with columns: question,value")
    parser.add_argument("--output", type=str, required=True, help="Path to output CSV file")
    parser.add_argument("--num_responses", type=int, default=5, help="Number of GPT responses to sample per question")
    parser.add_argument("--temperature", type=float, default=0.7, help="GPT temperature setting")
    args = parser.parse_args()

    compute_semantic_entropy(args.input, args.output, args.num_responses, args.temperature)
