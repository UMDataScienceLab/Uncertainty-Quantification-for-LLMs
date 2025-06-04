import pandas as pd
from paraphrase import paraphrase
from respond import respond
from correctness import get_correctness
import openai
import sys
import os
import time

# List of OpenAI API keys
API_KEYS = [
    "your_key1",
    "your_key2",
    "your_key3"
    # Add more keys as needed
]



# Global key index tracker
current_key_index = 0

def set_api_key(index):
    global current_key_index
    current_key_index = index
    openai.api_key = API_KEYS[index]

def try_with_api_keys(func, *args, **kwargs):
    global current_key_index
    num_keys = len(API_KEYS)
    for attempt in range(1000):
        set_api_key(current_key_index)
        try:
            return func(*args, **kwargs)
        except openai.OpenAIError as e:
            print(f"[Warning] OpenAIError with key {current_key_index}: {e}")
            current_key_index = (current_key_index + 1) % num_keys
            time.sleep(1)
    raise RuntimeError("All API keys failed.")

if len(sys.argv) < 2:
    print("Usage: python pipeline_agent.py <input_file.csv>")
    sys.exit(1)

input_path = sys.argv[1]


# make a directory for the results
base_name = os.path.splitext(os.path.basename(input_path))[0]
output_dir = base_name
os.makedirs(output_dir, exist_ok=True)

paraphrase_path = os.path.join(output_dir, 'paraphrase.csv')
responded_path = os.path.join(output_dir, 'responded.csv')
correctness_path = os.path.join(output_dir, 'correctness.csv')

df = pd.read_csv(input_path)
paraphrased = []
for i in range(len(df)):
    row_df = df.iloc[[i]]
    result = try_with_api_keys(paraphrase, row_df)
    paraphrased.append(result)
paraphrased = pd.concat(paraphrased)
paraphrased.index = df.index
paraphrased['value'] = df['value']
paraphrased.to_csv(paraphrase_path)

paraphrased = pd.read_csv(paraphrase_path, index_col=0)
responded = []
for i in range(len(paraphrased)):
    row_df = paraphrased.iloc[[i]]
    result = try_with_api_keys(respond, row_df)
    responded.append(result)
responded = pd.concat(responded)
responded.index = paraphrased.index
responded['question'] = paraphrased['original_question']
responded['value'] = paraphrased['value']
responded.to_csv(responded_path)

responded = pd.read_csv(responded_path, index_col=0)
correctnesses = []
for i in range(len(responded)):
    row_df = responded.iloc[[i]]
    result = try_with_api_keys(get_correctness, row_df)
    correctnesses.append(result)
correctnesses = pd.concat(correctnesses)
correctnesses.index = responded.index
correctnesses.to_csv(correctness_path)