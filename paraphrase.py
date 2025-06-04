import openai 
import pandas as pd

# Define function for paraphrasing a single sentence with GPT
def paraphrase_sentence_gpt(sentence, num_return_sequences=9, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that paraphrases sentences."},
            {"role": "user", "content": f"Provide {num_return_sequences} paraphrases for this sentence: {sentence}"}
        ],
        temperature=0.7
    )
    paraphrased_text = response['choices'][0]['message']['content']
    paraphrases = [p.strip() for p in paraphrased_text.split('\n') if p.strip()]
    return paraphrases[:num_return_sequences]  # Ensure 9 paraphrases are returned

def paraphrase(df):
    # Initialize list to store rows for the new table
    new_rows = []

    # Iterate over each row in the original table
    for index, row in df.iterrows():
        original_question = row['question']        # Get the question
        paraphrases = paraphrase_sentence_gpt(original_question)  # Call function to get 9 paraphrased questions

        # Ensure there are 9 paraphrased questions (pad with empty strings if necessary)
        if len(paraphrases) < 9:
            paraphrases += [""] * (9 - len(paraphrases))

        # Create a row: original question + 9 paraphrased questions
        new_row = [original_question] + paraphrases
        new_rows.append(new_row)

        # Print progress
        print(f"Processed {index + 1}/{len(df)}: {original_question}")

    # Create new DataFrame
    columns = ['original_question'] + [f'paraphrase_{i+1}' for i in range(9)]
    new_df = pd.DataFrame(new_rows, columns=columns)
    return new_df
