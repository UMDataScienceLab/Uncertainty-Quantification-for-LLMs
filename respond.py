import openai
import pandas as pd
import os

# Define a function that calls the ChatGPT API and retrieves answers
def ask_chatgpt(question, num_responses=5):
    responses = []
    for _ in range(num_responses):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"{question} Answer concisely and return only the name."}],
            temperature=0.7  # Set temperature
        )
        responses.append(response['choices'][0]['message']['content'])
    return responses

def respond(df):
    new_df = df.copy()
    # Iterate over each row of the original DataFrame
    for i in range(df.shape[0]):
        # Retrieve non-null values in the current row
        row_values = new_df.iloc[i].dropna().astype(str)

        print(f"Processing row {i}...")  # Display the current row being processed
        row_modified = False  # Flag to indicate whether the row was modified

        for j in range(df.shape[1]):
            question = df.iloc[i, j]
            if isinstance(question, str):  # Ensure cell content is a string
                try:
                    # Call the API and store the results
                    responses = ask_chatgpt(question, num_responses=5)
                    new_df.iloc[i, j] = str([question] + responses)  # Concatenate question and answers
                    row_modified = True  # Mark the row as modified
                except Exception as e:
                    print(f"Error occurred at row {i}, column {j}: {e}")
                    raise

    return new_df
