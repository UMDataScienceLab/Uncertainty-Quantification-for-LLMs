import openai
import pandas as pd


# Iterate through the questions
def get_correctness(data):
    completed_data = pd.DataFrame(columns=[
        "Question", "Answer", "Correctness"
    ])

    for index, row in data.iterrows():
        
        question = row["question"]
        standard_answer = row["value"]

        # Get the answer and verbal confidence
        prompt = f"{question} Answer concisely and return only the name.\n And use a percentage to tell me your confidence in your answer."
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        answer = response["choices"][0]["message"]["content"]


        # Check correctness
        correctness_prompt = (
            f"Are the following two answers to my question Q semantically equivalent?\n"
            f"Q: {question}\nA1: {standard_answer}\nA2: {answer}\n"
            "Please answer with a single word, either Yes or No."
        )
        correctness_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": correctness_prompt}
            ],
            temperature=0
        )

        # Extract and validate the response
        correctness_text = correctness_response["choices"][0]["message"]["content"].strip().lower()
        print(correctness_text, answer, standard_answer)
        correctness = 1 if correctness_text == "yes" else 0 if correctness_text == "no" else None


        # Append to completed data using pd.concat
        new_row = pd.DataFrame([{
            "Question": question,
            "Answer": answer,
            "Correctness": correctness
        }])

        completed_data = pd.concat([completed_data, new_row], ignore_index=True)

    return completed_data
