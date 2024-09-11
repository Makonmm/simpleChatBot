import threading
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


model_name = "facebook/blenderbot-400M-distill"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


chat_history = []

print("Welcome, do you wanna talk? :)")


def chatting():
    while True:
        input_text = input("You: ")

        if input_text.lower() in ["exit", "quit"]:
            print("Ending...")
            break

        chat_history.append(f"You: {input_text}")

        history_string = "\n".join(chat_history)

        inputs = tokenizer.encode_plus(history_string, return_tensors="pt")

        outputs = model.generate(**inputs)

        response = tokenizer.decode(
            outputs[0], skip_special_tokens=True).strip()

        print(f"ChatBot: {response}")
        chat_history.append(f"ChatBot: {response}")


chatting()
