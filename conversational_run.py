import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def generate_response(prompt, model, tokenizer, max_length=200):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        output = model.generate(input_ids, attention_mask=attention_mask, max_length=max_length, num_return_sequences=1,  pad_token_id=tokenizer.eos_token_id)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def main():
    model_path = "custom_gpt2_model"
    tokenizer = GPT2Tokenizer.from_pretrained(model_path, pad_token_id=model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()

    print("Chatbot: Hello! I'm your chatbot. Let's have a conversation. (Type 'exit' to end the conversation)")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye! Have a great day!")
            break

        # Generate response from the model based on user input
        response = generate_response(user_input, model, tokenizer)
        print("Chatbot:", response)

if __name__ == "__main__":
    main()
