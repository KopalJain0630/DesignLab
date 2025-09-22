import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

@st.cache_resource
def load_model_and_tokenizer():
    model_name = "microsoft/phi-3.5-mini-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return model, tokenizer

def generate_answer(model, tokenizer, question, max_new_tokens=100):
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    answer_only = answer.split("Answer:")[-1].strip()
    return answer_only

def compute_perplexity(model, tokenizer, text):
    encodings = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = encodings.input_ids

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    perplexity = np.exp(loss.item())
    return perplexity

st.title("Question Answering & Perplexity with Phi-3.5")

question = st.text_area("Enter your question:", height=150)

if st.button("Get Answer & Perplexity"):
    if question.strip() == "":
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("Loading model and generating answer..."):
            model, tokenizer = load_model_and_tokenizer()
            answer = generate_answer(model, tokenizer, question)
            perplexity = compute_perplexity(model, tokenizer, answer)

        st.subheader("Answer:")
        st.write(answer)
        st.subheader("Perplexity of the answer:")
        st.write(f"**{perplexity:.2f}**")
