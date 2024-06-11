import streamlit as st
import PyPDF2
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Function to read PDF and extract text
def read_pdf(file_path):
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfFileReader(f)
        text = ''
        for page_num in range(reader.numPages):
            page = reader.getPage(page_num)
            text += page.extract_text()
    return text

# Function to generate questions using T5 model
def generate_questions(text):
    # Load T5 model
    model_name = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Optimize model for tokenization
    model = optimize_model(model, tokenizer)
    
    # Tokenize and generate questions
    inputs = tokenizer.encode("translate English to English: " + text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1, early_stopping=True)
    
    questions = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return questions

# Streamlit UI
def main():
    st.title("PDF Question Generator with T5 Model")
    st.write("Upload a PDF file to generate questions.")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        st.write("Generating questions...")

        if st.button("Generate Questions"):
            text = read_pdf(uploaded_file)
            questions = generate_questions(text)
            st.markdown("### Generated Questions:")
            st.write(questions)

if __name__ == "__main__":
    main()
