import streamlit as st
from transformers import pipeline

# Load sentiment model
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

sentiment_pipeline = load_model()

# Page config
st.set_page_config(page_title="Sentiment Analysis", layout="centered")

st.title("📊 Sentiment Analysis (Dictionary Input)")
st.write("Enter text in **ID : Sentence** format")

# Text input
text_input = st.text_area(
    "Input",
    placeholder="1: I love this application\n2: The software is slow\n3: It works as expected",
    height=200
)

# Analyze button
if st.button("Analyze Sentiment"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        results = []

        lines = text_input.strip().split("\n")
        for line in lines:
            if ":" in line:
                key, sentence = line.split(":", 1)

                prediction = sentiment_pipeline(sentence.strip())[0]
                label = prediction["label"]
                score = prediction["score"]

                if score < 0.6:
                    sentiment = "Neutral"
                elif label == "POSITIVE":
                    sentiment = "Positive"
                else:
                    sentiment = "Negative"

                results.append({
                    "ID": key.strip(),
                    "Sentence": sentence.strip(),
                    "Sentiment": sentiment
                })

        if results:
            st.success("Analysis Complete ✅")
            st.table(results)
        else:
            st.error("Invalid format. Use ID : Sentence")

