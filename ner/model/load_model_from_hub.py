from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import streamlit as st

REPO_ID = "HallowsYves/soundslike-ner"

@st.cache_resource(show_spinner=False)
def get_ner_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
    model = AutoModelForTokenClassification.from_pretrained(REPO_ID)
    return pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
