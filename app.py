# app.py
import streamlit as st
from mismatch_checker import classify_mismatch_with_steps

st.set_page_config(page_title="Multilingual Cultural Context Detector - Stepwise", layout="centered")
st.title("Multilingual Cultural Context Detector")

st.markdown("""
This demo runs the NLP techniques **one-by-one** and shows outputs for each step.
Useful for demos, reports, and learning.
""")

src_text = st.text_area("Enter Original Text (English)", height=120)
tgt_text = st.text_area("Enter Translated Text (Hindi / Kannada)", height=120)
tgt_lang = st.selectbox("Target Language", ["Hindi", "Kannada"])  
threshold = st.slider("Similarity threshold (lower = more tolerant)", min_value=0.30, max_value=0.95, value=0.65, step=0.01)

if st.button("Run Step-by-Step Analysis"):
    if not src_text.strip() or not tgt_text.strip():
        st.error("Please enter both source and translated text.")
    else:
        # Run the pipeline
        result = classify_mismatch_with_steps(src_text, tgt_text, tgt_lang, threshold)

        st.header("Step-by-step Outputs")
        for idx, step in enumerate(result["steps"], start=1):
            st.subheader(f"Step {idx}: {step['technique']}")
            # Pretty display of the output dict
            st.json(step["output"])

        st.header("Final Summary")
        final = result["summary"]
        st.write(f"**Flagged as mismatch:** {'Yes' if final['flag'] else 'No'}")
        st.write(f"**Mismatch Type:** {final['mismatch_type']}")
        st.write(f"**Similarity Score:** {final['similarity']}")
        if final["suggestion"]:
            st.success(f"Suggested translation (expected in target): {final['suggestion']}")
        if final["src_highlight"]:
            st.info(f"Problematic phrase(s) in source: {', '.join(final['src_highlight'])}")
        if final["tgt_expected_highlight"]:
            st.info(f"Expected phrase(s) in target: {', '.join(final['tgt_expected_highlight'])}")
