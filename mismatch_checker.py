# mismatch_checker.py
from sentence_transformers import SentenceTransformer, util
from idiom_dict import idiom_dict, entity_list
import re
import numpy as np

# Load model once
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
model = SentenceTransformer(MODEL_NAME)

def simple_tokenize(text):
    # Very lightweight tokenizer for display/demo purposes
    # keep punctuation so idiom matching still works (we lowercase for matching)
    tokens = re.findall(r"\w+['’]?\w+|\w+|[^\s\w]", text, flags=re.UNICODE)
    return tokens

def embed_and_sample(text):
    emb = model.encode(text, convert_to_numpy=True)
    # Return embedding shape and first 6 values for demonstration
    sample = emb.flatten()[:6].tolist()
    norm = float(np.linalg.norm(emb))
    return {"shape": emb.shape, "sample_values": sample, "norm": round(norm, 4)}

def find_idiom_match(src_text, tgt_text, tgt_lang):
    matches = []
    for idiom in idiom_dict:
        if idiom.lower() in src_text.lower() and tgt_lang in idiom_dict[idiom]:
            expected = idiom_dict[idiom][tgt_lang]
            found_in_tgt = expected.lower() in tgt_text.lower()
            matches.append({
                "idiom": idiom,
                "expected_translation": expected,
                "found_in_target": found_in_tgt
            })
    return matches

def find_entity_match(src_text, tgt_text, tgt_lang):
    matches = []
    for ent in entity_list:
        if ent.lower() in src_text.lower() and tgt_lang in entity_list[ent]:
            expected = entity_list[ent][tgt_lang]
            found_in_tgt = expected.lower() in tgt_text.lower()
            matches.append({
                "entity": ent,
                "expected_translation": expected,
                "found_in_target": found_in_tgt
            })
    return matches

def classify_mismatch_with_steps(src_text, tgt_text, tgt_lang='hi', threshold=0.65):
    """
    Runs each technique step-by-step and returns a dict containing:
    - a 'steps' list with technique name and its output
    - final summary with flag, mismatch type, suggestion, similarity
    """
    steps = []

    # ---------- Step 1: Preprocessing ----------
    technique = "Preprocessing / Normalization"
    # Minimal normalization for demo (strip extra whitespace)
    src_norm = src_text.strip()
    tgt_norm = tgt_text.strip()
    steps.append({"technique": technique, "output": {"src_norm": src_norm, "tgt_norm": tgt_norm}})

    # ---------- Step 2: Tokenization ----------
    technique = "Tokenization"
    src_tokens = simple_tokenize(src_norm)
    tgt_tokens = simple_tokenize(tgt_norm)
    steps.append({"technique": technique, "output": {"src_tokens": src_tokens, "tgt_tokens": tgt_tokens}})

    # ---------- Step 3: Sentence Embeddings ----------
    technique = f"Sentence Embedding (model={MODEL_NAME})"
    src_emb_info = embed_and_sample([src_norm])  # list input supported
    tgt_emb_info = embed_and_sample([tgt_norm])
    steps.append({"technique": technique, "output": {"src_embedding": src_emb_info, "tgt_embedding": tgt_emb_info}})

    # ---------- Step 4: Semantic Similarity ----------
    technique = "Semantic Similarity (Cosine)"
    src_emb = model.encode(src_norm, convert_to_numpy=True)
    tgt_emb = model.encode(tgt_norm, convert_to_numpy=True)
    sim_score = float(util.cos_sim(src_emb, tgt_emb).item())
    steps.append({"technique": technique, "output": {"similarity_score": round(sim_score, 4), "threshold": threshold}})

    # ---------- Step 5: Threshold-based Decision ----------
    technique = "Threshold-based Classification"
    is_below_thresh = sim_score < threshold
    decision = "Potential Mismatch" if is_below_thresh else "Likely Match"
    steps.append({"technique": technique, "output": {"decision": decision, "below_threshold": bool(is_below_thresh)}})

    # ---------- Step 6: Dictionary-based Idiom Detection ----------
    technique = "Dictionary-based Idiom Detection"
    idiom_checks = find_idiom_match(src_norm, tgt_norm, tgt_lang)
    #steps.append({"technique": technique, "output": {"idiom_checks": idiom_checks}})

    # ---------- Step 7: Dictionary-based Entity Detection ----------
    technique = "Dictionary-based Entity Detection"
    entity_checks = find_entity_match(src_norm, tgt_norm, tgt_lang)
    #git asteps.append({"technique": technique, "output": {"entity_checks": entity_checks}})

    # ---------- Step 8: Phrase Highlighting (rule-based) ----------
    technique = "Phrase Highlighting"
    src_highlight = []
    tgt_expected_highlight = []
    suggestion = None
    mismatch_type = None

    # If idiom found but not in target => IdiomLost
    for ic in idiom_checks:
        if not ic["found_in_target"]:
            src_highlight.append(ic["idiom"])
            tgt_expected_highlight.append(ic["expected_translation"])
            suggestion = ic["expected_translation"]
            mismatch_type = "IdiomLost"
    # If entity found but not in target => EntityMismatch (only if not already idiom)
    if not mismatch_type:
        for ec in entity_checks:
            if not ec["found_in_target"]:
                src_highlight.append(ec["entity"])
                tgt_expected_highlight.append(ec["expected_translation"])
                suggestion = ec["expected_translation"]
                mismatch_type = "EntityMismatch"

    # If no dictionary-based issues but similarity low => Cultural/Semantic mismatch
    # if not mismatch_type and is_below_thresh:
     #   mismatch_type = "Possible Cultural/Semantic Mismatch (e.g., sarcasm or untranslated nuance)"

        # If idiom is correctly translated → don't flag mismatch
    has_correct_idiom = any(ic["found_in_target"] for ic in idiom_checks)

    if has_correct_idiom:
        mismatch_type = "ExactMatch"
        is_below_thresh = False  # ignore similarity


    # If nothing wrong
    if not mismatch_type:
        mismatch_type = "ExactMatch"

    steps.append({"technique": technique, "output": {
        "src_highlight": src_highlight,
        "tgt_expected_highlight": tgt_expected_highlight,
        "suggestion": suggestion,
        "mismatch_type": mismatch_type
    }})

    # ---------- Final summary ----------
    summary = {
        "flag": mismatch_type != "ExactMatch",
        "similarity": round(sim_score, 4),
        "mismatch_type": mismatch_type,
        "suggestion": suggestion,
        "src_highlight": src_highlight,
        "tgt_expected_highlight": tgt_expected_highlight
    }

    return {"steps": steps, "summary": summary}
