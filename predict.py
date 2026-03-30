import torch
from transformers import BertTokenizer, BertForSequenceClassification
from evidence_retriever import retrieve_evidence

# loading the trained model
MODEL_PATH = "models/fact_checker_model_1"

print("Loading model and tokenizer...")
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

# for inference
model.eval()

# label mapping
label_map = {
    0: "SUPPORT",
    1: "REFUTE",
    2: "NEUTRAL"
}

# validity score mapping
def get_validity_score(label, confidence):
    if label == "SUPPORT":
        return int(70 + confidence * 30)
    elif label == "REFUTE":
        return int(30 - confidence * 30)
    else:
        return int(30 + confidence * 40)
    
# prediction function
def predict_claim_with_evidence(claim):
    evidences = retrieve_evidence(claim, top_k = 1)
    evidence_text = evidences[0]["text"]
    combined_input = claim + " " + evidence_text
    inputs = tokenizer(
        combined_input,
        return_tensors = "pt",
        truncation = True,
        padding = True,
        max_length = 256
    )

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim = 1)
    predicted_class_id = torch.argmax(probabilities, dim = 1).item()
    confidence = probabilities[0][predicted_class_id].item()
    label = label_map[predicted_class_id]
    score = get_validity_score(label, confidence)
    return label, confidence, score, evidence_text
    
# interactive loop
print("welcome to fact checker!")
print("Type 'exit' to quit.")

while True:
    claim = input("Enter a claim to check: ")
    if claim == "exit":
        break
    label, confidence, score, evidence = predict_claim_with_evidence(claim)
    print("Results:")
    print("Claim:", claim)
    print("Predicted Label:", label)
    print("Confidence:", confidence)
    print("Validity Score:", score, "/100")
    print("Evidence retrieved:")
    print(evidence[:500], "...")
    if label == "SUPPORT":
        print("Interpretation: LIKELY TRUE/ Healthly")
    elif label == "REFUTE":
        print("Interpretation: LIKELY FALSE/ Problematic")
    else:
        print("Interpretation: NEUTRAL/ Cannot be certain.")