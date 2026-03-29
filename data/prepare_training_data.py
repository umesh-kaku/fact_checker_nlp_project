import json
from datasets import load_dataset

print("Loading datasets...")

fever = load_dataset("fever", "v1.0")
fever_subset = fever["train"].shuffle(seed=42).select(range(30000))
# scifact = load_dataset("scifact", "claims")
# scifact_corpus = load_dataset("scifact", "corpus")
liar = load_dataset("liar")
liar_subset = liar["train"].shuffle(seed=42).select(range(9000))

print("FEVER subset size:", len(fever_subset))
print("LIAR subset size:", len(liar_subset))
# print("SciFact size:", len(scifact["train"]))

training_data = []

#processing the FEVER dataset
print("Processing FEVER dataset...")
for row in fever_subset:
    claim = row["claim"]
    label = row["label"]
    label_map = {
        "SUPPORTS": "SUPPORT",
        "REFUTES": "REFUTE",
        "NOT ENOUGH INFO": "NEUTRAL"
    }
    label = label_map.get(label, "NEUTRAL")
    training_data.append({
        "claim": claim, 
        "evidence": "", 
        "label": label
    })

#processing the LIAR dataset
print("Processing LIAR dataset...")
for row in liar_subset:
    claim = row["statement"]
    label = row["label"]
    label_map = {
        5: "SUPPORT",
        4: "SUPPORT",
        3: "NEUTRAL",
        2: "REFUTE",
        1: "REFUTE",
        0: "REFUTE"
    }
    label = label_map.get(label, "NEUTRAL")
    training_data.append({
        "claim": claim,
        "evidence": "",
        "label": label
    })

print("Saving unified training dataset to JSON file...")
with open("data/training_data.json", "w") as f:
    json.dump(training_data, f, indent = 2)

#processing the SciFact dataset
#SciFact dataset do not contain the labels, hence it cannot go into supervised training under BERT model
# print("Creating corpus dictionary for SciFact...")
# corpus_dict = {}
# for row in scifact_corpus["train"]:
#     corpus_dict[row["doc_id"]] = row["abstract"]

# print("Processing SciFact dataset...")
# for row in scifact["train"]:
#     claim = row["claim"]
#     label = row["evidence_label"]
#     label_map = {
#         "SUPPORT": "SUPPORT",
#         "CONTRADICT": "REFUTE",
#         "NEUTRAL": "NEUTRAL"
#     }
#     label = label_map.get(label, "NEUTRAL")
#     evidence_text = ""
#     evidence_dict = row["evidence_sentences"]
#     evidence_list = []
#     for doc_id, sentence_ids in evidence_dict.items():
#         doc_id = int(doc_id)
#         if doc_id in corpus_dict:
#             abstract_sentences = corpus_dict[doc_id]
#             for s in sentence_ids:
#                 if s < len(abstract_sentences):
#                     evidence_list.append(abstract_sentences[s])
#     evidence_text = " ".join(evidence_list)
#     training_data.append({
#         "claim": claim,
#         "evidence": evidence_text,
#         "label": label
#     })

print("Training dataset created")
print("Total training samples:", len(training_data))