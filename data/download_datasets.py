from datasets import load_dataset

#fever dataset
print("Loading FEVER dataset...")
fever = load_dataset("fever", "v1.0")
print("FEVER dataset loaded successfully.")
print("Fever dataset train size:", len(fever['train']))
print(fever)

#scifact dataset
print("Loading SciFact dataset...")
scifact = load_dataset("scifact", "claims")
print("SciFact dataset loaded successfully.")
print("SciFact dataset train size:", len(scifact['train']))
print(scifact)

#LIAR dataset
print("Loading LIAR dataset...")
liar = load_dataset("liar")
print("LIAR dataset loaded successfully.")
print("LIAR dataset train size:", len(liar['train']))
print(liar)