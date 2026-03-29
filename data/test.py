from datasets import load_dataset
scifact = load_dataset("scifact","claims")
print(scifact["train"][0])