# from datasets import load_dataset
# scifact = load_dataset("scifact","claims")
# print(scifact["train"][0])

try:
    import huggingface_hub
    print(f"huggingface_hub is installed. Version: {huggingface_hub.__version__}")
except ImportError:
    print("huggingface_hub is not installed.")
except Exception as e:
    print(f"An error occurred: {e}")
