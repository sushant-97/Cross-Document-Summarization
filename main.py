import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BartForConditionalGeneration, BartTokenizer, AdamW

# Custom dataset class
class SummarizationDataset(Dataset):
    def __init__(self, source_texts, target_texts, tokenizer, max_source_length=1024, max_target_length=150):
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        source_text = self.source_texts[idx]
        target_text = self.target_texts[idx]

        # Tokenize source and target texts
        source_inputs = self.tokenizer(source_text, max_length=self.max_source_length, return_tensors="pt", truncation=True, padding=True)
        target_inputs = self.tokenizer(target_text, max_length=self.max_target_length, return_tensors="pt", truncation=True, padding=True)

        return {
            "input_ids": source_inputs["input_ids"].squeeze(),
            "attention_mask": source_inputs["attention_mask"].squeeze(),
            "labels": target_inputs["input_ids"].squeeze(),
        }

# Function to load and tokenize data
def load_data():
    # Load your own dataset or use a pre-existing one
    # Replace this with your data loading logic
    # For simplicity, we'll use a dummy dataset
    source_texts = ["This is the first document.", "Another example document."]
    target_texts = ["Summary of the first document.", "Summary of the second document."]
    return source_texts, target_texts

# Fine-tuning function
def fine_tune_bart():
    # Load BART model and tokenizer
    model_name = "facebook/bart-large-cnn"
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)

    # Load training data
    source_texts, target_texts = load_data()

    # Create dataset and dataloader
    dataset = SummarizationDataset(source_texts, target_texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=1e-5)
    num_epochs = 3

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for batch in dataloader:
            inputs = {key: batch[key].to(model.device) for key in batch}
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Save the fine-tuned model at the end of each epoch
        model.save_pretrained(f"./bart_finetuned_model_epoch_{epoch + 1}")

if __name__ == "__main__":
    fine_tune_bart()
