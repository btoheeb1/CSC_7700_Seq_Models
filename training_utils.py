import torch
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.nn as nn
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tabulate import tabulate
from dataset import collate_fn

def train(model, train_loader, val_loader, optimizer, criterion, epochs=30, patience=3):
    train_losses, val_losses = [], []
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return train_losses, val_losses

def plot_losses(train_losses, val_losses, title):
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{title.replace(' ', '_').lower()}_loss.png")
    plt.close()

def compute_perplexity(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            total_loss += loss.item()
    return torch.exp(torch.tensor(total_loss / len(data_loader)))

def compute_bleu(model, tokenizer, prompts, references):
    model.eval()
    bleu_scores = []
    smoothie = SmoothingFunction().method4
    for prompt, ref in zip(prompts, references):
        input_ids = torch.tensor([tokenizer.EncodeAsIds(prompt)])
        output_ids = model.sample(input_ids)
        decoded = tokenizer.DecodeIds(output_ids[0].tolist())
        ref_tokens = ref.split()
        gen_tokens = decoded.split()
        score = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smoothie)
        bleu_scores.append(score)
    return sum(bleu_scores) / len(bleu_scores)

def run_pipeline(model_class, model_name, tokenizer, train_data, val_data, emb_dim=128, hidden_dim=256, nhead=4, num_layers=2):
    if model_name == 'Transformer':
        model = model_class(len(tokenizer), emb_dim, nhead, num_layers)
    else:
        model = model_class(len(tokenizer), emb_dim, hidden_dim)

    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=128, collate_fn=collate_fn)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=0.001)

    train_losses, val_losses = train(model, train_loader, val_loader, optimizer, criterion)
    ppl = compute_perplexity(model, val_loader, criterion)
    print(f"{model_name} Perplexity: {ppl.item():.2f}")
    plot_losses(train_losses, val_losses, f"{model_name} Loss Curve")

    torch.save(model.state_dict(), f"{model_name.lower()}_model.pt")
    print(f"✅ Saved {model_name} model to {model_name.lower()}_model.pt")

    return model

def generate_from_prompt(model, tokenizer, prompt, temperature=1.0):
    input_ids = torch.tensor([tokenizer.EncodeAsIds(prompt)])
    output_ids = model.sample(input_ids, temperature=temperature)
    return tokenizer.DecodeIds(output_ids[0].tolist())

