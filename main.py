# === main.py ===
import os
import torch
from torch.utils.data import DataLoader
from tokenizer_utils import train_tokenizer, load_tokenizer
from dataset import TextDataset, collate_fn
from models import RNNLanguageModel, LSTMLanguageModel, TransformerLanguageModel
from training_utils import run_pipeline, compute_bleu, compute_perplexity, generate_from_prompt
from torch.nn import CrossEntropyLoss
from tabulate import tabulate

if __name__ == "__main__":
    train_path = "data/train.jsonl"
    test_path = "data/test.jsonl"
    model_prefix = "tokenizer"
    tokenizer_model = f"{model_prefix}.model"

    # Step 1: Train tokenizer if needed
    if not os.path.exists(tokenizer_model):
        train_tokenizer(train_path, model_prefix)

    # Step 2: Load tokenizer
    tokenizer = load_tokenizer(tokenizer_model)

    # Step 3: Load datasets
    train_data = TextDataset(train_path, tokenizer)
    test_data = TextDataset(test_path, tokenizer)

    # Step 4: Model prompts
    prompt = "Which do you prefer? Dogs or cats?"
    custom_prompt = "Would you call yourself brave?"
    metrics_table = []

    # Step 5: Train and evaluate RNN
    rnn_model = run_pipeline(RNNLanguageModel, "RNN", tokenizer, train_data, test_data)
    rnn_bleu = compute_bleu(rnn_model, tokenizer, [prompt], ["I like Dogs better"])
    rnn_ppl = compute_perplexity(rnn_model, DataLoader(test_data, batch_size=128, collate_fn=collate_fn), CrossEntropyLoss()).item()
    metrics_table.append(["RNN", f"{rnn_ppl:.2f}", f"{rnn_bleu:.4f}"])

    # Step 6: Train and evaluate LSTM
    lstm_model = run_pipeline(LSTMLanguageModel, "LSTM", tokenizer, train_data, test_data)
    lstm_bleu = compute_bleu(lstm_model, tokenizer, [prompt], ["I like Dogs better"])
    lstm_ppl = compute_perplexity(lstm_model, DataLoader(test_data, batch_size=128, collate_fn=collate_fn), CrossEntropyLoss()).item()
    metrics_table.append(["LSTM", f"{lstm_ppl:.2f}", f"{lstm_bleu:.4f}"])

    # Step 7: Train and evaluate Transformer
    transformer_train_data = TextDataset(train_path, tokenizer, max_length=512)
    transformer_test_data = TextDataset(test_path, tokenizer, max_length=512)
    transformer_model = run_pipeline(TransformerLanguageModel, "Transformer", tokenizer, train_data, test_data)
    trans_bleu = compute_bleu(transformer_model, tokenizer, [prompt], ["I like Dogs better"])
    trans_ppl = compute_perplexity(transformer_model, DataLoader(test_data, batch_size=128, collate_fn=collate_fn), CrossEntropyLoss()).item()
    metrics_table.append(["Transformer", f"{trans_ppl:.2f}", f"{trans_bleu:.4f}"])

    # Step 8: Save generations
    with open("generations.txt", "w") as gen_file:
        for name, model in zip(["RNN", "LSTM", "Transformer"], [rnn_model, lstm_model, transformer_model]):
            gen_file.write(f"\n--- {name} Output ---\n")
            gen_file.write(generate_from_prompt(model, tokenizer, prompt) + "\n")
            gen_file.write("Custom Prompt - {0}:\n".format(name) + generate_from_prompt(model, tokenizer, custom_prompt) + "\n")

    # Step 9: Save metrics
    with open("evaluation_metrics.txt", "w") as eval_file:
        eval_file.write(tabulate(metrics_table, headers=["Model", "Perplexity", "BLEU Score"], tablefmt="grid"))

    print("\n=== Evaluation Metrics ===")
    print(tabulate(metrics_table, headers=["Model", "Perplexity", "BLEU Score"], tablefmt="grid"))

