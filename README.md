# Foundational AI Project 2 - Sequential Language Models

This project implements and compares three sequential language models — **RNN**, **LSTM**, and **Transformer** — using PyTorch for text generation tasks. The models are trained on subword tokenized data using a **BPE tokenizer (SentencePiece)** and evaluated on **perplexity** and **BLEU score**.

## 📁 Project Structure

```
.
├── dataset.py              # Dataset and collate_fn
├── models.py               # Model definitions (RNN, LSTM, Transformer)
├── tokenizer_utils.py      # Tokenizer training/loading (SentencePiece)
├── training_utils.py       # Training loop, evaluation, BLEU, loss plots
├── main.py                 # Full training + evaluation pipeline
├── data/
│   ├── train.jsonl         # Training dataset (prompt + completion)
│   └── test.jsonl          # Testing dataset
├── tokenizer.model         # Saved SentencePiece model (generated)
├── rnn_model.pt            # Trained RNN model weights (generated)
├── lstm_model.pt           # Trained LSTM model weights (generated)
├── transformer_model.pt    # Trained Transformer model weights (generated)
├── generations.txt         # Prompt-based outputs from each model (generated)
├── evaluation_metrics.txt  # Perplexity & BLEU score results (generated)
```

## 🚀 How to Run

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train models and evaluate**
   ```bash
   python main.py
   ```
   This will:
   - Train the tokenizer on `train.jsonl`
   - Train RNN, LSTM, and Transformer models
   - Generate text from prompts
   - Evaluate and save PPL + BLEU metrics

3. **View Outputs**
   - `generations.txt` → generated responses
   - `evaluation_metrics.txt` → PPL and BLEU for all models
   - `*_loss.png` → Training/validation loss curves

## 🧠 Prompts Used

- **"Which do you prefer? Dogs or cats?"**
- **"Would you call yourself brave?"**

You can change these in `main.py` to test your own prompts.

## 📊 Evaluation Metrics

- **Perplexity (PPL)** — measures how well the model predicts the next token
- **BLEU Score** — evaluates text similarity to the reference

## 🧪 Model Training Details

- Tokenizer: SentencePiece BPE, vocab size 10,000
- Embedding dim: 128
- Hidden dim: 256
- Optimizer: AdamW
- Loss: CrossEntropyLoss
- Epochs: 30 max, with early stopping
- Batch size: 128

## 📄 License
MIT License

---
Made for CSC 7700 Foundational AI - Project 2

