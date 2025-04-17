import sentencepiece as spm
import os

def train_tokenizer(input_file, model_prefix='tokenizer', vocab_size=10000, model_type='bpe'):

    """
    Trains a SentencePiece tokenizer on the specified input file.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Training data file not found: {input_file}")

    spm.SentencePieceTrainer.train(
        f"--input={input_file} "
        f"--model_prefix={model_prefix} "
        f"--vocab_size={vocab_size} "
        f"--model_type={model_type} "
        "--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 "
        "--minloglevel=2"
    )
    print(f"✅ Tokenizer trained and saved as {model_prefix}.model")

def load_tokenizer(model_path='tokenizer.model'):
    """
    Loads a trained SentencePiece tokenizer from file.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Tokenizer model file not found: {model_path}")

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(model_path)
    print(f"✅ Tokenizer loaded from {model_path}")
    return tokenizer

