import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        self.samples = []
        dropped = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    prompt = entry.get("prompt", "").strip()
                    completion = entry.get("completion", "").strip()
                    full_text = (prompt + " " + completion).strip()
                    if not full_text:
                        continue
                    tokens = torch.tensor(tokenizer.EncodeAsIds(full_text))[:max_length]
                    if len(tokens) > 2:
                        self.samples.append(tokens)
                    else:
                        dropped += 1
                except Exception as e:
                    print(f"Skipping bad line: {line[:50]}... ({e})")
        print(f"✅ Loaded {len(self.samples)} samples from {file_path}, dropped {dropped} short or invalid entries.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq = self.samples[idx]
        return seq[:-1], seq[1:]

def collate_fn(batch):
    inputs, targets = zip(*batch)
    return pad_sequence(inputs, batch_first=True, padding_value=0), pad_sequence(targets, batch_first=True, padding_value=0)
