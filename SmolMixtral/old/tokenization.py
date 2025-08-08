import os
import sentencepiece as spm
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from dotenv import load_dotenv

# --- Configuration ---
TOKEN = os.getenv('HF_TOKEN')
DATASET_NAME = "HuggingFaceFW/fineweb"
DATASET_CONFIG = "sample-10BT"  # adjust if needed
TRAIN_SPLIT = "train"
TRAIN_TEXT_FILE = "train_texts.txt"
MODEL_PREFIX = "sp_tokenizer"
VOCAB_SIZE = 32000

# --- Step 1: Load the Dataset ---
print("Loading dataset...")
dataset = load_dataset(DATASET_NAME, name=DATASET_CONFIG, split=TRAIN_SPLIT, token=TOKEN)
print(f"Dataset loaded with {len(dataset)} samples.")

# --- Step 2: Write Texts to a File Using Batched Processing ---
def write_batch(batch, indices):
    # Use the first index to decide the file mode: if it's 0, write mode; else, append.
    mode = "w" if indices[0] == 0 else "a"
    with open(TRAIN_TEXT_FILE, mode, encoding="utf-8") as f:
        texts = [text.strip() for text in batch["text"] if text and text.strip()]
        if texts:
            f.write("\n".join(texts) + "\n")

# Remove the file if it already exists
if os.path.exists(TRAIN_TEXT_FILE):
    os.remove(TRAIN_TEXT_FILE)

print("Processing dataset in batches and writing texts to file...")
dataset.map(write_batch, batched=True, batch_size=1000, with_indices=True)
print(f"Finished writing texts to {TRAIN_TEXT_FILE}.")

# --- Step 3: Train the SentencePiece Tokenizer ---
print("Training SentencePiece tokenizer...")
spm.SentencePieceTrainer.train(
    input=TRAIN_TEXT_FILE,
    model_prefix=MODEL_PREFIX,
    vocab_size=VOCAB_SIZE,
    model_type="bpe",  # Using BPE as per your preference
    character_coverage=1.0,
    pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    input_sentence_size=25000000,  # Limit the number of sentences used
    shuffle_input_sentence=True,    # Shuffle sentences to get a representative sample
    user_defined_symbols=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
)
print("Tokenizer training completed.")

# --- Step 4: Load the Tokenizer Using Hugging Face ---
print("Loading trained tokenizer with Hugging Face interface...")
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=f"/content/{MODEL_PREFIX}.model",
    unk_token="[UNK]",
    pad_token="[PAD]",
    bos_token="[BOS]",
    eos_token="[EOS]"
)

# Test the tokenizer on a sample sentence
sample_text = "Hello, how are you doing today?"
encoded = tokenizer.encode(sample_text)
decoded = tokenizer.decode(encoded)
print("Sample text:", sample_text)
print("Encoded IDs:", encoded)
print("Decoded text:", decoded)
