import gradio as gr
import torch
import torch.nn.functional as F
import os
import sys
import argparse

# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Transformer
from tokenizer import initialize_tokenizer
from config import ModelArgs

def topk_sampling(model, prompt, tokenizer, device, max_length=30, top_k=50, temperature=0.8, repetition_penalty=1.2):
    """Top-k sampling generation"""
    model.eval()
    
    # Tokenize the source text
    input_ids = tokenizer(prompt, add_special_tokens=True, max_length=max_length, padding='max_length', return_tensors="pt")['input_ids'].to(device)

    print("Input IDs: ", input_ids.shape)
    # Create an initial target sequence with just the BOS token
    target_ids = torch.tensor([[tokenizer.bos_token_id]], device=device)
    
    src_mask = torch.ones(input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype).to(device)
    src_mask = src_mask.masked_fill(input_ids == tokenizer.pad_token_id, 0)
    
    # First, encode the source sequence - do this only once
    with torch.no_grad():
        encoder_output = model.encoder(input_ids, src_mask)
    
    for i in range(max_length):
        with torch.no_grad():
            # Pass the encoder output and current target sequence to the decoder
            decoder_output = model.decoder(encoder_output, encoder_output, target_ids, src_mask, None)
            
            # Get logits from the final linear layer
            logits = model.linear_layer(model.norm(decoder_output))

            logits = logits[:, -1, :] / temperature
            
            probs = F.softmax(logits, dim=-1)
            
            # Top-k filtering
            top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
            
            # Sample from top-k
            next_token = torch.multinomial(top_k_probs, num_samples=1)
            xcol = torch.gather(top_k_indices, -1, next_token)
            
            target_ids = torch.cat([target_ids, xcol], dim=1)
            
            # Check if we generated an EOS token
            if xcol.item() == tokenizer.eos_token_id:
                break
    
    print("Target IDs: ", target_ids.shape)
    model.train()
    
    # Decode the generated sequence
    generated_text = tokenizer.decode(target_ids[0])
    return generated_text

def beam_search_corrected(
    model,
    prompt: str,
    tokenizer,
    device: str,
    block_size: int,
    beam_width: int = 5,
    max_length: int = 50,
    temperature: float = 1.0
):
    """Beam search generation"""
    model.eval()
    model = model.to(device)

    # Tokenize and prepare source input with padding and mask
    inputs = tokenizer(
        prompt,
        add_special_tokens=True,
        max_length=block_size,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    src_input_ids = inputs["input_ids"].to(device)
    
    src_mask = torch.ones(src_input_ids.shape[0], src_input_ids.shape[1], dtype=torch.long, device=device)
    src_mask = src_mask.masked_fill(src_input_ids == tokenizer.pad_token_id, 0)

    with torch.no_grad():
        # Encoder output is computed once using the properly masked source
        encoder_output = model.encoder(src_input_ids, src_mask)

    # Initialize beams: list of (sequence_tensor, log_probability_score)
    initial_sequence = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long, device=device)
    beams = [(initial_sequence, 0.0)]
    completed_sequences = []

    for step in range(max_length):
        if not beams:
            break

        all_next_candidates = []

        for current_seq_tensor, current_score in beams:
            # If sequence already ended with EOS, move to completed and skip expansion
            if current_seq_tensor[0, -1].item() == tokenizer.eos_token_id:
                completed_sequences.append((current_seq_tensor, current_score))
                continue

            # Prepare decoder input (current target sequence)
            decoder_input_ids = current_seq_tensor

            with torch.no_grad():
                decoder_out = model.decoder(
                    encoder_output,
                    encoder_output,
                    decoder_input_ids,
                    src_mask
                )
                # Get logits for the *last* token in the sequence
                logits_last_token = model.linear_layer(model.norm(decoder_out[:, -1, :]))
                logits_last_token = logits_last_token / temperature
                log_probs = F.log_softmax(logits_last_token, dim=-1)

            # Get top k next tokens for *this* beam
            top_k_log_probs, top_k_indices = torch.topk(log_probs.squeeze(0), beam_width, dim=-1)

            for i in range(beam_width):
                next_token_id = top_k_indices[i].item()
                next_token_log_prob = top_k_log_probs[i].item()

                new_seq_tensor = torch.cat(
                    [current_seq_tensor, torch.tensor([[next_token_id]], device=device)],
                    dim=1
                )
                new_score = current_score + next_token_log_prob
                all_next_candidates.append((new_seq_tensor, new_score))

        # Filter out completed sequences from the list of candidates for the next step
        new_beams = []
        temp_completed = []

        all_next_candidates.sort(key=lambda x: x[1], reverse=True)

        for cand_seq, cand_score in all_next_candidates:
            if cand_seq[0, -1].item() == tokenizer.eos_token_id:
                temp_completed.append((cand_seq, cand_score))
            elif len(new_beams) < beam_width:
                new_beams.append((cand_seq, cand_score))

        beams = new_beams
        completed_sequences.extend(temp_completed)

        # Early stopping
        if len(completed_sequences) >= beam_width:
            completed_sequences.sort(key=lambda x: x[1], reverse=True)
            if not beams or (beams and completed_sequences[beam_width-1][1] > beams[0][1]):
                break

    # If no sequences were completed, use the active beams
    if not completed_sequences and beams:
        completed_sequences.extend(beams)
    elif not completed_sequences and not beams:
        model.train()
        return "[Beam Search Error: No sequences generated]"

    # Sort all completed sequences by score (descending)
    completed_sequences.sort(key=lambda x: x[1], reverse=True)

    if not completed_sequences:
        model.train()
        return "[Beam Search Error: No completed sequences found]"

    best_sequence_tensor = completed_sequences[0][0].squeeze(0)
    model.train()
    generated_text = tokenizer.decode(best_sequence_tensor)
    
    return generated_text

class TranslationApp:
    def __init__(self, checkpoint_path=None):
        self.model = None
        self.tokenizer = None
        self.model_args = ModelArgs()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model on initialization if checkpoint is provided
        if checkpoint_path:
            print(f"Attempting to load checkpoint from: {checkpoint_path}")
            self.load_model(checkpoint_path)
    
    def load_model(self, checkpoint_path=None):
        """Load the trained model"""
        print("\n" + "="*50)
        print("MODEL LOADING PROCESS")
        print("="*50)
        
        try:
            print("\n[1/4] Initializing tokenizer...")
            self.tokenizer = initialize_tokenizer(self.model_args.hf_token)
            print("✓ Tokenizer initialized successfully")
            
            # Update vocab sizes
            self.model_args.src_vocab_size = len(self.tokenizer)
            self.model_args.tgt_vocab_size = len(self.tokenizer)
            print(f"✓ Vocab sizes updated - Source: {self.model_args.src_vocab_size}, Target: {self.model_args.tgt_vocab_size}")
            
            print("\n[2/4] Initializing model architecture...")
            # Initialize model on the correct device
            self.model = Transformer(
                src_vocab_size=self.model_args.src_vocab_size,
                tgt_vocab_size=self.model_args.tgt_vocab_size,
                use_liger=False  # Disable for inference
            ).to(self.device)
            print(f"✓ Model architecture initialized on {self.device}")
            
            print(f"\n[3/4] Loading checkpoint from: {checkpoint_path}")
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")
                
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            print("✓ Checkpoint loaded successfully")
            
            print("\n[4/4] Processing and loading model weights...")
            # Clean up the state dict by removing _ori_mode from keys if present
            state_dict = checkpoint['MODEL_STATE']
            cleaned_state_dict = {}
            
            for key, value in state_dict.items():
                # Remove _ori_mode from key if it exists
                new_key = key.replace('_orig_mod.', '')
                cleaned_state_dict[new_key] = value
                if new_key != key:
                    print(f"  - Renamed: {key} -> {new_key}")
            
            # Load the cleaned state dict
            self.model.load_state_dict(cleaned_state_dict)
            self.model.eval()
            print("✓ Model weights loaded successfully (with _ori_mode keys removed)")
            
            print("\n" + "="*50)
            print("MODEL LOADED SUCCESSFULLY!")
            print("="*50 + "\n")
            
            return "Model loaded successfully!"
            
        except Exception as e:
            print("\n" + "!"*50)
            print("ERROR LOADING MODEL:")
            print(f"Type: {type(e).__name__}")
            print(f"Error: {str(e)}")
            print("!"*50 + "\n")
            return f"Error loading model: {str(e)}"
    
    def translate(self, text, method="Top-K Sampling", max_length=100, temperature=0.8, top_k=50, beam_width=5):
        """Translate text using the selected method"""
        if self.model is None or self.tokenizer is None:
            return "Please load a model first!"
        
        if not text.strip():
            return "Please enter some text to translate."
        
        try:
            if method == "Top-K Sampling":
                result = topk_sampling(
                    model=self.model,
                    prompt=text,
                    tokenizer=self.tokenizer,
                    device=self.device,
                    max_length=max_length,
                    top_k=top_k,
                    temperature=temperature
                )
            else:  # Beam Search
                result = beam_search_corrected(
                    model=self.model,
                    prompt=text,
                    tokenizer=self.tokenizer,
                    device=self.device,
                    block_size=self.model_args.block_size,
                    beam_width=beam_width,
                    max_length=max_length,
                    temperature=temperature
                )
            
            return result
            
        except Exception as e:
            return f"Translation error: {str(e)}"


# Create Gradio interface
def create_interface(app):
    with gr.Blocks(title="SmolTransformer - English to Hindi Translation", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🌟 SmolTransformer Translation App
        
        A compact Encoder-Decoder Transformer for English to Hindi translation.
        
        **Instructions:**
        1. First, load a trained model (optional - will use random weights if no checkpoint)
        2. Enter English text to translate
        3. Choose generation method and adjust parameters
        4. Click translate to get Hindi output
        """)
        
        with gr.Tab("Translation"):
            with gr.Row():
                with gr.Column(scale=2):
                    # Translation section
                    gr.Markdown("### 💬 Translation")
                    input_text = gr.Textbox(
                        label="English Text",
                        placeholder="Enter English text to translate to Hindi...",
                        lines=3
                    )
                    
                    with gr.Row():
                        method = gr.Radio(
                            choices=["Top-K Sampling", "Beam Search"],
                            value="Top-K Sampling",
                            label="Generation Method"
                        )
                        max_length = gr.Slider(
                            minimum=10,
                            maximum=200,
                            value=100,
                            step=10,
                            label="Max Length"
                        )
                    
                    with gr.Row():
                        temperature = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=0.8,
                            step=0.1,
                            label="Temperature"
                        )
                        top_k = gr.Slider(
                            minimum=1,
                            maximum=100,
                            value=50,
                            step=1,
                            label="Top-K (for Top-K sampling)"
                        )
                        beam_width = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1,
                            label="Beam Width (for Beam Search)"
                        )
                    
                    translate_btn = gr.Button("🔄 Translate", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    gr.Markdown("### 🎯 Translation Output")
                    output_text = gr.Textbox(
                        label="Hindi Translation",
                        lines=5,
                        interactive=False
                    )
                    
                    gr.Markdown("### 📊 Model Information")
                    gr.Markdown(f"""
                    - **Model Size**: ~25M parameters
                    - **Architecture**: 6-layer Encoder-Decoder Transformer
                    - **Context Length**: {ModelArgs().block_size} tokens
                    - **Vocabulary**: IndicBARTSS tokenizer
                    - **Device**: {ModelArgs().device}
                    """)
        
        with gr.Tab("Examples"):
            gr.Markdown("### 🌟 Example Translations")
            
            # Create a row for each example with English and Hindi side by side
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### English")
                with gr.Column(scale=1):
                    gr.Markdown("### हिंदी (Hindi)")
            
            # Example 1
            with gr.Row():
                with gr.Column(scale=1):
                    eg1 = gr.Textbox("Hello! Myself an AI Assistant. How are you?", label="Example 1", interactive=False)
                with gr.Column(scale=1):
                    gr.Textbox("और (अज़सरतापा) तुम्हारे तुम लोग (बेहिश्त में) कैसे हो", label="Top-K Sampling", interactive=False)
                    gr.Textbox("मेरे लिए एक सहायक है", label="Beam Search", interactive=False)
            
            # Example 2
            with gr.Row():
                with gr.Column(scale=1):
                    eg2 = gr.Textbox("My name is Khan", label="Example 2", interactive=False)
                with gr.Column(scale=1):
                    gr.Textbox("मेरा नाम खान है", label="Translation", interactive=False)
            
            # Example 3
            with gr.Row():
                with gr.Column(scale=1):
                    eg3 = gr.Textbox("How are you?", label="Example 3", interactive=False)
                with gr.Column(scale=1):
                    gr.Textbox("आप कैसे थे?", label="Top-K Sampling", interactive=False)
                    gr.Textbox("आप कैसे हैं?", label="Beam Search", interactive=False)
            
            # Example 4
            with gr.Row():
                with gr.Column(scale=1):
                    eg4 = gr.Textbox("The AI will take our jobs ahhh!", label="Example 4", interactive=False)
                with gr.Column(scale=1):
                    gr.Textbox("आइए हम उसकी नौकरियां लाएँगे", label="Top-K Sampling", interactive=False)
                    gr.Textbox("एआई हमारी नौकरियां लेगा!", label="Beam Search", interactive=False)
            
            # Still include the clickable examples for easy testing
            gr.Markdown("### Try these examples:")
            examples = [
                ["Hello! Myself an AI Assistant. How are you?", "Top-K Sampling", 50, 0.8, 50, 5],
                ["My name is Khan", "Beam Search", 30, 0.8, 50, 5],
                ["How are you?", "Top-K Sampling", 20, 0.9, 50, 3],
                ["The AI will take our jobs ahhh!", "Beam Search", 40, 0.7, 40, 5]
            ]
            
            gr.Examples(
                examples=examples,
                inputs=[input_text, method, max_length, temperature, top_k, beam_width],
                outputs=output_text,
                fn=app.translate,
                cache_examples=False
            )
        
        with gr.Tab("About"):
            gr.Markdown("""
            ## 🏗️ SmolTransformer Architecture
            
            SmolTransformer is a compact implementation of the Transformer architecture for sequence-to-sequence translation tasks.
            
            ### Key Features:
            - **Encoder-Decoder Architecture**: Separate encoder and decoder with cross-attention
            - **Sinusoidal Positional Embeddings**: Better position encoding for sequences
            - **Multi-Head Attention**: Self-attention and cross-attention mechanisms
            - **Advanced Generation**: Top-K sampling and beam search methods
            - **Mixed Precision Training**: Efficient training with FP16
            
            ### Model Components:
            1. **Encoder**: Processes the source (English) text
            2. **Decoder**: Generates the target (Hindi) text autoregressively
            3. **Attention**: Masked self-attention in decoder, cross-attention between encoder-decoder
            
            ### Training Details:
            - **Dataset**: Hindi-English Samanantar dataset
            - **Optimizer**: AdamW with weight decay
            - **Learning Rate**: Warmup + cosine decay schedule
            - **Regularization**: Dropout and gradient clipping
            
            ### Generation Methods:
            - **Top-K Sampling**: Samples from top-k most likely tokens
            - **Beam Search**: Explores multiple sequence hypotheses
            
            ---
            
            **Note**: This is a demo model. For production use, train on larger datasets for better quality.
            """)
        
        # Event handler for translation
        translate_btn.click(
            fn=app.translate,
            inputs=[input_text, method, max_length, temperature, top_k, beam_width],
            outputs=[output_text]
        )
        
        # # Auto-load model on startup
        # demo.load(
        #     fn=lambda: app.load_model(),
        #     outputs=[]
        # )
    
    return demo

def parse_args():
    parser = argparse.ArgumentParser(description='Run the SmolTransformer translation app')
    parser.add_argument('--checkpoint', type=str, default="./checkpoints/snapshot_9999.pt",
                      help='Path to the model checkpoint file')
    parser.add_argument('--port', type=int, default=7860,
                      help='Port to run the Gradio app on (default: 7860)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                      help='Host to run the Gradio app on (default: 0.0.0.0)')
    parser.add_argument('--share', action='store_true',
                      help='Create a public shareable link')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug mode')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    print("Starting SmolTransformer Translation App...")
    print(f"Arguments: {vars(args)}")
    
    # Convert relative checkpoint path to absolute path
    # if args.checkpoint and not os.path.isabs(args.checkpoint):
    #     args.checkpoint = os.path.abspath(args.checkpoint)
    
    # Initialize the app with checkpoint
    print(f"Initializing app with checkpoint: {args.checkpoint}")
    app = TranslationApp(checkpoint_path=args.checkpoint)
    demo = create_interface(app)
    
    # Launch with command-line arguments
    print(f"Launching Gradio interface on {args.host}:{args.port}")
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        show_error=True
    )
