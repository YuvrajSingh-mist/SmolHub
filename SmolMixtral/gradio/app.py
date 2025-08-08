import gradio as gr
import torch
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import create_model_args, create_tokenizer
from model import Mixtral
from inference import topk_sampling

# Global variables
model = None
model_args = None
tokenizer_instance = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_components(hf_token=None):
    """Initialize model args and tokenizer"""
    global model_args, tokenizer_instance, tokenizer
    
    # Create model args with tokenizer
    model_args = create_model_args(hf_token=hf_token)
    tokenizer_instance = model_args.tokenizer_instance
    tokenizer = model_args.tokenizer
    
    return model_args, tokenizer_instance, tokenizer

def load_model(checkpoint_path=None, hf_token=None):
    global model, model_args, tokenizer_instance, tokenizer
    
    # Initialize components if not already done
    if model_args is None:
        initialize_components(hf_token)
    
    model = Mixtral(
        attn_dropout=model_args.attn_dropout,
        embeddings_dims=model_args.embeddings_dims,
        no_of_heads=model_args.no_of_heads,
        block_size=model_args.block_size,
        dropout=model_args.dropout,
        no_of_decoder_layers=model_args.no_of_decoder_layers,
        vocab_size=model_args.vocab_size,
        device=device,
        tokenizer=tokenizer
    )
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Remove _orig_mod prefix from checkpoint keys if present
        state_dict = checkpoint['MODEL_STATE']
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace('_orig_mod.', '') if key.startswith('_orig_mod.') else key
                new_state_dict[new_key] = value
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
        print(f"Loaded model from {checkpoint_path}")
    else:
        print("No checkpoint provided, using randomly initialized model")
    
    model = model.to(device)
    model.eval()
    return model

def generate_text(prompt, max_length=100, top_k=50, temperature=1.0):
    global model, tokenizer_instance
    
    if model is None:
        load_model()
    
    try:
        generated_text = topk_sampling(
            model=model,
            prompt=prompt,
            tokenizer=tokenizer_instance,
            device=device,
            max_length=max_length,
            top_k=top_k,
            temperature=temperature
        )
        return generated_text
    except Exception as e:
        return f"Error generating text: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="SmolMixtral Text Generator") as demo:
    gr.Markdown("# SmolMixtral Text Generator")
    gr.Markdown("Generate text using the Mixtral model")
    
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Enter your prompt here...",
                value="Once upon a time",
                lines=3
            )
            
            with gr.Row():
                max_length = gr.Slider(
                    minimum=10,
                    maximum=512,
                    value=100,
                    step=10,
                    label="Max Length"
                )
                top_k = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=50,
                    step=1,
                    label="Top K"
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="Temperature"
                )
            
            generate_btn = gr.Button("Generate", variant="primary")
        
        with gr.Column():
            output_text = gr.Textbox(
                label="Generated Text",
                placeholder="Generated text will appear here...",
                lines=10,
                max_lines=20
            )
    
    generate_btn.click(
        fn=generate_text,
        inputs=[prompt_input, max_length, top_k, temperature],
        outputs=output_text
    )

if __name__ == "__main__":
    # Initialize components first
    hf_token = os.environ.get('HF_TOKEN')  # Get HF token from environment
    initialize_components(hf_token)
    
    # Try to load model on startup
    checkpoint_path = "/mnt/c/Users/yuvra/OneDrive/Desktop/Work/pytorch/SmolMixtral/checkpoints/snapshot_18000_.pt"  # Adjust path as needed
    load_model(checkpoint_path, hf_token)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
