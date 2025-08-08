import gradio as gr
import torch
import torch.nn.functional as F
import os
import sys
import argparse

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ModelArgs, get_args
from model import DeepSeekV3, initialize_tokenizer
from tokenizer import Tokenizer
from inference import topk_sampling

# Global tokenizer variable
tk = None

# Parse HF token from command line or environment
def get_app_args():
    parser = argparse.ArgumentParser(description='StoryKimi Gradio App - Kimi K2 Inspired Model')
    parser.add_argument('--hf_token', type=str, default=None, help='Hugging Face token for accessing gated models')
    parser.add_argument('--model_dir', type=str, default='./checkpoints', help='Directory containing model checkpoints')
    parser.add_argument('--port', type=int, default=7860, help='Port to run the Gradio app')
    parser.add_argument('--share', action='store_true', default=False, help='Create public link')
    return parser.parse_args()




# Model paths - you can modify these paths based on your checkpoints
model_paths = {
    "Checkpoint 2000": "/mnt/c/Users/yuvra/OneDrive/Desktop/Work/pytorch/StoryKimi/checkpoints/checkpoint_2000.pt",
}

def load_model(model_path, device, model_args):
    """Load model from checkpoint"""
    model = DeepSeekV3(
        embeddings_dims=model_args.embeddings_dims,
        block_size=model_args.block_size,
        vocab_size=model_args.vocab_size,
        dropout=model_args.dropout,
        device=device
    )
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
        print(f"Model loaded from {model_path}")
    else:
        print(f"Checkpoint {model_path} not found. Using randomly initialized model.")
    
    return model



def generate_text(prompt, model_choice, max_length, temperature, top_k):
    """Generate text using the selected model and top-k sampling"""
    global tk
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model args
    model_args = ModelArgs()
    
    # Load the selected model
    model_path = model_paths.get(model_choice, "checkpoint_latest.pt")
    model = load_model(model_path, device, model_args)
    model = model.to(device)
    try:
        generated_text = topk_sampling(
            model=model,
            prompt=prompt,
            device=device,
            max_length=max_length,
            top_k=top_k,
            temperature=temperature,
            tokenizer=tk
        )
            
        return generated_text
        
    except Exception as e:
        return f"Error generating text: {str(e)}"

def create_interface():
    """Create the Gradio interface"""
    global tk
    
    # Initialize tokenizer with token inside the function
    app_args = get_app_args()
    print(f"HF Token: {app_args.hf_token}")
    
    # Initialize both global tokenizers
    if tk is None:
        tk = Tokenizer(hf_token=app_args.hf_token)
        tk = tk.ready_tokenizer()
    
    # Initialize the global tokenizer in model.py
    initialize_tokenizer(hf_token=app_args.hf_token)
    
    # Initialize model args to get default values
    model_args = ModelArgs()
    
    with gr.Blocks(title="StoryKimi Text Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🚀 StoryKimi Text Generator")
        gr.Markdown("Generate text using the Kimi K2 inspired StoryKimi model with different checkpoints and generation methods.")
        
        with gr.Row():
            with gr.Column(scale=2):
                prompt_input = gr.Textbox(
                    label="Input Prompt",
                    placeholder="Enter your prompt here...",
                    lines=3,
                    value="Once upon a time there lived a baby deer named Bambi."
                )
                
                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        choices=list(model_paths.keys()),
                        label="Model Checkpoint",
                        value="Latest"
                    )
                
                with gr.Row():
                    max_length_slider = gr.Slider(
                        minimum=10,
                        maximum=128,
                        value=50,
                        step=10,
                        label="Max Length"
                    )
                    
                    temperature_slider = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.9,
                        step=0.1,
                        label="Temperature"
                    )
                
                with gr.Row():
                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=50,
                        step=1,
                        label="Top-k"
                    )
                
                generate_btn = gr.Button("🎯 Generate Text", variant="primary", size="lg")
                
            with gr.Column(scale=3):
                output_text = gr.Textbox(
                    label="Generated Text",
                    lines=15,
                    interactive=False
                )
                
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary")
        
        # Event handlers
        generate_btn.click(
            fn=generate_text,
            inputs=[
                prompt_input,
                model_dropdown,
                max_length_slider,
                temperature_slider,
                top_k_slider
            ],
            outputs=output_text
        )
        
        clear_btn.click(
            fn=lambda: ("", ""),
            outputs=[prompt_input, output_text]
        )
        
        # Model information
        gr.Markdown("## ℹ️ Model Information")
        gr.Markdown(f"""
        - **Model Architecture**: Kimi K2 inspired (StoryKimi)
        - **Parameters**: {model_args.embeddings_dims} embedding dims, {model_args.no_of_decoder_layers} layers
        - **Experts**: {model_args.experts} experts with top-{model_args.top_experts} routing
        - **Block Size**: {model_args.block_size} tokens
        - **Vocabulary Size**: {model_args.vocab_size} tokens
        """)
    
    return demo

if __name__ == "__main__":
    # Get app arguments
    app_args = get_app_args()
    
    # Create and launch the interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=app_args.port,
        share=app_args.share,
        debug=True
    )
