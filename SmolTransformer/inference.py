import torch
import torch.nn.functional as F
from config import ModelArgs
import copy

model_args = ModelArgs()

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
            tgt_self_attention_mask = torch.ones_like(decoder_input_ids, device=device)

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

def save_text(step, text):
    """Save generated text to file"""
    with open(f'generated_data/generations_{step}.txt', 'w') as f:
        f.write(f"------------------------------------------------Step: {step}--------------------------------------------\n\n")
        f.write(text + "\n\n")
