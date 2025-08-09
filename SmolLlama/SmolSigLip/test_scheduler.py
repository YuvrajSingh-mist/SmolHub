#!/usr/bin/env python3
"""
Test script for the LinearWarmupCosineDecayLR scheduler
Verifies that the learning rate progression follows the expected schedule.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(__file__))

from train import LinearWarmupCosineDecayLR

def test_scheduler():
    """Test the learning rate scheduler and plot the progression."""
    
    # Create a simple model and optimizer for testing
    model = nn.Linear(10, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Scheduler configuration (matching our training setup)
    WARMUP_EXAMPLES = 200_000_000  # 200M examples
    TOTAL_EXAMPLES = 700_000_000   # 700M examples
    PEAK_LR = 0.001
    INITIAL_LR = 1e-4  # Current head learning rate
    MIN_LR = 0.0
    EFFECTIVE_BATCH_SIZE = 16384
    
    # Create scheduler
    scheduler = LinearWarmupCosineDecayLR(
        optimizer=optimizer,
        initial_lr=INITIAL_LR,
        peak_lr=PEAK_LR,
        min_lr=MIN_LR,
        warmup_examples=WARMUP_EXAMPLES,
        total_examples=TOTAL_EXAMPLES,
        effective_batch_size=EFFECTIVE_BATCH_SIZE
    )
    
    # Test learning rate progression
    lrs = []
    steps = []
    
    # Sample every 100 steps for visualization
    for step in range(0, scheduler.total_steps + 1, 100):
        scheduler.step_count = step
        lr = scheduler.get_lr()
        lrs.append(lr)
        steps.append(step)
        
        # Print key milestones
        if step == 0:
            print(f"Step {step:6d}: LR = {lr:.6f} (Initial)")
        elif step == scheduler.warmup_steps:
            print(f"Step {step:6d}: LR = {lr:.6f} (End of warmup)")
        elif step == scheduler.total_steps:
            print(f"Step {step:6d}: LR = {lr:.6f} (End of training)")
    
    # Verify key points
    print("\n📊 Verification:")
    
    # Test initial LR
    scheduler.step_count = 0
    initial_lr = scheduler.get_lr()
    print(f"Initial LR: {initial_lr:.6f} (expected: {INITIAL_LR:.6f})")
    assert abs(initial_lr - INITIAL_LR) < 1e-8, f"Initial LR mismatch: {initial_lr} != {INITIAL_LR}"
    
    # Test peak LR (at end of warmup)
    scheduler.step_count = scheduler.warmup_steps
    peak_lr = scheduler.get_lr()
    print(f"Peak LR: {peak_lr:.6f} (expected: {PEAK_LR:.6f})")
    assert abs(peak_lr - PEAK_LR) < 1e-8, f"Peak LR mismatch: {peak_lr} != {PEAK_LR}"
    
    # Test final LR
    scheduler.step_count = scheduler.total_steps
    final_lr = scheduler.get_lr()
    print(f"Final LR: {final_lr:.6f} (expected: {MIN_LR:.6f})")
    assert abs(final_lr - MIN_LR) < 1e-8, f"Final LR mismatch: {final_lr} != {MIN_LR}"
    
    # Test mid-warmup (should be between initial and peak)
    scheduler.step_count = scheduler.warmup_steps // 2
    mid_warmup_lr = scheduler.get_lr()
    expected_mid_warmup = INITIAL_LR + (PEAK_LR - INITIAL_LR) * 0.5
    print(f"Mid-warmup LR: {mid_warmup_lr:.6f} (expected: {expected_mid_warmup:.6f})")
    assert abs(mid_warmup_lr - expected_mid_warmup) < 1e-8, f"Mid-warmup LR mismatch"
    
    print("\n✅ All scheduler tests passed!")
    
    # Plot the learning rate schedule
    plt.figure(figsize=(12, 8))
    
    # Convert steps to examples for x-axis
    examples = [step * EFFECTIVE_BATCH_SIZE for step in steps]
    examples_millions = [e / 1_000_000 for e in examples]
    
    plt.subplot(2, 1, 1)
    plt.plot(examples_millions, lrs, 'b-', linewidth=2)
    plt.axvline(x=WARMUP_EXAMPLES/1_000_000, color='r', linestyle='--', alpha=0.7, label='End of Warmup (200M)')
    plt.axhline(y=PEAK_LR, color='g', linestyle='--', alpha=0.7, label=f'Peak LR ({PEAK_LR})')
    plt.xlabel('Training Examples (Millions)')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule: Linear Warmup + Cosine Decay')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Zoomed view of warmup phase
    plt.subplot(2, 1, 2)
    warmup_idx = len([e for e in examples_millions if e <= WARMUP_EXAMPLES/1_000_000])
    plt.plot(examples_millions[:warmup_idx], lrs[:warmup_idx], 'b-', linewidth=2)
    plt.xlabel('Training Examples (Millions)')
    plt.ylabel('Learning Rate')
    plt.title('Warmup Phase (First 200M Examples)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/speech/advait/yuvraj/LLMs/SmolSigLip/lr_schedule.png', dpi=300, bbox_inches='tight')
    print(f"\n📈 Learning rate schedule plot saved to: lr_schedule.png")
    
    return True

def test_optimizer_integration():
    """Test that the scheduler properly updates optimizer learning rates."""
    print("\n🔧 Testing optimizer integration...")
    
    # Create model and optimizer
    model = nn.Linear(10, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Create scheduler
    scheduler = LinearWarmupCosineDecayLR(
        optimizer=optimizer,
        initial_lr=1e-4,
        peak_lr=0.001,
        min_lr=0.0,
        warmup_examples=1000,  # Small values for quick test
        total_examples=5000,
        effective_batch_size=100
    )
    
    # Test that step() updates optimizer learning rates
    initial_lr = optimizer.param_groups[0]['lr']
    print(f"Initial optimizer LR: {initial_lr:.6f}")
    
    # Take a few steps
    for i in range(3):
        new_lr = scheduler.step()
        optimizer_lr = optimizer.param_groups[0]['lr']
        print(f"Step {i+1}: scheduler LR = {new_lr:.6f}, optimizer LR = {optimizer_lr:.6f}")
        assert abs(new_lr - optimizer_lr) < 1e-8, "Scheduler and optimizer LR mismatch!"
    
    print("✅ Optimizer integration test passed!")
    return True

if __name__ == "__main__":
    print("🧪 Testing LinearWarmupCosineDecayLR scheduler...\n")
    
    # Run tests
    test_scheduler()
    test_optimizer_integration()
    
    print("\n🎉 All tests completed successfully!")
