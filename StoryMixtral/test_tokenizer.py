#!/usr/bin/env python3
"""
Test script to verify the new tokenizer implementation works correctly
"""

import os
import sys

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_tokenizer_basic():
    """Test basic tokenizer functionality"""
    print("Testing basic tokenizer functionality...")
    
    from tokenizer import Tokenizer
    
    # Test with no token (should use public model)
    tokenizer = Tokenizer()
    
    test_text = "Hello, world! This is a test."
    
    # Test encoding
    tokens = tokenizer.encode(test_text)
    print(f"Original text: {test_text}")
    print(f"Encoded tokens: {tokens}")
    
    # Test decoding
    decoded = tokenizer.decode(tokens)
    print(f"Decoded text: {decoded}")
    
    # Test vocab size
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocabulary size: {vocab_size}")
    
    print("✅ Basic tokenizer test passed!")
    return True

def test_config_factory():
    """Test the config factory functions"""
    print("\nTesting config factory functions...")
    
    from config import create_tokenizer, create_model_args
    
    # Test tokenizer creation
    tokenizer_instance = create_tokenizer()
    print(f"Tokenizer created: {type(tokenizer_instance)}")
    
    # Test model args creation
    model_args = create_model_args()
    print(f"Model args created: {type(model_args)}")
    print(f"Model args has tokenizer: {hasattr(model_args, 'tokenizer')}")
    print(f"Model args vocab size: {model_args.vocab_size}")
    
    print("✅ Config factory test passed!")
    return True

def test_tokenizer_with_hf_token():
    """Test tokenizer with HF token"""
    print("\nTesting tokenizer with HF token...")
    
    from config import create_tokenizer
    
    # Test with various token scenarios
    test_cases = [
        None,  # No token
        "...",  # Placeholder token
        os.environ.get('HF_TOKEN', '...'),  # Environment token
    ]
    
    for i, token in enumerate(test_cases):
        print(f"Test case {i+1}: token={token}")
        tokenizer_instance = create_tokenizer(token)
        
        # Test basic functionality
        test_text = "Once upon a time"
        tokens = tokenizer_instance.encode(test_text)
        decoded = tokenizer_instance.decode(tokens)
        
        print(f"  Encoded/decoded successfully: {test_text} -> {decoded}")
    
    print("✅ HF token test passed!")
    return True

def main():
    """Run all tests"""
    print("🚀 Starting StoryMixtral tokenizer tests...\n")
    
    tests = [
        test_tokenizer_basic,
        test_config_factory, 
        test_tokenizer_with_hf_token,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
    
    print(f"\n🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The tokenizer refactor is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
