#!/usr/bin/env python3
"""
Simple launcher for SmolTransformer Gradio app
Run from the SmolTransformer root directory
"""

import os
import sys

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Now import and run the app
try:
    from gradio.app import create_interface
    
    print("🚀 Launching SmolTransformer Translation App...")
    print("📍 Make sure you're running from the SmolTransformer root directory")
    print("🔗 The app will be available at http://localhost:7860")
    
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True for public sharing
        debug=True
    )
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("📝 Make sure all dependencies are installed:")
    print("   pip install gradio")
    print("   Run from SmolTransformer root directory")
    
except Exception as e:
    print(f"❌ Error launching app: {e}")
    print("📝 Check that all model files are properly set up")
