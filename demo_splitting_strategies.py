#!/usr/bin/env python3
# demo_splitting_strategies.py
"""
Demonstration of different text splitting strategies.
Shows the differences between recursive, sentence-aware, and semantic splitting.
"""

import sys
import os
sys.path.append('/home/runner/work/rag_for_learning/rag_for_learning')

from core.enhanced_splitters import get_enhanced_splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

def demo_splitting_strategies():
    """Demonstrate different text splitting strategies."""
    
    # Sample text with mixed Chinese and English content
    sample_text = """机器学习是人工智能的一个重要分支。它通过算法让计算机系统自动学习和改进性能。
    深度学习是机器学习的一个子集。它使用多层神经网络来模拟人脑的工作方式，能够处理复杂的模式识别任务。
    自然语言处理帮助计算机理解和生成人类语言。这项技术在机器翻译、聊天机器人、文本分析等领域有广泛应用。
    Computer vision enables machines to interpret and understand visual information. This technology is crucial for autonomous vehicles, medical imaging, and security systems.
    The future of AI lies in the integration of these technologies. Multi-modal AI systems that can process text, images, and audio simultaneously are becoming increasingly important."""
    
    print("=== Text Splitting Strategy Comparison ===\n")
    print(f"Original text length: {len(sample_text)} characters\n")
    print("Original text:")
    print(sample_text)
    print("\n" + "="*60 + "\n")
    
    # Test different strategies
    strategies = [
        ("Recursive Character Splitter", "recursive"),
        ("Sentence-Aware Splitter", "sentence_aware"),
    ]
    
    chunk_size = 200
    overlap = 30
    
    for strategy_name, strategy_type in strategies:
        print(f"=== {strategy_name} ===")
        print(f"Chunk size: {chunk_size}, Overlap: {overlap}")
        
        try:
            splitter = get_enhanced_splitter(
                strategy=strategy_type,
                chunk_size=chunk_size,
                overlap=overlap
            )
            
            chunks = splitter.split_text(sample_text)
            
            print(f"Number of chunks created: {len(chunks)}")
            
            for i, chunk in enumerate(chunks):
                print(f"\n--- Chunk {i+1} ({len(chunk)} chars) ---")
                print(chunk)
                
                # Show sentence boundaries in the chunk
                sentences = chunk.split('。')
                if len(sentences) > 1:
                    print(f"  → Contains {len(sentences)} sentence segments")
                    
        except Exception as e:
            print(f"Error with {strategy_name}: {e}")
        
        print("\n" + "-"*60 + "\n")
    
    # Demonstrate the benefits of sentence-aware splitting
    print("=== Benefits of Sentence-Aware Splitting ===")
    print("Sentence-aware splitting preserves semantic boundaries and reduces the risk")
    print("of breaking important context across chunks, which improves RAG performance.")
    print("\nKey advantages:")
    print("1. Respects sentence boundaries in both Chinese and English")
    print("2. Maintains semantic coherence within chunks")
    print("3. Reduces context fragmentation")
    print("4. Better handling of mixed-language content")

def demo_configuration_usage():
    """Show how to use the new configuration options."""
    
    print("\n=== Configuration Usage Examples ===\n")
    
    config_examples = [
        ("Default (Recursive)", "recursive", 400, 40),
        ("Sentence-Aware", "sentence_aware", 300, 50),
        ("Large Semantic Chunks", "semantic", 600, 60),
    ]
    
    print("You can configure text splitting by setting environment variables:")
    print("")
    
    for name, strategy, chunk_size, overlap in config_examples:
        print(f"# {name}")
        print(f"export TEXT_SPLITTING_STRATEGY={strategy}")
        if strategy == "sentence_aware":
            print(f"export SENTENCE_CHUNK_SIZE={chunk_size}")
            print(f"export SENTENCE_OVERLAP={overlap}")
        elif strategy == "semantic":
            print(f"export SEMANTIC_CHUNK_SIZE={chunk_size}")
            print(f"export SEMANTIC_SIMILARITY_THRESHOLD=0.7")
        else:
            print(f"# Uses default CHUNK_SIZE={chunk_size} and CHUNK_OVERLAP={overlap}")
        print("")

if __name__ == "__main__":
    demo_splitting_strategies()
    demo_configuration_usage()
    print("\n=== Demo Complete ===")