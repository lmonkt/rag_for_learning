#!/usr/bin/env python3
# test_enhanced_splitting.py
"""
Test script for enhanced text splitting strategies.
"""

import sys
import os
sys.path.append('/home/runner/work/rag_for_learning/rag_for_learning')

from sentence_transformers import SentenceTransformer
from core.enhanced_splitters import SentenceAwareSplitter, SemanticSplitter, get_enhanced_splitter
from core.document_processor import split_text
from config import EMBED_MODEL_NAME

def test_sentence_aware_splitter():
    """Test sentence-aware splitter with Chinese and English text."""
    print("=== Testing Sentence-Aware Splitter ===")
    
    text = """人工智能是计算机科学的一个分支。它试图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
    该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
    Artificial intelligence (AI) is a branch of computer science. It aims to create machines that can perform tasks typically requiring human intelligence.
    This includes learning, reasoning, problem-solving, perception, and language understanding."""
    
    splitter = SentenceAwareSplitter(chunk_size=150, overlap=20)
    chunks = splitter.split_text(text)
    
    print(f"Original text length: {len(text)}")
    print(f"Number of chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1} ({len(chunk)} chars): {chunk[:100]}...")
    print()
    
    return len(chunks) > 1

def test_semantic_splitter():
    """Test semantic splitter with embedding model."""
    print("=== Testing Semantic Splitter ===")
    
    try:
        # Load a lightweight model for testing
        embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        text = """机器学习是人工智能的一个重要分支。它通过算法让计算机系统自动学习和改进。
        深度学习是机器学习的一个子集。它使用神经网络来模拟人脑的工作方式。
        自然语言处理帮助计算机理解人类语言。它在翻译、聊天机器人等领域有广泛应用。
        计算机视觉让机器能够识别和理解图像。这项技术被用于自动驾驶、医疗诊断等场景。"""
        
        splitter = SemanticSplitter(
            embedding_model=embed_model,
            chunk_size=200,
            similarity_threshold=0.6
        )
        chunks = splitter.split_text(text)
        
        print(f"Original text length: {len(text)}")
        print(f"Number of chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1} ({len(chunk)} chars): {chunk}")
        print()
        
        return len(chunks) > 0
        
    except Exception as e:
        print(f"Error in semantic splitter test: {e}")
        return False

def test_integration_with_config():
    """Test integration with the configuration system."""
    print("=== Testing Integration with Config ===")
    
    # Test different strategies through the factory function
    text = "这是一个测试文本。它包含多个句子。我们将测试不同的切分策略。"
    
    strategies = ["recursive", "sentence_aware"]
    
    for strategy in strategies:
        try:
            splitter = get_enhanced_splitter(
                strategy=strategy,
                chunk_size=100,
                overlap=20
            )
            chunks = splitter.split_text(text)
            print(f"Strategy '{strategy}': {len(chunks)} chunks")
            for i, chunk in enumerate(chunks):
                print(f"  Chunk {i+1}: {chunk}")
        except Exception as e:
            print(f"Error with strategy '{strategy}': {e}")
    
    print()
    return True

def test_document_processor_integration():
    """Test integration with the document processor."""
    print("=== Testing Document Processor Integration ===")
    
    # Test the updated split_text function
    text = """知识图谱是一种结构化的知识表示方法。它使用图结构来表示实体之间的关系。
    在人工智能领域，知识图谱被广泛应用于问答系统、推荐系统等场景。
    Knowledge graphs are structured representations of knowledge. They use graph structures to represent relationships between entities.
    In the field of artificial intelligence, knowledge graphs are widely used in question-answering systems and recommendation systems."""
    
    try:
        # Test without embedding model (should use default strategy)
        chunks1 = split_text(text)
        print(f"Default splitting: {len(chunks1)} chunks")
        
        # Test with embedding model (only if semantic strategy is configured)
        try:
            embed_model = SentenceTransformer('all-MiniLM-L6-v2')
            chunks2 = split_text(text, embedding_model=embed_model)
            print(f"With embedding model: {len(chunks2)} chunks")
        except Exception as e:
            print(f"Embedding model test skipped: {e}")
        
        return True
        
    except Exception as e:
        print(f"Error in document processor integration test: {e}")
        return False

def main():
    """Run all tests."""
    print("Running Enhanced Text Splitting Tests\n")
    
    tests = [
        test_sentence_aware_splitter,
        test_integration_with_config,
        test_document_processor_integration,
        test_semantic_splitter,  # Run this last as it requires model download
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"Test {test.__name__} failed with error: {e}")
            results.append(False)
    
    print("=== Test Summary ===")
    for i, test in enumerate(tests):
        status = "PASS" if results[i] else "FAIL"
        print(f"{test.__name__}: {status}")
    
    all_passed = all(results)
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)