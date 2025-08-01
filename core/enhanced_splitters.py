# core/enhanced_splitters.py
"""
Enhanced text splitting strategies for improved semantic coherence in RAG systems.
"""

import re
import jieba
import logging
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter


class SentenceAwareSplitter:
    """
    A splitter that respects sentence boundaries for better semantic coherence.
    Particularly effective for Chinese text processing.
    """
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Chinese sentence endings
        self.chinese_sentence_endings = r'[。！？；]'
        # English sentence endings
        self.english_sentence_endings = r'[.!?]'
        
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences, handling both Chinese and English."""
        # Combine Chinese and English sentence patterns
        sentence_pattern = f'({self.chinese_sentence_endings}|{self.english_sentence_endings})'
        
        # Split by sentence endings but keep the endings
        parts = re.split(sentence_pattern, text)
        
        sentences = []
        for i in range(0, len(parts) - 1, 2):
            if i + 1 < len(parts):
                sentence = parts[i] + parts[i + 1]
                sentence = sentence.strip()
                if sentence:
                    sentences.append(sentence)
        
        # Handle the last part if it doesn't end with punctuation
        if len(parts) % 2 == 1 and parts[-1].strip():
            sentences.append(parts[-1].strip())
            
        return sentences
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks while respecting sentence boundaries."""
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return []
        
        # If text is short enough to fit in one chunk, return as is
        if len(text) <= self.chunk_size:
            return [text.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size and we have content
            if current_chunk and len(current_chunk) + len(sentence) > self.chunk_size:
                # Save current chunk
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap from previous chunk
                if self.overlap > 0 and chunks:
                    # Find overlap text from the end of the last chunk
                    last_chunk = chunks[-1]
                    if len(last_chunk) > self.overlap:
                        overlap_text = last_chunk[-self.overlap:]
                        # Try to find a sentence boundary in the overlap
                        overlap_sentences = self._split_into_sentences(overlap_text)
                        if overlap_sentences:
                            current_chunk = overlap_sentences[-1] + " "
                        else:
                            current_chunk = overlap_text + " "
                    else:
                        current_chunk = last_chunk + " "
                else:
                    current_chunk = ""
            
            # Handle very long sentences that exceed chunk size
            if len(sentence) > self.chunk_size:
                # If we have content in current chunk, save it first
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # Split the long sentence by character limits
                words = sentence.split()
                temp_chunk = ""
                for word in words:
                    if len(temp_chunk) + len(word) + 1 > self.chunk_size:
                        if temp_chunk.strip():
                            chunks.append(temp_chunk.strip())
                        temp_chunk = word + " "
                    else:
                        temp_chunk += word + " "
                
                if temp_chunk.strip():
                    current_chunk = temp_chunk
            else:
                current_chunk += sentence + " "
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks


class SemanticSplitter:
    """
    A splitter that uses semantic similarity to determine optimal chunk boundaries.
    Uses sentence embeddings to group semantically related content together.
    """
    
    def __init__(self, embedding_model: SentenceTransformer, 
                 chunk_size: int = 600, 
                 similarity_threshold: float = 0.7):
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.similarity_threshold = similarity_threshold
        self.sentence_splitter = SentenceAwareSplitter(chunk_size=100, overlap=0)
        
    def _compute_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Compute embeddings for a list of sentences."""
        if not sentences:
            return np.array([])
        
        # Filter out very short sentences that might not be meaningful
        meaningful_sentences = [s for s in sentences if len(s.strip()) > 5]
        if not meaningful_sentences:
            return np.array([])
            
        try:
            embeddings = self.embedding_model.encode(meaningful_sentences)
            return np.array(embeddings)
        except Exception as e:
            logging.warning(f"Error computing embeddings: {e}")
            return np.array([])
    
    def _find_semantic_boundaries(self, sentences: List[str], embeddings: np.ndarray) -> List[int]:
        """Find boundaries where semantic similarity drops significantly."""
        if len(embeddings) < 2:
            return [len(sentences)]
        
        boundaries = []
        
        for i in range(1, len(embeddings)):
            # Compute similarity between consecutive sentences
            similarity = cosine_similarity(
                embeddings[i-1:i], 
                embeddings[i:i+1]
            )[0][0]
            
            # If similarity drops below threshold, it's a potential boundary
            if similarity < self.similarity_threshold:
                boundaries.append(i)
        
        # Always include the end as a boundary
        boundaries.append(len(sentences))
        
        return boundaries
    
    def split_text(self, text: str) -> List[str]:
        """Split text into semantically coherent chunks."""
        # First split into sentences
        sentences = self.sentence_splitter._split_into_sentences(text)
        
        if not sentences:
            return []
        
        if len(sentences) == 1:
            # If only one sentence, apply regular chunking if it's too long
            if len(sentences[0]) > self.chunk_size:
                regular_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=50,
                    separators=["\n\n", "\n", "。", "，", "；", "：", " ", ""]
                )
                return regular_splitter.split_text(sentences[0])
            return sentences
        
        try:
            # Compute embeddings for sentences
            embeddings = self._compute_sentence_embeddings(sentences)
            
            if len(embeddings) == 0:
                # Fallback to sentence-aware splitting if embeddings fail
                logging.warning("Semantic chunking failed, falling back to sentence-aware splitting")
                fallback_splitter = SentenceAwareSplitter(chunk_size=self.chunk_size, overlap=50)
                return fallback_splitter.split_text(text)
            
            # Find semantic boundaries
            boundaries = self._find_semantic_boundaries(sentences, embeddings)
            
            # Create chunks based on boundaries
            chunks = []
            start_idx = 0
            
            for boundary in boundaries:
                chunk_sentences = sentences[start_idx:boundary]
                chunk_text = " ".join(chunk_sentences).strip()
                
                # If chunk is too large, split it further
                if len(chunk_text) > self.chunk_size * 1.5:  # Allow some flexibility
                    # Use sentence-aware splitter for oversized chunks
                    sub_splitter = SentenceAwareSplitter(
                        chunk_size=self.chunk_size, 
                        overlap=50
                    )
                    sub_chunks = sub_splitter.split_text(chunk_text)
                    chunks.extend(sub_chunks)
                elif chunk_text:  # Only add non-empty chunks
                    chunks.append(chunk_text)
                
                start_idx = boundary
            
            return chunks
            
        except Exception as e:
            logging.warning(f"Semantic chunking error: {e}, falling back to sentence-aware splitting")
            # Fallback to sentence-aware splitting
            fallback_splitter = SentenceAwareSplitter(chunk_size=self.chunk_size, overlap=50)
            return fallback_splitter.split_text(text)


def get_enhanced_splitter(strategy: str, embedding_model: Optional[SentenceTransformer] = None, 
                         chunk_size: int = 400, overlap: int = 40, 
                         similarity_threshold: float = 0.7) -> object:
    """
    Factory function to create the appropriate text splitter based on strategy.
    
    Args:
        strategy: "recursive", "semantic", or "sentence_aware"
        embedding_model: Required for semantic strategy
        chunk_size: Target chunk size
        overlap: Overlap between chunks
        similarity_threshold: Threshold for semantic similarity (semantic strategy only)
    
    Returns:
        Text splitter instance
    """
    if strategy == "semantic":
        if embedding_model is None:
            logging.warning("Embedding model required for semantic splitting. Falling back to sentence_aware.")
            strategy = "sentence_aware"
        else:
            return SemanticSplitter(
                embedding_model=embedding_model,
                chunk_size=chunk_size,
                similarity_threshold=similarity_threshold
            )
    
    if strategy == "sentence_aware":
        return SentenceAwareSplitter(chunk_size=chunk_size, overlap=overlap)
    
    # Default to recursive character splitter
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", "，", "；", "：", " ", ""]
    )