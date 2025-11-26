"""
Optimized Parallel Whisper Processing Module

This module provides high-performance parallel Whisper processing by using
multiple lightweight model instances instead of a single locked instance.
"""

import whisper
import tempfile
import os
import threading
import concurrent.futures
import psutil
from typing import List, Tuple, Optional, Dict, Any
from pydub import AudioSegment
import wave
import time

class ParallelWhisperProcessor:
    """
    High-performance parallel Whisper processor that uses multiple model instances
    for true parallel processing instead of serialized access to a single model.
    """
    
    def __init__(self, model_size="tiny", max_workers=None):
        """
        Initialize parallel Whisper processor
        
        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
            max_workers: Maximum number of parallel workers (defaults to CPU count)
        """
        self.model_size = model_size
        cpu_count = os.cpu_count() or 4
        self.max_workers = max_workers or min(4, cpu_count)  # Limit to prevent memory overflow
        self._model_pool = {}
        self._model_lock = threading.Lock()
        
        print(f"Initializing parallel Whisper processor with {self.max_workers} workers using '{model_size}' model")
    
    def _get_worker_model(self, worker_id: int):
        """
        Get or create a Whisper model for a specific worker thread
        
        Args:
            worker_id: Unique identifier for the worker thread
            
        Returns:
            Whisper model instance for this worker
        """
        with self._model_lock:
            if worker_id not in self._model_pool:
                try:
                    print(f"Loading Whisper {self.model_size} model for worker {worker_id}")
                    self._model_pool[worker_id] = whisper.load_model(self.model_size, device="cpu")
                    print(f"Worker {worker_id} model loaded successfully")
                except Exception as e:
                    print(f"Error loading model for worker {worker_id}: {e}")
                    raise
            
            return self._model_pool[worker_id]
    
    def _transcribe_chunk(self, chunk_data: Tuple[int, str, Optional[str]]) -> Tuple[int, Dict[str, Any]]:
        """
        Transcribe a single audio chunk using worker-specific model
        
        Args:
            chunk_data: Tuple of (chunk_id, audio_file_path, language)
            
        Returns:
            Tuple of (chunk_id, transcription_result)
        """
        chunk_id, audio_path, language = chunk_data
        
        # Get current thread ID as worker ID
        worker_id = threading.get_ident() % self.max_workers
        
        try:
            # Get worker-specific model
            model = self._get_worker_model(worker_id)
            
            # Optimized transcription options for speed
            options = {
                "task": "transcribe",
                "verbose": False,
                "word_timestamps": True,
                "fp16": False,
                "beam_size": 1,  # Greedy decoding for speed
                "best_of": 1,
                "temperature": 0.0,
                "compression_ratio_threshold": 2.4,
                "logprob_threshold": -1.0,
                "no_speech_threshold": 0.6,
            }
            
            # Add language if specified (skip auto-detection for speed)
            if language:
                options["language"] = language
            
            start_time = time.time()
            result = model.transcribe(audio_path, **options)
            end_time = time.time()
            
            print(f"Worker {worker_id} completed chunk {chunk_id} in {end_time - start_time:.2f}s")
            
            return chunk_id, result
            
        except Exception as e:
            print(f"Worker {worker_id} failed to transcribe chunk {chunk_id}: {e}")
            return chunk_id, {"text": "", "segments": [], "error": str(e)}
    
    def transcribe_audio_chunks_parallel(self, audio_chunks: List[Tuple[float, AudioSegment]], 
                                       language: Optional[str] = None) -> List[Tuple[float, str]]:
        """
        Transcribe multiple audio chunks in parallel
        
        Args:
            audio_chunks: List of (timestamp, audio_segment) tuples
            language: Optional language code for all chunks
            
        Returns:
            List of (timestamp, transcribed_text) tuples
        """
        if not audio_chunks:
            return []
        
        print(f"Starting parallel transcription of {len(audio_chunks)} chunks with {self.max_workers} workers")
        
        # Prepare temporary files for each chunk
        temp_files = []
        chunk_data = []
        
        try:
            for i, (timestamp, audio_segment) in enumerate(audio_chunks):
                # Create temporary file for this chunk
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_file.close()
                
                # Export audio segment to temporary file
                audio_segment.export(temp_file.name, format="wav", parameters=["-ac", "1", "-ar", "16000"])
                
                temp_files.append(temp_file.name)
                chunk_data.append((i, temp_file.name, language))
            
            # Process chunks in parallel
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all transcription tasks
                future_to_chunk = {executor.submit(self._transcribe_chunk, data): data[0] 
                                 for data in chunk_data}
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_chunk):
                    chunk_id = future_to_chunk[future]
                    try:
                        chunk_id, result = future.result()
                        timestamp = audio_chunks[chunk_id][0]
                        text = result.get('text', '').strip()
                        results.append((chunk_id, timestamp, text))
                    except Exception as e:
                        print(f"Chunk {chunk_id} generated an exception: {e}")
                        timestamp = audio_chunks[chunk_id][0]
                        results.append((chunk_id, timestamp, ""))
            
            # Sort results by original chunk order and return
            results.sort(key=lambda x: x[0])
            return [(timestamp, text) for _, timestamp, text in results]
            
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
    
    def cleanup(self):
        """Clean up all loaded models"""
        with self._model_lock:
            for worker_id, model in self._model_pool.items():
                try:
                    del model
                    print(f"Cleaned up model for worker {worker_id}")
                except:
                    pass
            self._model_pool.clear()


# Global parallel processor instance
_parallel_whisper_instance = None
_parallel_lock = threading.Lock()


def get_optimized_parallel_whisper_processor() -> ParallelWhisperProcessor:
    """
    Get optimized parallel Whisper processor based on system capabilities
    
    Returns:
        ParallelWhisperProcessor instance
    """
    global _parallel_whisper_instance
    
    with _parallel_lock:
        if _parallel_whisper_instance is None:
            try:
                # Determine optimal configuration based on system resources
                memory_gb = psutil.virtual_memory().total / (1024**3)
                cpu_count = os.cpu_count() or 4
                
                # Choose model size based on memory and parallel workers
                if memory_gb >= 16 and cpu_count >= 8:
                    model_size = "base"
                    max_workers = min(4, cpu_count // 2)  # Conservative to prevent memory issues
                elif memory_gb >= 8 and cpu_count >= 4:
                    model_size = "tiny"
                    max_workers = min(3, cpu_count // 2)
                else:
                    model_size = "tiny"
                    max_workers = 2  # Safe default
                
                print(f"System: {memory_gb:.1f}GB RAM, {cpu_count} CPUs")
                print(f"Using parallel Whisper: {max_workers} workers with '{model_size}' model")
                
                _parallel_whisper_instance = ParallelWhisperProcessor(
                    model_size=model_size, 
                    max_workers=max_workers
                )
                
            except Exception as e:
                print(f"Error initializing parallel processor: {e}")
                # Fallback to single worker tiny model
                _parallel_whisper_instance = ParallelWhisperProcessor(
                    model_size="tiny", 
                    max_workers=1
                )
        
        return _parallel_whisper_instance


def transcribe_audio_segments_parallel(audio_chunks: List[Tuple[float, AudioSegment]], 
                                     language: Optional[str] = None) -> List[Tuple[float, str]]:
    """
    Convenience function for parallel audio transcription
    
    Args:
        audio_chunks: List of (timestamp, audio_segment) tuples
        language: Optional language code
        
    Returns:
        List of (timestamp, transcribed_text) tuples
    """
    processor = get_optimized_parallel_whisper_processor()
    return processor.transcribe_audio_chunks_parallel(audio_chunks, language)