"""
Parallel processing module for video analysis
This module provides functionality to process video chunks in parallel
"""

import concurrent.futures
import os
import time
import math
from typing import List, Tuple, Dict, Any, Optional, Callable

class ParallelProcessor:
    """
    Class for parallel processing of video chunks
    """
    
    def __init__(self, max_workers=None):
        """
        Initialize the parallel processor
        
        Args:
            max_workers: Maximum number of worker threads (defaults to CPU count)
        """
        self.max_workers = max_workers or (os.cpu_count() or 4)
    
    def run_parallel(self, task_function, task_data_list, *args, progress_callback=None):
        """
        Execute a task function in parallel across multiple chunks of data
        
        Args:
            task_function: Function to execute for each chunk (must accept chunk_data as first arg)
            task_data_list: List of data chunks to process in parallel
            *args: Additional arguments to pass to the task function
            progress_callback: Optional callback for progress updates
            
        Returns:
            list: Combined results from all parallel executions
        """
        start_time = time.time()
        print(f"Starting parallel processing with {self.max_workers} workers")
        
        # Use ThreadPoolExecutor for I/O-bound tasks
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_chunk = {
                executor.submit(
                    task_function, 
                    chunk_data, 
                    *args
                ): i for i, chunk_data in enumerate(task_data_list)
            }
            
            # Collect results as they complete
            results = []
            completed = 0
            
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_id = future_to_chunk[future]
                try:
                    chunk_result = future.result()
                    results.append((chunk_id, chunk_result))
                    
                    # Update progress if callback provided
                    completed += 1
                    if progress_callback:
                        progress_percent = completed / len(task_data_list)
                        progress_callback(
                            progress_percent,
                            f"Completed chunk {completed}/{len(task_data_list)}"
                        )
                except Exception as e:
                    print(f"Error processing chunk {chunk_id}: {e}")
            
            # Sort results by chunk_id to maintain original order
            results.sort(key=lambda x: x[0])
            
            # Extract just the results (without chunk IDs)
            final_results = [result for _, result in results]
            
        total_time = time.time() - start_time
        print(f"Parallel processing complete in {total_time:.2f}s")
        
        return final_results
    
    def split_into_chunks(self, data_list, chunk_count=None):
        """
        Split a list of data into chunks for parallel processing
        
        Args:
            data_list: List of data to split
            chunk_count: Number of chunks to create (defaults to max_workers)
            
        Returns:
            list: List of data chunks
        """
        if not data_list:
            return []
            
        count = chunk_count or self.max_workers
        chunk_size = math.ceil(len(data_list) / count)
        
        chunks = []
        for i in range(0, len(data_list), chunk_size):
            chunk = data_list[i:i + chunk_size]
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
                
        print(f"Split data list with {len(data_list)} items into {len(chunks)} chunks")
        return chunks

    def split_frame_range(self, start_frame, end_frame, chunk_count=None):
        """
        Split a range of frames into chunks for parallel processing
        
        Args:
            start_frame: Starting frame number
            end_frame: Ending frame number
            chunk_count: Number of chunks to create (defaults to max_workers)
            
        Returns:
            list: List of (start_frame, end_frame) tuples for each chunk
        """
        count = chunk_count or self.max_workers
        frame_count = end_frame - start_frame
        chunk_size = math.ceil(frame_count / count)
        
        chunks = []
        for i in range(start_frame, end_frame, chunk_size):
            chunk_start = i
            chunk_end = min(i + chunk_size, end_frame)
            chunks.append((chunk_start, chunk_end))
                
        print(f"Split frame range {start_frame}-{end_frame} into {len(chunks)} chunks")
        return chunks
    
    def combine_results(self, chunk_results, combine_function=None):
        """
        Combine results from parallel processing
        
        Args:
            chunk_results: List of results from each chunk
            combine_function: Optional function to combine results 
                             (defaults to simple flattening)
            
        Returns:
            Combined results
        """
        if combine_function:
            return combine_function(chunk_results)
            
        # Default behavior: flatten the list of lists
        flattened = []
        for result in chunk_results:
            if isinstance(result, list):
                flattened.extend(result)
            else:
                flattened.append(result)
                
        return flattened