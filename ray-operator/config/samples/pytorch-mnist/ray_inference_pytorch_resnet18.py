"""
Distributed Inference Benchmark using Ray

This script defines an inference workload that runs a ResNet18 model on synthetic data.
The model runs in evaluation mode and processes a specified number of batches.
Multiple Ray remote tasks execute the inference workload concurrently.
The script aggregates the inference latency and throughput, printing summary statistics.
Designed to run on a KubeRay cluster.
"""

import time
import torch
import ray
import torchvision.models as models

@ray.remote
def inference_task(batch_size: int, num_batches: int, device: str = "cpu"):
    """
    Runs inference on synthetic data using a ResNet18 model.
    
    Parameters:
        batch_size (int): Number of images per batch.
        num_batches (int): Number of batches to process.
        device (str): Device for inference ("cpu" or "cuda").
    
    Returns:
        A tuple (total_inference_time, total_batches) where:
          - total_inference_time is the total time (in seconds) taken to process all batches.
          - total_batches is the number of batches processed.
    """
    # Create a ResNet18 model with random weights (self-contained; no downloading required).
    model = models.resnet18(pretrained=False)
    model.eval()
    model.to(device)
    
    total_time = 0.0
    with torch.no_grad():
        for _ in range(num_batches):
            # Generate synthetic input data: a random tensor with the shape expected by ResNet18.
            input_tensor = torch.randn(batch_size, 3, 224, 224).to(device)
            start = time.time()
            _ = model(input_tensor)
            end = time.time()
            total_time += (end - start)
    return total_time, num_batches

def main():
    # Connect to the existing Ray cluster (this is important when running on KubeRay).
    ray.init(address="auto", ignore_reinit_error=True)
    benchmark_start = time.time()

    # Inference benchmark parameters.
    num_tasks = 10         # Number of parallel inference tasks.
    batch_size = 32        # Images per batch.
    num_batches = 20       # Number of batches per task.

    # Launch Ray remote tasks for inference.
    futures = [inference_task.remote(batch_size, num_batches) for _ in range(num_tasks)]
    results = ray.get(futures)

    # Aggregate results from all tasks.
    total_inference_time = sum(r[0] for r in results)
    total_batches = sum(r[1] for r in results)
    total_images = batch_size * total_batches
    avg_time_per_batch = total_inference_time / total_batches
    throughput = total_images / total_inference_time

    benchmark_end = time.time()

    # Print benchmark results.
    print("Distributed Inference Benchmark Results:")
    print(f"Total inference time: {total_inference_time:.4f} seconds")
    print(f"Total batches processed: {total_batches}")
    print(f"Total images processed: {total_images}")
    print(f"Average time per batch: {avg_time_per_batch:.4f} seconds")
    print(f"Throughput: {throughput:.2f} images/second")
    print(f"Benchmark (including overhead) completed in {benchmark_end - benchmark_start:.4f} seconds")

    # Shutdown Ray.
    ray.shutdown()

if __name__ == "__main__":
    main()
