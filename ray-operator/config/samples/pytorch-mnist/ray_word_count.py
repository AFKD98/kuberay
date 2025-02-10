"""
Distributed Word Count Benchmark using Ray

This script downloads the text of "Moby Dick" from Project Gutenberg (if not cached),
splits the text into chunks, and distributes the word count computation across available Ray workers.
It then aggregates the results and prints summary statistics. This benchmark is self-contained
and does not rely on any pre-existing data.
"""

import os
import re
import time
import requests
import ray
from collections import Counter

# Remote function to count words in a chunk of text.
@ray.remote
def count_words(text_chunk: str) -> Counter:
    # Remove punctuation and convert to lowercase.
    cleaned = re.sub(r"[^\w\s]", "", text_chunk)
    words = cleaned.lower().split()
    return Counter(words)

def download_text(url: str, filename: str) -> str:
    """
    Downloads the text from the given URL if the file does not exist locally.
    Returns the text content as a string.
    """
    if not os.path.exists(filename):
        print(f"Downloading data from {url} ...")
        response = requests.get(url)
        response.raise_for_status()  # Ensure the request succeeded.
        with open(filename, "w", encoding="utf-8") as f:
            f.write(response.text)
    else:
        print("Using cached data.")
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()

def aggregate_counters(counters) -> Counter:
    """
    Aggregates a list of Counter objects into a single Counter.
    """
    total = Counter()
    for counter in counters:
        total.update(counter)
    return total

def main():
    # Initialize Ray.
    ray.init(ignore_reinit_error=True)
    start_time = time.time()

    # URL for "Moby Dick" (from Project Gutenberg) and local cache filename.
    moby_dick_url = "https://www.gutenberg.org/files/2701/2701-0.txt"
    filename = "moby_dick.txt"

    # Download or load the text.
    text = download_text(moby_dick_url, filename)
    print(f"Text downloaded. Length: {len(text)} characters.")

    # Split the text into chunks.
    # Here we split by lines and group a fixed number of lines per chunk.
    lines = text.splitlines()
    chunk_size = 500  # Number of lines per chunk.
    chunks = [
        "\n".join(lines[i:i + chunk_size])
        for i in range(0, len(lines), chunk_size)
    ]
    print(f"Text split into {len(chunks)} chunks.")

    # Launch Ray tasks for each chunk.
    futures = [count_words.remote(chunk) for chunk in chunks]
    counters = ray.get(futures)

    # Aggregate the word counts from all chunks.
    total_counter = aggregate_counters(counters)
    most_common = total_counter.most_common(10)

    end_time = time.time()

    # Print results.
    print("\nTop 10 most common words:")
    for word, count in most_common:
        print(f"{word}: {count}")
    total_words = sum(total_counter.values())
    print(f"\nTotal word count: {total_words}")
    print(f"Distributed word count benchmark completed in {end_time - start_time:.2f} seconds.")
    print('Benchmark Completed Successfully.')

    # Shutdown Ray.
    ray.shutdown()

if __name__ == "__main__":
    main()
