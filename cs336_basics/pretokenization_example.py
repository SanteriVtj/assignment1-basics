import os
from typing import BinaryIO
import regex as re
from functools import reduce
import time
from multiprocessing import Pool


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def count_words(chunk: list[str]) -> dict[tuple[bytes, ...], int]:
    word_count = {}
    for word in chunk:
        bword = tuple(word.encode("utf-8"))
        if bword in word_count:
            word_count[bword] += 1
        else:
            word_count[bword] = 1
    
    return word_count

def merge_count_dicts(wcl: dict[tuple[bytes ,...], int], wcr: dict[tuple[bytes ,...], int]) -> dict[tuple[bytes ,...],int]:
    wcl_cpy = wcl.copy()

    for k,v in wcr.items():
        if k in wcl_cpy:
            wcl_cpy[k] += v
        else:
            wcl_cpy[k] = v
    
    return wcl_cpy

def read_from_n(start: int, end: int, f: BinaryIO) -> str:
    f.seek(start)
    chunk = f.read(end - start).decode("utf-8", errors="ignore")
    return chunk

def pretokenize(x):
    chunk = read_from_n(x[0], x[1], f)
    pretokens = re.findall(PAT, chunk)
    
    return count_words(pretokens) 

## Usage
if __name__=="__main__":
    with open("data/TinyStoriesV2-GPT4-valid.txt", "rb") as f:
    # with open("data/small_example.txt", "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        # count_list = []

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        t0 = time.time()

        p = Pool(num_processes)
        count_list = p.map(pretokenize, zip(boundaries[:-1], boundaries[1:]))

        print(f"Time {(time.time()-t0):.3f}")
        
        pretoken_counts = reduce(merge_count_dicts, count_list)
        
        # ds = ""
        # for i, (k,v) in enumerate(pretoken_counts.items()):
        #     sep = "\n" if i % 5 == 0 else ""
        #     ds += f" {bytes(k).decode("utf-8")}:{v}"
        # print(ds)