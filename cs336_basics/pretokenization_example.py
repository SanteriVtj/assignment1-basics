import os
from typing import BinaryIO
import regex as re
from functools import reduce
import time
from multiprocessing import Pool
import logging
from datetime import datetime


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

def count_words(chunk: tuple[int,int]) -> dict[tuple[bytes, ...], int]:
    with open(FILE_PATH, "rb") as f:
        text = read_from_n(chunk[0], chunk[1], f)

    word_count = {}
    for segment in text.split(SPLIT_TOKEN.decode("utf-8")):
        for word in PAT.finditer(segment):
            bword = tuple(word.group().encode("utf-8"))
            if bword in word_count:
                word_count[bword] += 1
            else:
                word_count[bword] = 1
    
    return word_count

def read_from_n(start: int, end: int, f: BinaryIO) -> str:
    f.seek(start)
    chunk = f.read(end - start).decode("utf-8", errors="ignore")
    return chunk


## Usage
if __name__=="__main__":
    num_processes = os.cpu_count() or 4

    CHUNK_SIZE = num_processes*1024**2
    PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    FILE_PATH = "data/TinyStoriesV2-GPT4-train.txt"
    # FILE_PATH = "data/small_example.txt"
    SPLIT_TOKEN = b"<|endoftext|>"
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='pretokenizer.log', level=logging.INFO)

    
    with open(FILE_PATH, "rb") as f:
        f.seek(0, os.SEEK_END)
        f_size = f.tell()

        target_size = max(num_processes, f_size // CHUNK_SIZE)
        boundaries = find_chunk_boundaries(f, target_size, SPLIT_TOKEN)

    chunks = list(zip(boundaries[:-1], boundaries[1:]))
    pretoken_counts: dict[tuple[bytes, ...], int] = {}

    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    t0 = time.time()

    with Pool(num_processes) as p:
        for c in p.imap_unordered(count_words, chunks, chunksize=4):
            for k,v in c.items():
                pretoken_counts[k] = pretoken_counts.get(k,0) + v

    logger.info(f"{FILE_PATH.split("/")[-1]} pretokenized {datetime.today().strftime('%Y-%m-%d %H:%M:%S')} in {(time.time()-t0):.3f}s")

    token_str = ""
    for k,v in pretoken_counts.items():
        token_str += f"{k}; {v}\n"
    with open(f"artefacts/pretokens-{FILE_PATH.split("/")[-1]}", "w") as f:
        f.write(token_str)
        logger.info(f"Pretokens written to artefacts/pretokens-{FILE_PATH.split("/")[-1]}.txt")

    
    # ds = ""
    # for i, (k,v) in enumerate(pretoken_counts.items()):
    #     sep = "\n" if i % 5 == 0 else ""
    #     ds += f" {bytes(k).decode("utf-8")}:{v}"
    # print(ds)