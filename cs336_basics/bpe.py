from ast import literal_eval



def get_pretokens(file_path: str) -> dict[tuple[bytes, ...], int]:
    with open(file_path, "r") as f:
        token_str = f.read()

    token_list = token_str.split("\n")

    pretoken_dict: dict[tuple[bytes, ...], int] = {}

    for pretoken in token_list:
        try:
            k,v = pretoken.split(";")
            byte_tuple = tuple(map(lambda x: x.to_bytes(1, "big"), literal_eval(k)))
            pretoken_dict[byte_tuple] = int(v)
        except:
            pass
    
    return pretoken_dict

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    vocab = {i:bytes([i]) for i in range(256)}
    pretoken_dict = get_pretokens(input_path)
    merges: list[tuple[bytes,bytes]] = [] 

    for st in special_tokens:
        vocab[len(vocab)] = st.encode("utf-8")

    for _ in range(vocab_size - len(vocab)):
        pair_count: dict[tuple[bytes, bytes], int] = {}

        for s,f in pretoken_dict.items():
            for a,b in zip(s, s[1:]):
                pair_count[(a,b)] = pair_count.get((a,b),0) + f

        if not pair_count:
            break

        max_pair = max(pair_count, key=lambda x: (pair_count[x], x[0], x[1]))

        vocab[len(vocab)] = max_pair[0]+max_pair[1]
        merges.append(max_pair)

        new_dict: dict[tuple[bytes, ...], int] = {}
        for s,f in pretoken_dict.items():
            out, i = [], 0
            while i < len(s):
                if i + 1 < len(s) and s[i] == max_pair[0] and s[i+1] == max_pair[1]:
                    i += 2
                else:
                    out.append(s[i])
                    i += 1
            ns = tuple(out)
            new_dict[ns] = new_dict.get(ns, 0) + f

        pretoken_dict = new_dict

    return vocab, merges



if __name__=="__main__":
    SPLIT_TOKEN = b"<|endoftext|>"
    
    # print(get_pretokens("artefacts/pretokens-TinyStoriesV2-GPT4-valid.txt"))

    vocab, merges = train_bpe("artefacts/pretokens-TinyStoriesV2-GPT4-valid.txt", 2500, [SPLIT_TOKEN.decode("utf-8")])

    print(vocab)

