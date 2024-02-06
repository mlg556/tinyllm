# %%
import torch

# %%
# tinyshakespeare from https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt

fname_inp = "tinyshakespeare.txt"

with open(fname_inp, "r") as f:
    text = f.read()

# %%
alphabet = sorted(set(text))

print(
    f"{len(text):_} chars. alphabet is: '{''.join(alphabet)}' with len={len(alphabet)} \ntext start: \n-------------\n{text[:200]}"
)

# %%
# tokenize characters

# set comprehensions, neat!
char_to_int = {ch: i for i, ch in enumerate(alphabet)}
int_to_char = {i: ch for i, ch in enumerate(alphabet)}

print(list(char_to_int.items())[:10])
print(list(int_to_char.items())[:10])


# %%
# encode/decode
def encode(s: str) -> list[int]:
    return [char_to_int[c] for c in s]


def decode(l: list[int]) -> str:
    return "".join([int_to_char[i] for i in l])


# %%
print(encode("hello there"), decode(encode("hello there")))

# %%
data = torch.tensor(encode(text), dtype=torch.long)
