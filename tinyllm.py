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

# %%
print(data.shape, data.dtype)

print(data[:100])

# %%
N = int(0.9 * len(data))

# split the data into training and validation

train_data = data[:N]
validation = data[N:]

# %%

# this basically determines how much the LM can "remember". when training/infering, only the previous BLOCK_SIZE-1 characters are used

BLOCK_SIZE = 8  # context length

# %%

# there are BLOCK_SIZE-1 "examples" in BLOCK_SIZE characters, so we add +1
print("data:", train_data[: BLOCK_SIZE + 1].tolist())

x = train_data[:BLOCK_SIZE]
y = train_data[1 : BLOCK_SIZE + 1]

for t in range(BLOCK_SIZE):
    context = x[: t + 1]
    target = y[t]
    print(f"when context: {context.tolist()}, target: {target}")

# %%

torch.manual_seed(1337)
BATCH_SIZE = 4
BLOCK_SIZE = 8
