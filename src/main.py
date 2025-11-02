import tiktoken
import torch
import torch.nn.functional as F
from utils import get_device, MODEL
from config import Config, outPut

torch.manual_seed(42)
torch.cuda.manual_seed(42)


def main(sentence):

    device = get_device()
    model = MODEL(Config())
    model.to(device)
    max_length = outPut.max_length
    max_return_sequences = outPut.max_return_seq

    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(sentence)
    tokens = torch.tensor(tokens, dtype=torch.long)
    x = tokens.to(device)

    while x.size(1) < max_length:
        with torch.no_grad():
            logits = model(x)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix)
            x = torch.cat((x, xcol), dim=1)

    for i in range(max_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)


if __name__ == "__main__":
    main("Hello my name is")
