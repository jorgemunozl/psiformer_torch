import tiktoken
import torch
from config import MODEL


model = MODEL(Config())
model.to(device)

def main():
    enc = tiktoken.get_encoding('gpt2')
    torch.manual_seed(42) # random seed
    torch.cuda.manual_seed(42) # manual_seed
    while x.size(1) < max_length:
    with torch.no_grad(): # Let's take the
        logits = model(x)  # first call to MODEL IMPORTANT, shape (1,T,vocab_size) , (1,number of words, features per word)
        logits = logits[:,-1,:] #only last token
        probs = F.softmax(logits,dim=-1) #Softmax
        topk_probs, topk_indices =torch.topk(probs,50,dim=-1) #choose the 50 "words" more probably
        ix = torch.multinomial(topk_probs,1) # from those 50 choose randomly one.
        xcol = torch.gather(topk_indices,-1,ix) # assign that choose?
        x = torch.cat((x, xcol),dim=1) #append to form a sentence

    for i in range(max_return_sequences):
        tokens = x[i,:max_length].tolist()
        decoded = enc.decode(tokens)
        print(">",decoded)

if __name__ == "__main__":
    main()