from dataclasses import dataclass
import torch # the all life torch library
import torch.nn as nn # Neural Network stuff
from torch.nn import functional as F #pydantic
import pydantic
import math

device = 'cuda'
import tiktoken

"""
max_return_sequences = 7 #Number of answers
max_length = 100 #Numbers of characters
enc = tiktoken.get_encoding('gpt2')

tokens = enc.encode("I like dogs because")

print(tokens,type(tokens[0]))

tokens=torch.tensor(tokens,dtype=torch.long)

tokens=tokens.unsqueeze(0).repeat(max_return_sequences,1)

print(tokens)

x=tokens.to('cuda')
"""

with open('/content/input.txt','r',encoding='utf-8') as f:
text = f.read()

text= text[:1000]
tokens = enc.encode(text)
B, T = 4, 32
buf = torch.tensor(tokens[:B*T+1])
x = buf[:-1].view(B,T) # input to transformer
y = buf[1:].view(B,T) # targets, to calculate loss
x.to('cuda')
y.to('cuda')

# deco dataclass -> Basemodel
# Config = dataclass(Config), then dataclass is a function!


class CausalSelfAttention(nn.Module): #Another class, for what reason we use the Causal name?
  def __init__(self,config):
    super().__init__()
    assert config.n_embd % config.n_head == 0 # 768/12 = 64 features per head.
    # key, query, value projection for all heads, (each) in batch (tensor)) How you specify the batch size

    self.c_attn = nn.Linear(config.n_embd,3*config.n_embd) #Linear layer, (dimension grow) # the number three because we need kery, key and value (three)

    #output projection

    self.c_proj = nn.Linear(config.n_embd,config.n_embd) #Linear layer (dimension stays the same)

    #regularization (?)
    self.n_head = config.n_head #config?
    self.n_embd = config.n_embd

    self.register_buffer("bias",torch.tril(torch.ones(config.block_size,config.block_size)).view(1,1,config.block_size,config.block_size)) # Applies causal (mask),
                                                                                                                  # basically creating a lower triangle
  def forward(self,x):

    B,T,C = x.size() # Batch Size=> how prompts are being processing at the same time
                     # T sequence length (contex size) numbers of word in one prompt sentence=>1024 (block size) ,
                     # C=embedding size = n_embd



    qkv = self.c_attn(x) # A linear layer, that obtain the key, query. matrices

    q,k,v = qkv.split(self.n_embd,dim=2) # split into three tensors each (B,T,C)

    k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # A tensor shape (B,T,12,64) and transpose for (B,12,T,64) - sense
    v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # same
    q = q.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # same

    att = (q@k.transpose(-2,-1))/math.sqrt(k.size(-1)) # context using kery and key, tranpose for a good multiplication matrix result matrix TXT word x word attend
                                                       # ; divides for sqrt (64).
    att = att.masked_fill(self.bias[:,:,:T,:T]==0,float('-inf')) # mask , -infi
    att = F.softmax(att,dim=-1) #Softmax, att turns into a vector of numbers

    y = att @ v # y return to a tensor
    y = y.transpose(1,2).contiguous().view(B,T,C)
    y = self.c_proj(y)

    return y

class MLP(nn.Module): # There's no communication between tokens in this step MULTILAYER PERCEPTRON

  def __init__(self,config):
    super().__init__()
    self.c_fc  = nn.Linear(config.n_embd,4*config.n_embd) #Linear layer that take the matrix to a space with more dimensions
    self.gelu  = nn.GELU(approximate='tanh') # Apply the pseudo activation function (?) than
    self.c_proj = nn.Linear(4*config.n_embd,config.n_embd) #return to the

  def forward(self,x): # practice
    x=self.c_fc(x)
    x=self.gelu(x) # gelu->gaussian error linear function
    x=self.c_proj(x)
    return x #straighforward

class Layer(nn.Module): #This contains the three classes.

  def __init__(self,config): #this config take sense in the initialization of instances
    super().__init__() #inherits, what inherits?

    self.ln_1 = nn.LayerNorm(config.n_embd) # I really don't understand what make this. pytorch: Applies Layer Normalization over a mini-batch of inputs.
    self.attn = CausalSelfAttention(config) #apply attention, (made by hand)
    self.ln_2 = nn.LayerNorm(config.n_embd) #normalize - residual, there exist a paper for understand this, that I wish.
    self.mlp  = MLP(config) # an instance of the MLP class that implements the ffn steps where the tokens are treated like individuals.

  def forward(self,x):# the question what recieves? a batch?

    x += self.attn(self.ln_1(x))
    x += self.mlp(self.ln_2(x))

    return x #So, this is called Pre-LN (normalize before computation).

from transformers import GPT2LMHeadModel # huggin face stuff



  from_pretrained=classmethod(from_pretrained) #decorator make it clean.

  #---------end---------------

  def forward(self, idx,target=None): #So recievec idx wich is a
    B,T = idx.size() # idx is a matrix, B is the batch size and T is the secuence lengh

    #no matter the embedding dimension? no obvious

    tok_emb = self.transformer.wte(idx) #

    pos_emb = self.transformer.wpe(torch.arange(T,device=idx.device)) #possitional encodding. The same but creating a positional embedding matrix

    x=tok_emb+pos_emb #yes the sum, x contains all the information, of the input

    for block in self.transformer.h: #transformers acting
      x = block(x)
    x = self.transformer.ln_f(x) #normalize it. residual optimize stuff

    logits = self.lm_head(x)
    target = None
    if target is not None:
      loss=F.cross_entropy(logits.view(-1,logits.size(-1)),target.view(-1))
    return logits, loss # so MODEL return logits raw number , ready to applied a softmax and then a topk


#model = MODEL.from_pretrained('gpt2') #creating and charging parameters for the model
#model.eval() #evaluation mode, kill the gradients?

logits, loss = model(x,y)
print(logits,loss)

import tiktoken
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
    x = torch.cat((x,xcol),dim=1) #append to form a sentence

for i in range(max_return_sequences):
  tokens = x[i,:max_length].tolist()
  decoded = enc.decode(tokens)
  print(">",decoded)