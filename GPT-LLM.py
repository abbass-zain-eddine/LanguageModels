import torch
from torch import nn
from torch.nn import functional as F
import time

#hyperparameters:
batchSize=64 # the batch sizs, let us denote it by B
blockSize=256 # number of consecutive characters we are taking, let us denot it by T 
maxIterations=2500 
evalInterval=500
lr=3e-4
device= 'cuda' if torch.cuda.is_available() else 'cpu'
evalIteration=200
embeddingSize=384
numHeads=6
numBlocks=6
dropout=0.2
###############################


#function to read the dataset from txt file in our case
def get_whole_data(path):
    with open(path,'r') as fr:
        return fr.readlines()

#path to the data set
path="input.txt"
#read the data and extracting the unique carachters found in this curpos,
# it is also important to know that we are going to work on character level  #tokens.
data=''.join(get_whole_data(path))

uniqueChars=sorted(list(set(data)))
print(f"the unique values in the dataset: {''.join(uniqueChars)}")
vocabSize=len(uniqueChars)
#create and encoder and decoder to transform our characters into integers and #vise versa
s2i={ch:i for i,ch in enumerate(uniqueChars)}
i2s={i:ch for i,ch in enumerate(uniqueChars)}
encoder= lambda s: [s2i[c] for c in s ]
decoder= lambda e: ''.join([i2s[i] for i in e])
data=torch.tensor(encoder(data),dtype=torch.long)
#split data to 90% for training and 10% for validataion 
n=int(0.9*len(data))
trainData=data[:n]
valData=data[n:]

#print(encoder("abbass ZEIN EDDINE!"))
#print(decoder(encoder("Abbass ZEIN EDDINE!")))

#create a get batch function that will return for us a batch of dimension (batchSize x blockSize). this fuction is similar to dataloader.
def get_batch(split):
    data=trainData if split=='train' else valData

    randIndx=torch.randint(len(data)-blockSize,(batchSize,))
    x=torch.stack([data[i:i+blockSize] for i in randIndx]) #we take a block size list starting from the randomly selected index i
    y=torch.stack([data[i+1:i+blockSize+1] for i in randIndx])#we take a block size list strating from i+1 index, and this is because each set of charecters starting from 0 until index j in the x array will be used to output the char at position j in the y array. So in fact for each row in the x and y arrays we have several training example.
    x,y=x.to(device),y.to(device)# jsut move the data to the same device as the model.
    return x,y

class Head(nn.Module):
    """single head of self attention that is created of a decoder block because in our case we want to generate text and in this case we just need is that each token can just talk to the tokens before it so when we multiplied queries and keys we just multiply the current query by the previous keys. in Encoder block such as the case when we are making sentiment analysis we will multiply all the tokens with all the keys before and after the current token.
    what are queries, keys and values:
    each token will emit a query and a key. the query vector represent "what I am (the token) looking for" and the key value represent " what I ( the token) contain". then we make a dote product between the keys and the queries. so for each token, its query will be multiplied by all the keys of all other tokens. that is each token will search by its query in all the other tokens keys. 
    simply query, key and value are created as simple linear layer without bias."""

    def __init__(self, headSize,embeddingSize=32,dropout=0.2):
        super().__init__()
        self.keys=nn.Linear(embeddingSize,headSize,bias=False)
        self.query=nn.Linear(embeddingSize,headSize,bias=False)
        self.value=nn.Linear(embeddingSize,headSize,bias=False)
        #this is not a parameter and it is not trainable thus to create it we use register_buffer method. the goal of this triangle variable is to create a lower triangle matrix of ones. this will be used to convert all the result of the dot product between queries to a lower triangle matrix and thus converts the values related to the relation of the current token and the future tokens to zero. and this what we mentioned in the deffinition of the class.
        self.register_buffer('triangle',torch.tril(torch.ones(blockSize,blockSize)))
        self.dropout=nn.Dropout(dropout)

    def forward(self,input):
        B,T,C =input.shape
        k=self.keys(input) # (B,T,C)
        q=self.query(input)  # (B,T,C)
        v=self.value(input)  # (B,T,C)
        #computing the attention scores
        weightes=q @ k.transpose(-2,-1) * C**-0.5 #C**-0.5 this is to normalize the weights for the soft max to work better and this demonstrated in the "attention is all you need" article. (B,T,C) @ (B,T,C) ==> (B,T,T)
        weightes = weightes.masked_fill(self.triangle[:T,:T] ==0,float('-inf'))# here where we use the triangle variable to remove the relation of the token with future tokens and convert them to -infinity which will be equal to zero when we apply softmax in the next line.(B,T,T) 
        weightes=F.softmax(weightes,dim=-1) #(B,T,T)
        weightes=self.dropout(weightes)
        output= weightes @ v #(B,T,T) @ (B,T,C) ==> (B,T,C)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self,numHeads,headSize,embeddingSize,dropout=0.2 ):
        super().__init__()
        self.heads=nn.ModuleList([Head(headSize,embeddingSize) for  _ in range(numHeads)])
        self.ffw=nn.Linear(embeddingSize,embeddingSize)
        self.dropout=nn.Dropout(dropout)
    def forward(self, input):
        output=torch.cat([head(input) for head in self.heads],dim=-1)
        output=self.dropout(self.ffw(output))
        return output
    

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self,numHeads, headSize,embeddingSize,dropout=0.2 ):
        super().__init__()
        self.sa=MultiHeadAttention(numHeads,headSize,embeddingSize)
        self.ffw=nn.Sequential(
            nn.Linear(embeddingSize,4*embeddingSize), #the multiplication by 4 is also inspired by the "Attention is all you need" article
            nn.ReLU(),
            nn.Linear(4*embeddingSize,embeddingSize),
            nn.Dropout(dropout)
        )
        self.ln1=nn.LayerNorm(embeddingSize)
        self.ln2=nn.LayerNorm(embeddingSize)

    def forward(self,input):
        x= input + self.sa(self.ln1(input)) #layer norm in the "Attention is all you need " article are applied after the application of the selfattention (self.sa) and after the ffw layers also. But in other versions of transformers it seems that it is better to apply the layernorm just befor the selfattention and the ffw.
        x= x + self.ffw(self.ln2(x))
        return x



class BigramLM(nn.Module):

    def __init__(self, vocabSize,device,numBloks=10,embeddingSize=36,dropout=0.2) -> None:
        super().__init__()
        self.token_embedding_table= nn.Embedding(vocabSize,embeddingSize)
        self.positin_embedding_table=nn.Embedding(blockSize,embeddingSize)
        self.blocks= nn.Sequential(*[MultiHeadAttentionBlock(numHeads,embeddingSize//numHeads,embeddingSize,dropout=dropout) for _ in range(numBloks)],
        
        )
        self.ln=nn.LayerNorm(embeddingSize)       
        self.output=nn.Linear(embeddingSize,vocabSize)
        self.to(device)

    def forward(self, encoding,targets=None):
        B,T = encoding.shape
        #encoding variable and target are both (B,T) tensor of integers
        tokenEmbedd=self.token_embedding_table(encoding) #(B,T,C)
        positionEmbedd=self.positin_embedding_table(torch.arange(T,device=device)) #(T,C)
        totalEmbedd=tokenEmbedd+positionEmbedd #(B,T,C)
        x=self.blocks(totalEmbedd)
        x=self.ln(x)
        logits=self.output(x)# (B,T,vocabSize)
        
        if targets is None:
            loss=None
        else:
            B,T,C= logits.shape
            logits=logits.view(B*T,C)
            targets=targets.view(B*T)
            loss=F.cross_entropy(logits,targets)
        return logits,loss
    
    def generateText(self,encodings,maxTokensToGenerate):
        for _ in range(maxTokensToGenerate):
            encodingsPart=encodings[:,-blockSize:]#this is because we are adding positional encoding in the Head class thus we cannot have more than the blockSize values in the encoding.
            logits,_=self(encodingsPart)
            logits=logits[:,-1,:] #we are predicting the new character based just on the last charachter thus no need to keep all the C arrays for each charachter in each batch. we need just the C arrays of the last letter. and to do that we take the last index on the T dimension by putting -1 for the T dimension.
            probs=F.softmax(logits,dim=-1)# this softmax will convert those logits into probability distribution for each C array.
            nextEncoding=torch.multinomial(probs,num_samples=1) #the multinomail function will take a probability distribution and will select a value based on it with the dimention (B,1) in our case
            encodings=torch.cat((encodings, nextEncoding),dim=1)# (B,T+1) 
        return encodings
    

model=BigramLM(vocabSize=vocabSize,device=device,numBloks=numBlocks,embeddingSize=embeddingSize,dropout=dropout)
# xb,yb=get_batch("train")
# logits,loss=model(xb,yb)
# print(loss)
# print(logits.shape)

optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)

def evalModel(model):
    out={}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(evalIteration)
        for k in range(evalIteration):
            X,Y=get_batch(split)
            _,loss=model(X,Y)
            losses[k]=loss.item()
        out[split]=losses.mean()
    model.train()
    return out

def train(model):
    for step in range(maxIterations):
        if step % evalInterval==0:
            losses=evalModel(model)
            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")


        xb,yb=get_batch('train')

        logits,loss=model(xb,yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    #testing the model
    modelPath="model.pth"
    case="load"
    TextLength=1000


    if case=="load":
        model=torch.load(modelPath)
    else:
        train(model=model)
        torch.save(model, 'model.pth')


    text=decoder(model.generateText(encodings=torch.zeros((1,1),dtype=torch.long),maxTokensToGenerate=TextLength)[0].tolist())
    for char in text:
        print(char,end="",flush=True)
        time.sleep(0.03)
