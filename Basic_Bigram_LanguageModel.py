import torch
from torch import nn
from torch.nn import functional as F
import time

#hyperparameters:
batchSize=32 # the batch sizs, let us denote it by B
blockSize=8 # number of consecutive characters we are taking, let us denot it by T 
maxIterations=10000 
evalInterval=300
lr=1e-2
device= 'cuda' if torch.cuda.is_available() else 'cpu'
evalIteration=200

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

class BigramLM(nn.Module):

    def __init__(self, vocabSize,device) -> None:
        super().__init__()
        self.token_embedding_table= nn.Embedding(vocabSize,vocabSize)#this will represent a tensor of dimention vocabSize x vocabSize and since we are working ont he character level so this table will represent the relation between each to characters. i.e the table is simillar to the correlation metrix where columns and rows represent all the charachters we have in the vocabulary. In another words, each cell will represent the logits (we can convert into probability) of character i to come after character j, where i represent the character on the columns and j represnet character on the rows.
        self.to(device)
    def forward(self, encoding,targets=None):
        #encoding variable and target are both (B,T) tensor of integers
        logits=self.token_embedding_table(encoding)# the encodings represent the index of the characters in the sorted vocabularies we created using s2i and encoding function. thus when we send them tot he embedding tab;e we created, for each batch we are going to take each row which represent set of consecutive characters and from that row we will pass each character encoding and we will get the row from the embedding table that correspond for it. thus we will get a tensor of the following shape (B,T,C) where C is the vocabSize.

        #we can calculate the loss also between the predictions and the targets using the nigative log likelihood which is implemented using cross_entropy function in pytorch. the cross_entropy can take the output (the predicted characters in fact their encoding) and the target (real values) or it gan take the logits and it will convert them into probability and take the most probabel character (encoding) and then compare it to the target output. In our case we will send the logits and thus it is required to convert the logits shape to some thing acceptable by the cross_entropy function.
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
            logits,_=self(encodings)
            logits=logits[:,-1,:] #we are predicting the new character based just on the last charachter thus no need to keep all the C arrays for each charachter in each batch. we need just the C arrays of the last letter. and to do that we take the last index on the T dimension by putting -1 for the T dimension.
            probs=F.softmax(logits,dim=-1)# this softmax will convert those logits into probability distribution for each C array.
            nextEncoding=torch.multinomial(probs,num_samples=1) #the multinomail function will take a probability distribution and will select a value based on it with the dimention (B,1) in our case
            encodings=torch.cat((encodings, nextEncoding),dim=1)# (B,T+1) 
        return encodings
    

model=BigramLM(vocabSize=vocabSize,device=device)
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

#testing the model
train(model=model)
text=decoder(model.generateText(encodings=torch.zeros((1,1),dtype=torch.long),maxTokensToGenerate=1000)[0].tolist())
for char in text:
    print(char,end="",flush=True)
    time.sleep(0.03)
