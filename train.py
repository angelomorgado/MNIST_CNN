import torch
from model import CNN
import gzip
import numpy as np
from tqdm import trange

#======================================= Get Data ==================================================================

#Function that turns the .gzip file to a data type variable
def fetchData(path):
    f = open(path,"rb")
    data = f.read()
    dec = gzip.decompress(data)
    return np.frombuffer(dec,dtype=np.uint8).copy()

#Import train images
xTrain = fetchData("data/train-images-idx3-ubyte.gz") 
yTrain = fetchData("data/train-labels-idx1-ubyte.gz")
xTest = fetchData("data/test-images-idx3-ubyte.gz")
yTest = fetchData("data/test-labels-idx1-ubyte.gz")

yTrain = yTrain[8:] # Only consider the first 60000
yTest = yTest[8:]

#Transform data to tensor 
xTrain_tensor = torch.tensor(xTrain[16:].reshape(-1, 60, 1, 28, 28)).float() # Reshape(numOfImages, numOfChannels, height, width) || 
xTest_tensor = torch.tensor(xTest[16:].reshape(-1, 1, 28, 28)).float() # The [16:] is so it ignores the first 16 bits so it's 60000 images
yTrain_tensor = torch.tensor(yTrain).long().reshape(-1, 60)
yTest_tensor = torch.tensor(yTest).long()

dataloader = (xTrain_tensor, yTrain_tensor, xTest_tensor, yTest_tensor) # To join all data

#============================================ Configurations ======================================================

# Get cpu or gpu device for training
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f'Using "{device}" device')

# Get our model
model = CNN().to(device)

# Get the loss and optimizer function
loss_fn = torch.nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-3) # Stocastic gradient descent || lr = learning rate, how quicky it learns
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3) # Adam optimizer

#============================================ Train ===============================================================

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader[0]) # 1000
    model.train() # Change to training mode
    
    for batch in (t := trange(size)):
        #Organize data
        X = dataloader[0] # train data tensor
        y = dataloader[1] # train labels tensor
        
        tData = X[batch]#.unsqueeze(0) # increase dimention
        tData = tData.to(device) #, y[batch].to(device)
        
        tLabel = y[batch]#.unsqueeze(0)
        tLabel = tLabel.to(device)
        
        # Compute prediction error
        pred = model(tData) # Model's prediction
        loss = loss_fn(pred, tLabel)
        
        # Backprop
        optimizer.zero_grad()
        loss.backward() # DO NOT FORGET
        optimizer.step()
        
        if batch % 100 == 0:
            t.set_description(f"loss: {loss.item():>7f}")
        
        '''if batch % 100 == 0:
            loss, current = loss.item(), batch * len(tData)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")'''
 
            
#================================================ Test ============================================================

def test(dataloader, model, loss_fn):
    size = len(dataloader[3])
    numBatches = size
    model.eval()
    testLoss, correct = 0, 0
    
    with torch.no_grad():
        for batch in range(size):
           #Organize data
            X = dataloader[2] # train data tensor
            y = dataloader[3] # train labels tensor
            
            tData = X[batch].unsqueeze(0) # increase dimention
            tData = tData.to(device) #, y[batch].to(device)
            
            tLabel = y[batch].unsqueeze(0)
            tLabel = tLabel.to(device)
            
            pred = model(tData)
            testLoss += loss_fn(pred, tLabel).item()
            correct += (pred.argmax(1) == tLabel).type(torch.float).sum().item()
        testLoss /= numBatches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {testLoss:>8f} \n")
        
#================================================ Main =========================================================
if __name__ == "__main__":
    epochs = 2
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------------")
        train(dataloader, model, loss_fn, optimizer)
    
    test(dataloader,model,loss_fn)
    print("Done!")
    
    torch.save(model.state_dict(), "model.pth")
    print("Saved!")
        