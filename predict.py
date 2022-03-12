import torch
from model import CNN
import gzip
import numpy as np
from tqdm import trange

# ======================================== Get data ==================================================

#Function that turns the .gzip file to a data type variable
def fetchData(path):
    f = open(path,'rb')
    data = f.read()
    dec = gzip.decompress(data)
    return np.frombuffer(dec,dtype=np.uint8).copy()

#Import train images
xTest = fetchData('data/test-images-idx3-ubyte.gz')
yTest = fetchData('data/test-labels-idx1-ubyte.gz')

yTest = yTest[8:]

#Transform data to tensor 
xTest_tensor = torch.tensor(xTest[16:].reshape(-1, 1, 28, 28)).float() # The [16:] is so it ignores the first 16 bits so it's 60000 images
yTest_tensor = torch.tensor(yTest).long()

dataloader = (xTest_tensor, yTest_tensor) # To join all data

# =========================================== Configurations ====================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

# Get our model
model = CNN().to(device)

# Get the loss and optimizer function
loss_fn = torch.nn.CrossEntropyLoss()

# =================================== Test ===================================================
def test(dataloader, model, loss_fn):
    size = len(dataloader[1])
    numBatches = size
    model.eval()
    testLoss, correct = 0, 0
    
    with torch.no_grad():
        for batch in (t := trange(size)):
           #Organize data
            X = dataloader[0] # train data tensor
            y = dataloader[1] # train labels tensor
            
            tData = X[batch].unsqueeze(0) # increase dimention
            tData = tData.to(device) #, y[batch].to(device)
            
            tLabel = y[batch].unsqueeze(0)
            tLabel = tLabel.to(device)
            
            pred = model(tData)
            testLoss += loss_fn(pred, tLabel).item()
            correct += (pred.argmax(1) == tLabel).type(torch.float).sum().item()
            if batch % 100 == 0:
                t.set_description(f"batch {batch + 1}")
        testLoss /= numBatches
        correct /= size
        print(f'Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {testLoss:>8f}')
        

# ==================================== Main ==================================================

if __name__ == '__main__':
    #Load the model
    model.load_state_dict(torch.load("model.pth"))
    print('Model loaded!')
    
    # Possible classes in a list
    classes = [
        '0',
        '1',
        '2',
        '3',
        '4',
        '5',
        '6',
        '7',
        '8',
        '9',
    ]
    
    print('--- Testing ---')
    test(dataloader,model,loss_fn)
    print('--- Test complete ---')
    
    x, y = dataloader[0].to(device), dataloader[1].to(device)
    
    #If you want to change which number the agent is guessing change this variable
    testIndex = 7 # The index of the object being predicted
    
    with torch.no_grad():
        pred = model(x)
        predicted = classes[pred[testIndex].argmax(0)]
        actual =  classes[int(y[testIndex].item())]
        print('\n\n--- Prediction ---')
        print(f'Predicted: {predicted} | Actual: {actual}')
