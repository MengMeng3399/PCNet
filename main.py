import torch
import torch.nn as nn
print(torch.cuda.is_available())
from lib.bcr_utils import *
import random
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score, log_loss
import pandas as pd
from tensor_utils import *

# 1.pytorch() 
class BaselineDeepNetwork(nn.Module):
    def __init__(self, featDict, featPosDict, featType_enabled, targetAPINum, reductSize=128, initStd=0.0001, dropoutRate=0.5, device='cuda'):
        super(BaselineDeepNetwork, self).__init__() 
        self.featDict = featDict
        self.featPosDict = featPosDict
        self.featType_enabled = featType_enabled
        self.targetAPINum = targetAPINum         
        self.reductSize = reductSize              
        self.device = device                    
        self.dropoutRate = dropoutRate             
        self.initStd = initStd                    
        self.flag_mask = True                       
        self.hiddenUnits = [200, 80]            
        self.mashupFeats = []                   
        self.apiFeats = []                      
        self.reductFeats = []                   
        for featType in self.featType_enabled:
            for feat in self.featDict['mashup'][featType]:
                self.mashupFeats.append(feat)
                self.reductFeats.append(feat)
            for feat in self.featDict['api'][featType]:
                self.apiFeats.append(feat)
                self.reductFeats.append(feat)   
    
        self.mashupFeats.append('MashupDes')
        self.reductFeats.append('MashupDes')
        self.apiFeats.append('Des')
        self.reductFeats.append('Des')

    
        linearsDict = {}
        for feat in self.reductFeats:
            if feat in self.mashupFeats:
                linearsDict[feat] = nn.Linear(self.featPosDict[feat][1] - self.featPosDict[feat][0], self.reductSize, bias=False)
            elif feat in self.apiFeats:
                linearsDict[feat] = nn.Linear(self.featPosDict['c_' + feat][1] - self.featPosDict['c_' + feat][0], self.reductSize, bias=False)
        self.reductLinearsDict = nn.ModuleDict(linearsDict)
        for parameter in self.reductLinearsDict.values():
            nn.init.normal_(parameter.weight, mean=0) 
        self.reductLinearsDict.to(self.device)

        self.mashupEmbeddingSize = 0
        self.apiEmbeddingSize = 0
        for featType in self.featType_enabled:
            for feat in self.featDict['mashup'][featType]: 
                if feat in self.reductFeats:
                    self.mashupEmbeddingSize += self.reductSize
                else:
                    self.mashupEmbeddingSize += featPosDict[feat][1] - featPosDict[feat][0]
    
        self.mashupEmbeddingSize+=self.reductSize

        for featType in self.featType_enabled:
            for feat in self.featDict['api'][featType]:
                if feat in self.reductFeats:
                    self.apiEmbeddingSize += self.reductSize
                else:
                    self.apiEmbeddingSize += featPosDict['c_' + feat][1] - featPosDict['c_' + feat][0]
      
        self.apiEmbeddingSize+=self.reductSize
    

        self.inputSize = self.mashupEmbeddingSize + self.apiEmbeddingSize +  self.apiEmbeddingSize
   
        self.hiddenUnits = [self.inputSize] + self.hiddenUnits + [1]
        self.dropout = nn.Dropout(self.dropoutRate)
        self.linears = nn.ModuleList([nn.Linear(self.hiddenUnits[i], self.hiddenUnits[i+1]) for i in range(len(self.hiddenUnits)-1)]).to(self.device)
        self.relus = nn.ModuleList([nn.PReLU() for i in range(len(self.hiddenUnits)-1)]).to(self.device) 
      
        for name, parameter in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(parameter, mean=0, std=self.initStd)
        self.sigmoid_predict = nn.Sigmoid()

    def forward(self, x):
      
        mashupEmbedding = []            
        candidateApiEmbedding = []      
        targetApiEmbeddings = []     
           
         
        for featType in self.featType_enabled:
            for feat in self.featDict['mashup'][featType]:
                originalFeatVector = x[:, self.featPosDict[feat][0]:self.featPosDict[feat][1]]   
                if feat in self.reductFeats:
                    mashupEmbedding.append(self.reductLinearsDict[feat](originalFeatVector))
                else:
                    mashupEmbedding.append(originalFeatVector)
      
        originalFeatVector = x[:, self.featPosDict['MashupDes'][0]:self.featPosDict['MashupDes'][1]]   
        mashupEmbedding.append(self.reductLinearsDict['MashupDes'](originalFeatVector))
            
        for featType in self.featType_enabled:
            for feat in self.featDict['api'][featType]:
                originalFeatVector = x[:, self.featPosDict['c_' + feat][0]:self.featPosDict['c_' + feat][1]]
                if feat in self.reductFeats:
                    candidateApiEmbedding.append(self.reductLinearsDict[feat](originalFeatVector))
                else:
                    candidateApiEmbedding.append(originalFeatVector)
       
        originalFeatVector = x[:, self.featPosDict['c_' + 'Des'][0]:self.featPosDict['c_' + 'Des'][1]]  
        candidateApiEmbedding.append(self.reductLinearsDict['Des'](originalFeatVector))

        # the_targetApiMask=x[:,self.featPosDict['targetApiMask'][0]:self.featPosDict['targetApiMask'][1]]
        
        for i in range(self.targetAPINum):
            # if the_targetApiMask[:,i]!=0:
            newApiEmbedding = []
            for featType in self.featType_enabled:
                for feat in self.featDict['api'][featType]:
                    originalFeatVector = x[:, self.featPosDict['t{}_'.format(i) + feat][0]:self.featPosDict['t{}_'.format(i) + feat][1]]
                    if feat in self.reductFeats:
                        newApiEmbedding.append(self.reductLinearsDict[feat](originalFeatVector)) 
                    else:
                        newApiEmbedding.append(originalFeatVector)
    
            originalFeatVector = x[:, self.featPosDict['t{}_'.format(i) + 'Des'][0]:self.featPosDict['t{}_'.format(i) + 'Des'][1]] 
            newApiEmbedding.append(self.reductLinearsDict['Des'](originalFeatVector))
            targetApiEmbeddings.append(newApiEmbedding)  


        the_T_Cscore=[]
        for i in range(self.targetAPINum):
            # if the_targetApiMask[:,i]!=0:
            for feat in self.featDict['api']['dense']:
                originalFeatVector = x[:, self.featPosDict['t{}_'.format(i) + feat][0]:self.featPosDict['t{}_'.format(i) + feat][1]]
                the_T_Cscore.append(originalFeatVector)

      
        mashupEmbedding = torch.cat(mashupEmbedding, dim=-1) 
        candidateApiEmbedding = torch.cat(candidateApiEmbedding, dim=-1)
        y=0
        for the_i in range(len(targetApiEmbeddings)):
            targetApiEmbedding = torch.cat(targetApiEmbeddings[the_i], dim=-1)
            mergeVector = torch.cat([mashupEmbedding, candidateApiEmbedding, targetApiEmbedding], dim=-1)
            for i in range(len(self.linears)): # dnn
                tmp = self.linears[i](mergeVector)
                if i < len(self.linears) - 1: 
                    tmp = self.relus[i](tmp)
                tmp = self.dropout(tmp)
                mergeVector=tmp                                                                             
            y+=mergeVector*the_T_Cscore[the_i]
        y= self.sigmoid_predict(y) 
        return y



def setup_seed(seed):
    torch.manual_seed(seed)                      
    torch.cuda.manual_seed_all(seed)            
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True    

learningRate = 0.0003   #
batchSize = 128
device = torch.device('cuda')
weightDecay = 0.001
epochs = 30


            
def calculateHR(model, k=10):
    # 计算命中率
    hitNum = 0
    hitCount = 0
    model.eval()
    with torch.no_grad(): 
        for index, (x, y) in enumerate(test_loader):
            for i in range(x.shape[0]):
                if y[i]:
                  
                    candidateApiId = x[i][featPosDict['c_ApiId'][0]:featPosDict['c_ApiId'][1]]
                    candidateAPiIdIndex = onehot2int(candidateApiId) 
                  
                    targetApiIdIndexList = []
                    for t in range(maxTargetAPINum):
                        id = x[i][featPosDict['t{}_ApiId'.format(t)][0]:featPosDict['t{}_ApiId'.format(t)][1]]
                        idIndex = onehot2int(id)
                        if idIndex is None:
                            break
                        if idIndex not in targetApiIdIndexList:
                            targetApiIdIndexList.append(idIndex)
                 
                    x_sort = torch.zeros([len(apiEncodingDict), x.shape[1]], dtype=torch.float)
                    y_sort = torch.zeros([len(apiEncodingDict), 1], dtype=torch.float)
                    j = 0
                    
                    for apiIdIndex in apiEncodingDict:
                        if apiIdIndex not in targetApiIdIndexList:
                            x_sort[j] = x[i]
                            if apiIdIndex == candidateAPiIdIndex:
                                y_sort[j] = 1
                            else:
                                x_sort[j][start:end] = apiEncodingDict[apiIdIndex]    
                            j += 1  
                    
                    y_sort_pre = model(x_sort.to(device))
                    y_true = y_sort.tolist()
                    
                    for i in range(len(y_true)):
                        y_true[i] = y_true[i][0]
                    y_pred = y_sort_pre.tolist()
                    for i in range(len(y_pred)):
                        y_pred[i] = y_pred[i][0]
                    hit = get_hit(y_true=y_true, y_pred=y_pred, k=k)
                    hitNum += hit
                    hitCount += 1
    hitRate = hitNum / hitCount
    print('HR@{} is {} = {} / {}'.format(k, hitRate, hitNum, hitCount))
    return hitRate


dataPath='./data/dataset_unfixed(maxapinumber=10).json'
# 独热编码！
featType_enabled = ['oneHot', 'multiHot']
featDict = {
    'mashup' : {
        'oneHot' : ['MashupId', 'MashupCategory', 'MashupType'],
        'multiHot' : ['MashupTags'],
        'text' : ['MashupDescription'],
    },
    'api' : {
        'oneHot' : ['ApiId', 'ApiCategory', 'ApiProvider', 'ApiSSLSupport', 'ApiAuthModel', 'ApiNonProprietary', 'ApiScope', 'ApiDeviceSpecific', 'ApiArchitecture', 'ApiUnofficial', 'ApiHypermedia', 'ApiRestrictedAccess'],
        'multiHot' : ['ApiTags'],
        'text' : ['ApiDescription'],
        'dense' : ['T_C_score'],  
    }
}


MASHUP_DES_PATH="./bert_mashup_des.json"
API_DES_PATH="./bert_api_des.json"
with open(MASHUP_DES_PATH, 'r') as fd:
        mashupRawData = fd.read()
        vec_mashup_des = json.loads(mashupRawData)

with open(API_DES_PATH, 'r') as fd:
    apiRawData = fd.read()
    vec_api_des  = json.loads(apiRawData)

data,featPosDict, maxTargetAPINum = dcr_preprocess(dataPath,vec_mashup_des,vec_api_des,featDict=featDict, featType_enabled=featType_enabled)    

seed = 2023528
testSize = 0.2
train, test = train_test_split(data, test_size=testSize, random_state=seed)
train_x = train[:,:-1]
train_y = train[:,[-1]]
test_x = test[:,:-1]
test_y = test[:,[-1]]

workerNum = 0
train_tensor_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(np.array(train_y)))
train_loader = DataLoader(train_tensor_data, shuffle=True, batch_size=batchSize, num_workers=workerNum)
test_tensor_data = TensorDataset(torch.from_numpy(np.array(test_x)), torch.from_numpy(np.array(test_y)))
test_loader = DataLoader(test_tensor_data, batch_size=batchSize, num_workers=workerNum)

apiEncodingDict = {}
start = featPosDict['c_' + featDict['api']['oneHot'][0]][0]
end = featPosDict['c_' + featDict['api']['multiHot'][-1]][1]
for index, (x, y) in enumerate(train_loader):
    for i in range(x.shape[0]):
        apiEncodingVector = x[i][start:end]
        apiId =  x[i][featPosDict['c_ApiId'][0]:featPosDict['c_ApiId'][1]]
        apiIdIndex = onehot2int(apiId)
        if apiIdIndex not in apiEncodingDict:
            apiEncodingDict[apiIdIndex] = apiEncodingVector

setup_seed(3088) 
top_k=10   
model_dnn = BaselineDeepNetwork(featDict=featDict, featPosDict=featPosDict, featType_enabled=featType_enabled, targetAPINum=maxTargetAPINum, device=device).to(device)
optimizer = torch.optim.Adam(params=model_dnn.parameters(), lr=learningRate, weight_decay=weightDecay)
loss_func = torch.nn.BCELoss().to(device)
model_dnn.flag_mask = True 
model_dnn.enableAA = True     
model_dnn.enableMA = True
epochs = 50

for epoch in range(epochs):
    model_dnn.train()
    total_loss, total_len = 0, 0
    for index, (x, y) in enumerate(train_loader):
        x, y = x.to(device).float(), y.to(device).float()
      
        y_pre = model_dnn(x)
      
        optimizer.zero_grad()
        loss = loss_func(y_pre, y)

        loss.backward()
        optimizer.step()
     
        total_loss += loss.item() * len(y)
        total_len += len(y)
    train_loss = total_loss / total_len
  
    model_dnn.eval()
    labels, predicts = [], []

    with torch.no_grad():
        for index, (x, y) in enumerate(test_loader):
            x, y = x.to(device).float(), y.to(device).float()
            y_pre = model_dnn(x)
            labels.extend(y.tolist())
            predicts.extend(y_pre.tolist())
    
    rmse = np.sqrt(mean_squared_error(np.array(labels), np.array(predicts)))
    auc = roc_auc_score(np.array(labels), np.array(predicts))
    log_loss1 = log_loss(np.array(labels), np.array(predicts))
    print("epoch {}, train loss is {}, val rmse is {}, val auc is {}, val log_loss is {}".format(epoch+1, train_loss, rmse, auc, log_loss1))
            
print('evaluating on hit rate...')
hr = calculateHR(model_dnn, k=10)
