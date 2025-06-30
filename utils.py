import json
import numpy as np
import pandas as pd
import time
import random
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, LabelBinarizer
from multiprocessing import Queue, Process


def mpWorker(func, commonArgs, taskQueue:Queue, resQueue:Queue):
    while True:
        try:
            jobNum, args = taskQueue.get(timeout=1) # 
        except Exception as e: 
            break
        try:
            if commonArgs is None:
                res = func(*args)
            else:
                res = func(*(args+commonArgs))
            resQueue.put([jobNum, res])
        except:
            pass

def multiProcessFramework(func, argList, processNum:int, flag_show_progress=True, flag_merge_result=False, commonArgs:tuple=None, timeout=5):
    
    resDict = {}
   
    taskQueue = Queue()
    jobNum = 0
    
    for args in argList:
        taskQueue.put([jobNum, args])
        jobNum += 1

    totalNum = len(argList)
    resQueue = Queue()
    processPool = []
 

    for i in range(processNum): 
        p = Process(target=mpWorker, args=(func, commonArgs, taskQueue, resQueue))
        processPool.append(p)

    for p in processPool:
        p.start()

   
    while not taskQueue.qsize() == 0:
        if flag_show_progress:
            num = taskQueue.qsize()
            print(' {}/{}  {}%'.format(totalNum - num, totalNum, int((totalNum-num)/totalNum*100)), end = '\r')
            time.sleep(0.01)
        if not resQueue.qsize() == 0:
            jobNum, res  = resQueue.get(timeout=timeout)
            resDict[jobNum] = res
        else:
            if not flag_show_progress:
                time.sleep(0.01)
    if flag_show_progress:
        print('{}/{}  {}%'.format(totalNum, totalNum, 100), end = '\r')
        print('')


    while True:
        try:
            jobNum, res = resQueue.get(timeout=timeout)
            resDict[jobNum] = res
        except:
            break


    for p in processPool:
        p.join()

    resList = []
    for i in range(totalNum):
        try:
            if flag_merge_result:
                resList += resDict[i]
            else:
                resList.append(resDict[i])
        except:
            print('[error] lost: ', i)
    return resList

def encodeData(rawData, encoderDict, maxTargetAPINum, featDict, featType_enabled, featPosDict):
 
    newData = []
    for featType in featType_enabled:
        for feat in featDict['mashup'][featType]:
            newData += list(encoderDict[feat].transform([rawData['mashup'][feat]])[0])
    for feat in featDict['mashup']['dense']:
        newData += [rawData['mashup'][feat]]
    for featType in featType_enabled:
        for feat in featDict['api'][featType]:
            newData += list(encoderDict[feat].transform([rawData['candidate API'][feat]])[0])
    targetAPINum = len(rawData['target APIs'])
    if targetAPINum > maxTargetAPINum: targetAPINum = maxTargetAPINum
    for i in range(targetAPINum):
        for featType in featType_enabled:
            for feat in featDict['api'][featType]:
                newData += list(encoderDict[feat].transform([rawData['target APIs'][i][feat]])[0])
        for feat in featDict['api']['dense']:
            newData += [rawData['target APIs'][i][feat]]

    for i in range(targetAPINum, maxTargetAPINum):
        for featType in featType_enabled:
            for feat in featDict['api'][featType]:
                featLen = featPosDict['t{}_'.format(i) + feat][1] - featPosDict['t{}_'.format(i) + feat][0]
                newData += [0] * featLen
        for feat in featDict['api']['dense']:
            newData += [0]

    newData += [1] * targetAPINum
    newData += [0] * (maxTargetAPINum - targetAPINum)
    # invoke
    newData += [rawData['invoke']]
    print(len(newData))
    return np.array(newData, dtype=np.float32)

def encodeData_mp(rawDataList, encoderDict, maxTargetAPINum, featDict, featType_enabled, featPosDict):
    resList = []
    for rawData in rawDataList:
        print(type(rawData))
        resList.append(encodeData(rawData, encoderDict, maxTargetAPINum, featDict, featType_enabled, featPosDict))
    return resList

def dcr_preprocess(dataPath,vec_mashup_des,vec_api_des,featDict, featType_enabled = ['oneHot', 'multiHot']):
    # 读取数据集
    with open(dataPath, 'r') as fd:
        rawDataset = json.load(fd)
  
    # 统计最大目标API数量
    maxTargetAPINum = 0
    for rd in rawDataset:
        if len(rd['target APIs']) > maxTargetAPINum:
            maxTargetAPINum = len(rd['target APIs'])
    # 特征编码
    # 1、对特征进行汇总
    encodeFeatTypeList = {'oneHot', 'multiHot', 'text'} # 需要编码的特征类型列表
    sumdata = {} 
    for rawData in rawDataset:
        # mashup
        obj = 'mashup'
        for featType in encodeFeatTypeList:
            for feat in featDict[obj][featType]:
                value = rawData['mashup'][feat]
                if feat not in sumdata:
                    sumdata[feat] = []
                sumdata[feat].append(value)
        # api
        # 每一个feat都是一个list 把目标api与候选api的特征提取出来 不做区分
        obj = 'api'
        ## candidate api
        for featType in encodeFeatTypeList:
            for feat in featDict[obj][featType]:
                value=rawData["candidate API"][feat]
                if feat not in sumdata:
                    sumdata[feat] = []
                sumdata[feat].append(value)
        ## target apis
        targetApiNum = len(rawData['target APIs'])

        if targetApiNum > maxTargetAPINum: targetApiNum = maxTargetAPINum
     
        for featType in encodeFeatTypeList:
            for feat in featDict[obj][featType]:
                for i in range(targetApiNum):
                    value=rawData["target APIs"][i][feat]
                    if feat not in sumdata:
                        sumdata[feat] = []
                    sumdata[feat].append(value)

        # invoke
        feat = 'invoke'
        value = rawData[feat]
        if feat not in sumdata:
            sumdata[feat] = []
        sumdata[feat].append(value)

    # 2、训练编码器
    # fit：训练模型，将特征转换为二进制独热编码
    # 语法：LabelBinarizer() 将多标签转换为二标签；MultiLabelBinarizer() list数据的二标签(意思一致)
    # 这也太多了，eg：有5000 mashupid 则 编码数有5000位
    # 注：invoke没有fit
    encoderDict = {} # 编码器字典
    for feat in featDict['mashup']['oneHot']:
        encoderDict[feat] = LabelBinarizer()
        encoderDict[feat].fit(sumdata[feat])
    for feat in featDict['api']['oneHot']:
        encoderDict[feat] = LabelBinarizer()
        encoderDict[feat].fit(sumdata[feat])
    for feat in featDict['mashup']['multiHot']:
        encoderDict[feat] = MultiLabelBinarizer()
        encoderDict[feat].fit(sumdata[feat])
    for feat in featDict['api']['multiHot']:
        encoderDict[feat] = MultiLabelBinarizer()
        encoderDict[feat].fit(sumdata[feat])
    # 文本编码(暂时不处理)

    # 构建数据集
    # 构造feat_sizes特征查找表
    featPosDict = {} # 记录每个特征对应向量的起始与结束位置([startPos, endPos])，此后可通过startPos:endPos访问此特征 
    featPos = 0    # 辅助变量
    rd = rawDataset[0]  #所有样本的数据属性都一样，因此可以任选一个样本
  
    # mashup
    # 语法：transform这里返回的是二维nddary [0].shape[0]返回列数
    for featType in featType_enabled:
        for feat in featDict['mashup'][featType]:
            featLen = encoderDict[feat].transform([rd['mashup'][feat]])[0].shape[0]
            featPosDict[feat] = [featPos, featPos + featLen]
            featPos += featLen
    # 添加mashup的描述信息：
    featLen=len(vec_mashup_des[rd['mashup']['MashupName']])
    featPosDict['MashupDes']= [featPos, featPos + featLen]
    featPos+=featLen

    # 候选API 利用前缀c_,t_对候选api与目标api做了区分
    for featType in featType_enabled:
        for feat in featDict['api'][featType]:
            featLen = encoderDict[feat].transform([rd['candidate API'][feat]])[0].shape[0]
            featPosDict['c_' + feat] = [featPos, featPos + featLen]
            featPos += featLen
    # 添加c_api的描述信息：
    featLen=len(vec_api_des[rd['candidate API']['ApiName']])
    featPosDict['c_Des']= [featPos, featPos + featLen]
    featPos+=featLen

    # 目标API
    for i in range(maxTargetAPINum):
        for featType in featType_enabled:
            for feat in featDict['api'][featType]:
                featLen = encoderDict[feat].transform([rd['target APIs'][0][feat]])[0].shape[0]
                featPosDict['t{}_'.format(i) + feat] = [featPos, featPos + featLen]
                featPos += featLen
        for feat in featDict['api']['dense']:
            featPosDict['t{}_'.format(i) + feat] = [featPos, featPos + 1]
            featPos += 1

        featLen=len(vec_api_des[rd['target APIs'][0]['ApiName']])
        featPosDict['t{}_'.format(i) +'Des'] = [featPos, featPos + featLen]
        featPos+=featLen

    # 这些暂时用不上
    # # targetApiMask 这个参数用来 构建每条数据对应的目标API mask，用来标记哪些目标API是真实数据(1)，哪些目标API是填充的(0)
    featLen = maxTargetAPINum
    featPosDict['targetApiMask'] = [featPos, featPos + featLen]
    featPos += featLen

    # 3、编码并构建数据集  train
    argList_train = []
    inputLen = 100
    for i in range(0, len(rawDataset), inputLen):
        if i + inputLen < len(rawDataset):
            argList_train.append((rawDataset[i:i+inputLen],))
        else:
            argList_train.append((rawDataset[i:len(rawDataset)],))

    # # 参数
    commonArgs=(encoderDict, maxTargetAPINum, featDict, featType_enabled, featPosDict)

    # encoderDict(dict) one-hot编码 (fit之后)
    # maxTargetAPINum(int) 目标api的最大数
    # featDict(dict) 
    # featType_enabled
    # featPosDict(dict) 存放起始位置 注：相关系数的存放与 targetApiMask
    # encodeData(rawDataset[0], encoderDict, maxTargetAPINum, featDict, featType_enabled, featPosDict)
    # 返回的数据list 

    data = multiProcessFramework(func=encodeData_mp_s1, argList=argList_train, processNum=24, flag_show_progress=True, flag_merge_result=True, commonArgs=(encoderDict, maxTargetAPINum, featDict, featType_enabled, featPosDict,vec_mashup_des,vec_api_des))
    data = np.array(data)
    return data,featPosDict, maxTargetAPINum

#---------------------------------------------------------------------------------------------------------------
class myMultiLabelEncoder():
    def __init__(self) -> None:
        self.labelList = []  # 标签最大的种类数 所有标签的列表
        self.maxLabelNum = 0 # 一组多标签内最大的标签数量

    def fit(self, y):
        self.labelList = []
        self.maxLabelNum  = 0
        for multiLabels in y:
            if len(multiLabels) > self.maxLabelNum:
                self.maxLabelNum = len(multiLabels)
            for label in multiLabels:
                if label not in self.labelList:
                    self.labelList.append(label)

    def transform(self, y):
        result = []
        for i in range(len(y)):
            newLabels = []
            for j in range(self.maxLabelNum):
                if j < len(y[i]):
                    # 语法：.index 返回对应的索引位置
                    # newLabels 存放标签在 labelList中的索引位置加1 
                    newLabels.append(self.labelList.index(y[i][j]) + 1)
                else:
                    newLabels.append(0)
            result.append(newLabels)
        return np.array(result)

    def fit_transform(self, y):
        self.fit(y)
        result = self.transform(y)
        return result
    
def dcr_preprocess_se(dataPath, fixedTargetAPINum, featDict, featType_enabled = ['oneHot', 'multiHot']):
    '''deepctr专用版

        input : 
            dataPath : str
            fixedTargetAPINum : int : 要读取的数据集中的固定目标API数量,为0时会做特殊对齐处理

        return :
            data
            multiHotLenDict
            maxTargetAPINum : 对齐后的最大(固定)目标API数量
    '''
    # 读取数据集
    with open(dataPath, 'r') as fd:
        rawDataset = json.load(fd)

    if fixedTargetAPINum == 0:
        # 统计数据集中最大的目标API数量
        maxTargetAPINum = 0
        for rawData in rawDataset:
            if len(rawData['target APIs']) > maxTargetAPINum:
                maxTargetAPINum = len(rawData['target APIs'])
    else:
        maxTargetAPINum = fixedTargetAPINum

    data = {} 
    for rawData in rawDataset:
        # mashup
        obj = 'mashup'
        for featType in featDict[obj]:
            for feat in featDict[obj][featType]:
                value = rawData['mashup'][feat]
                if feat not in data:
                    data[feat] = []
                data[feat].append(value)
        # api
        obj = 'api'
        ## candidate api
        for featType in featDict[obj]:
            for feat in featDict[obj][featType]:
                # 这个if目的是什么 -2023.7.24
                if 'CC' in feat:
                    continue
                value = rawData["candidate API"][feat]
                featName = 'c_' + feat
                if featName not in data:
                    data[featName] = []
                data[featName].append(value)
        ## target apis
        if fixedTargetAPINum:
            for i in range(fixedTargetAPINum): # 这里不写死，且目标云API数量不固定的话，后面会错位
                for featType in featDict[obj]:
                    for feat in featDict[obj][featType]:
                        value = rawData["target APIs"][i][feat]
                        featName = 't{}_'.format(i) + feat
                        if featName not in data:
                            data[featName] = []
                        data[featName].append(value)
        else:
            # 对于 非固定数目 目标api 
            # <maxTargetAPINum的部分 增加至maxTargetAPINum的部分，将各个数据进行处理
            for i in range(maxTargetAPINum):
                for featType in featDict[obj]:
                    for feat in featDict[obj][featType]:
                        if len(rawData['target APIs']) > i:
                            value = rawData["target APIs"][i][feat]
                        else: 
                            value = rawData["target APIs"][0][feat]
                            if type(value) == int:
                                value = -1
                            elif type(value) == str:
                                value = 'None'
                            elif type(value) == list:
                                value = ['None']
                            elif type(value) == float:
                                value = 0
                            else:
                                print('undefined value type in dcr_preprocess_se')
                        featName = 't{}_'.format(i) + feat
                        if featName not in data:
                            data[featName] = []
                        data[featName].append(value) 

        # invoke
        feat = 'invoke'
        value = rawData[feat]
        if feat not in data:
            data[feat] = []
        data[feat].append(value)

    # 特征编码(oneHot) 使用的是标签编码
    oneHotFeatures = []
    oneHotFeatures += featDict['mashup']['oneHot']
    for feat in featDict['api']['oneHot']:
        oneHotFeatures.append('c_' + feat)
    if fixedTargetAPINum:
        for i in range(fixedTargetAPINum):
            for feat in featDict['api']['oneHot']:
                oneHotFeatures.append('t{}_{}'.format(i, feat))
    else:
        for i in range(maxTargetAPINum):
            for feat in featDict['api']['oneHot']:
                oneHotFeatures.append('t{}_{}'.format(i, feat))
    # labelEncoder(标签编码) 将数据转换为数值--与onehot编码不同，并不全是 0,1
    # eg：有n个取值，则编码为：0 1 2 ...n-1
    for feat in oneHotFeatures:
        lbe = LabelEncoder()
        # fit_transform 可以简单的看为 fit + transform  包含了训练与转换
        data[feat] = lbe.fit_transform(data[feat])

    # 特征编码(multiHot)
    multiHotFeatures = []
    multiHotLenDict = {}
    multiHotFeatures += featDict['mashup']['multiHot']
    for feat in featDict['api']['multiHot']:
        multiHotFeatures.append('c_' + feat)
    if fixedTargetAPINum:
        for i in range(fixedTargetAPINum):
            for feat in featDict['api']['multiHot']:
                multiHotFeatures.append('t{}_{}'.format(i, feat))
    else:
        for i in range(maxTargetAPINum):
            for feat in featDict['api']['multiHot']:
                multiHotFeatures.append('t{}_{}'.format(i, feat))
    for feat in multiHotFeatures:
        mlbe = myMultiLabelEncoder()
        data[feat] = mlbe.fit_transform(data[feat])
        multiHotLenDict[feat] = [len(mlbe.labelList), mlbe.maxLabelNum]

    return data, multiHotLenDict, maxTargetAPINum

# #------------------------------------------------
def my_shuffle(data, random_state=None):
    originState = random.getstate()
    # 语法:使用相同的seed 则每次生成的随机数都相同
    random.seed(random_state)

    newDataIndex = []
    # 生成索引,data数据中每个特征对应的每个value值的长度相等 (样本的长度)
    for key in data:
        dataIndex = list(range(len(data[key])))
        break
    # 相当于将dataIndex的顺序打乱
    while len(dataIndex):
        i = random.randint(0, len(dataIndex)-1)
        newDataIndex.append(dataIndex.pop(i))

    newData = {}
    for i in newDataIndex:
        for key in data:
            if key not in newData:
                newData[key] = [data[key][i]]
            else:
                newData[key].append(data[key][i])

    random.setstate(originState)
    return newData


def my_train_test_split(data, test_size, random_state=None, flag_shuffle=True):
    # 每次生成的随机数都一样
    originState = random.getstate()
    train = []
    test = []

    if flag_shuffle:
        # data中的顺序被打乱
        data = my_shuffle(data, random_state=random_state)

    # 随机生成训练集与测试集索引
    # 虽然data的数据打乱了，但是依然随机生成了训练集和测试集的索引
    random.seed(random_state)
    for key in data:
        trainIndexList = list(range(len(data[key])))
        count = int(test_size * len(data[key]))
        break
    testIndexList = []
    while count > 0:
        i = random.randint(0, len(trainIndexList)-1)
        testIndexList.append(trainIndexList.pop(i))
        count -= 1

    # 生成训练集与测试集
    train = {}
    test = {}
    for i in trainIndexList:
        for key in data:
            if key not in train:
                train[key] = [data[key][i]]
            else:
                train[key].append(data[key][i])
    for i in testIndexList:
        for key in data:
            if key not in test:
                test[key] = [data[key][i]]
            else:
                test[key].append(data[key][i])

    # 转成np.array  
    # 将value中的list转换为array 行数为 样本数
    for key in data:
        train[key] = np.array(train[key])
        test[key] = np.array(test[key])

    random.setstate(originState)
    return train, test


# 根据预测输出(是否可以理解为是rating) 进行排序，排序top-k  查看top-k中是否有命中项目
# 由于在样本处理中，每一个样本只有一个候选api，所以最多只能有一个命中
def get_hit(y_true, y_pred, k):
    '''计算此列表中是否有命中项目(hit@k)

        Args:
            y_true : list : 真实值列表
            y_pred : list : 预测值列表
            k : int : @k
        
        Return:
            bool : 0/1
    '''
    df = pd.DataFrame({"y_pred":y_pred, "y_true":y_true})
    df = df.sort_values(by="y_pred", ascending=False) 
    df = df.iloc[0:k, :]  
    df = df.to_numpy()
    for i in range(k):
        if df[i][1] == 1:
            return 1
    return 0
'''
    在变长阶数的场景中，由于只有一个mashup真实调用过的API可以作为候选API，y_true中最多只有一个1
    这会导致topk中能够命中的个数最大为1，进而recall@k与hit@k等价，且precision@k的最大值被限制在1/k
    因此最多计算hit@k/HR@k即可。即变长阶数场景下只适合计算排序指标中的命中率指标，且命中率指标一定会偏低，因为可供命中的项目只有一个
'''

def get_ndcg(y_true, y_pred, k):
    '''计算此列表的ndcg@k

        Args: 
            y_true : list : 真实值列表
            y_pred : list : 预测值列表
            k : int : @k
    '''
    def _get_dcg(y_true, y_pred, k):
        df = pd.DataFrame({"y_pred":y_pred, "y_true":y_true})
        df = df.sort_values(by="y_pred", ascending=False)  # 对y_pred进行降序排列，越排在前面的，越接近label=1
        df = df.iloc[0:k, :]  # 取前K个
        dcg = 0
        i = 1
        for y_true_i in df["y_true"]:
            dcg += (2 ** y_true_i - 1) / np.log2(1 + i)
            i += 1
        return dcg

    dcg = _get_dcg(y_true, y_pred, k)
    idcg = _get_dcg(y_true, y_true, k)
    ndcg = dcg / idcg
    return ndcg

def get_precision(y_true, y_pred, k):
    '''计算此列表的准确率(precision@k)

        Args:
            y_true : list : 真实值列表
            y_pred : list : 预测值列表
            k : int : @k
        
        Return:
            float
    '''
    df = pd.DataFrame({"y_pred":y_pred, "y_true":y_true})
    df = df.sort_values(by="y_pred", ascending=False) 
    df = df.iloc[0:k, :]  
    df = df.to_numpy()
    count0 = 0 # 分子是topk中命中的个数
    count1 = k # 分母是k
    for i in range(k):
        if df[i][1]:
            count0 += 1
    return count0 / count1

def get_recall(y_true, y_pred, k):
    '''计算此列表的召回率(recall@k)

        Args:
            y_true : list : 真实值列表
            y_pred : list : 预测值列表
            k : int : @k
        
        Return:
            float
    '''
    df = pd.DataFrame({"y_pred":y_pred, "y_true":y_true})
    df = df.sort_values(by="y_pred", ascending=False) 
    df = df.iloc[0:k, :]  
    df = df.to_numpy()
    count0 = 0 # 分子是topk中命中的个数
    count1 = 0 # 分母是y_true中1的个数
    for i in range(len(y_true)):
        if y_true[i]:
            count1 += 1
    for i in range(k):
        if df[i][1]:
            count0 += 1
    return count0 / count1

def onehot2int(oneHotEncodingVector):
    '''
    将oneHot编码向量转换回int
    onehot编码: 只有一个位置为1,其他位置均为0 其索引为值为1的位置,从后往前
       
    '''
    length = len(oneHotEncodingVector)
    for i in range(length):
        if oneHotEncodingVector[i]:
            return length - i                   #---------index 从1开始？
    return None
