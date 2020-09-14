import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support as score

trainX = pd.read_csv('X_train.txt', delim_whitespace=True,header = None) 
trainY = pd.read_csv('Y_train.txt',delim_whitespace=True,header = None) 
testX = pd.read_csv('X_test.txt',delim_whitespace=True,header = None)
testY = pd.read_csv('Y_test.txt',delim_whitespace=True,header = None)

#lin_clf = svm.LinearSVC()
#lin_clf.fit(trainX, trainY)
#est = lin_clf.predict(testX)

clf = svm.SVC(decision_function_shape='ovo')
clf.fit(trainX, trainY)
est = clf.predict(testX)

df = pd.DataFrame(data = est,columns=["est"])
df['act']=testY

precision, recall, fscore, support = score(df['act'], df['est'])
df2 = pd.DataFrame()
df2['Precision']=precision
df2['recall']=recall
df2['fscore']=fscore
df2['support']=support

    
    

#plt.plot(trainX[[5]],trainY,'x')
#plt.show()
#trainX.plot()

