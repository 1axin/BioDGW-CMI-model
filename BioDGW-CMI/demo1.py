# 作者:     wxf

# 开发时间: 2022/6/10 16:47

import warnings
import seaborn as sns
warnings.filterwarnings('ignore')

import csv

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from numpy import *
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
import numpy as np
from sklearn.metrics import roc_curve, auc

def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:  # 把每个rna疾病对加入OriginalData，注意表头
        SaveList.append(row)
    return

def storFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

#*****************************标签******************************************
#***************************划分训练集************************************
SampleFeature = []
ReadMyCsv(SampleFeature, '9905_SampleFeature_256.csv')
SampleFeature = np.array(SampleFeature)

print('Start training the model.')
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 10000)
i = 0
num=0
cv = StratifiedKFold(n_splits=5)
SampleFeature = np.array(SampleFeature)
SampleLabel = np.array(SampleLabel)
permutation = np.random.permutation(SampleLabel.shape[0])
SampleFeature = SampleFeature[permutation, :]
SampleLabel = SampleLabel[permutation]

mean_precision = 0.0
mean_recall = np.linspace(0,1,10000)
mean_average_precision = []

Y_test = []
Y_pre = []
Y_all = []
result = []
for train, test in cv.split(SampleFeature, SampleLabel):


    model = lgb.LGBMClassifier
    predicted = model.fit(SampleFeature[train], SampleLabel[train]).predict_proba(SampleFeature[test])
    fpr, tpr, thresholds = roc_curve(SampleLabel[test], predicted[:, 1])
    predicted1 = model.predict(SampleFeature[test])
    num = num + 1
    precision, recall, _ = precision_recall_curve(SampleLabel[test], predicted[:, 1])
    average_precision = average_precision_score(SampleLabel[test], predicted[:, 1])
    mean_average_precision.append(average_precision)
    mean_precision += interp(mean_recall, recall, precision)

    print("==================", num, "fold", "==================")
    print('Test accuracy: ', accuracy_score(SampleLabel[test], predicted1))
    print(classification_report(SampleLabel[test], predicted1, digits=4))
    print(confusion_matrix(SampleLabel[test], predicted1))
    result.append(classification_report(SampleLabel[test], predicted1, digits=4))
    Y_test.extend(SampleLabel[test])
    Y_pre.extend(predicted1)
    Y_test1 = []
    Y_pre1 = []
    Y_test1.extend(SampleLabel[test])
    Y_pre1.extend(predicted[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1.5, alpha=1,
             label='ROC fold %d (AUC = %0.4f)' % (i, roc_auc))
    i += 1


# storFile(result,'result.csv')
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',label=r'Mean ROC (AUC = %0.4f $\pm$ %0.4f)' % (mean_auc, std_auc), lw=1.5, alpha=1)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.5,label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
# plt.savefig('ROC-5fold-test.tif')
plt.show()





