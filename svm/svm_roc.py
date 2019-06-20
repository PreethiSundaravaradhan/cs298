# shuffle and split training and test sets
from itertools import cycle

from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
import glob,os
import json
from pprint import pprint
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from sklearn import svm
import matplotlib.pyplot as plt
import statistics



def get_axes(js):
    s_x = []
    s_y = []
    s_z = []
    for r in js:
        try:
            s_x.append(float(r['xAcceleration']))
            s_y.append(float(r['yAcceleration']))
            s_z.append(float(r['zAcceleration']))
        except KeyError:
            pass
        except TypeError:
            pass

    return [s_x, s_y, s_z]


def list_files(user_dirs, my_user):
    files = []
    users = glob.glob(user_dirs)#"/Users/preethi/Allclass/297/data1/*")
    for user in users:
        if user.endswith('/'+my_user):
            user_data = glob.glob(user+"/*")
            i_1 = 0
            try:
                for file in user_data:
                    i_1 += 1
                    if os.path.isfile(file) and not file.endswith(".jpg"):
                        files.append(file)
            except Exception:
                print("something went wrong..")
    return files

plt.figure()


users = ["preethi_s", "ash", "kar", "avinash", "ansu", "nive_n", "deepthi", "shreya_z", "kaushikc"]
#users = ["preethi_s","ash_s", "kar_s", "avinash_s", "ansu_s", "nive_s","deepthi_s", "shreya_s", "kaushik_s"]
x_means = []
Y = []
y1 = []
for idx in range(9):
    files = list_files('/Users/preethi/Allclass/297/data1/*', users[idx])
    pnt = 0
    for f in files:
        if pnt < 20:
            file1 = open(f)
            f1 = json.load(file1)
            x, y, z = get_axes(f1)
            x_m, y_m, z_m = statistics.median(x), statistics.median(y), statistics.median(z)
            cur = []
            cur.append(x_m)
            cur.append(y_m)
            cur.append(z_m)
            x_means.append(cur)
            my_y = []
            y1.append(idx)
            for i in range(9):
                if i is idx:
                    my_y.append(1)
                else:
                    my_y.append(0)
            Y.append(my_y)

            pnt += 1
X = np.asarray(x_means)
Y = np.asarray(Y)

s= svm.LinearSVC()
s.fit(X, y1)
print(s.score(X,y1))

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.4,
                                                    random_state=0)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(9):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
# Plot all ROC curves
plt.figure()


colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'r', 'b', 'g', 'c', 'y', 'm', 'darkblue'])
for i, color in zip(range(9), colors):
    plt.plot(fpr[i], tpr[i], color=color,
             label='user {0} (area = {1:0.2f})'
             ''.format(i+1, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - Multiclass SVM on median unique signatures')
plt.legend(loc="lower right")
plt.show()