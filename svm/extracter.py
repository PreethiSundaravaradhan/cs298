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

from plot_confusion_matrix import process_confusion_mat


def construct_speed3(js):
    s_x = []
    i1 = 0
    for r in js:
        try:
            s_x.append(float(js[r]['x']) - i1)
            i1 = float(js[r]['x'])
        except KeyError:
            pass
        except TypeError:
            pass

    return s_x

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

def get_axes_android(js):
    s_x = []
    s_y = []
    s_z = []
    for r in js:
        try:
            s_x.append(float(r['x']))
            s_y.append(float(r['y']))
            s_z.append(float(r['z']))
        except KeyError:
            pass
        except TypeError:
            pass

    return [s_x, s_y, s_z]


def construct_speed(js):
    s_x = []
    s_y = []
    s_z = []
    i1 = 0
    i2 = 0
    i3 = 0
    for r in js:
        s_x.append(float(r['xAcceleration']) - i1)
        s_y.append(float(r['yAcceleration']) - i2)
        s_z.append(float(r['zAcceleration']) - i3)
        i1 = float(r['xAcceleration'])
        i2 = float(r['yAcceleration'])
        i3 = float(r['zAcceleration'])

    #s_x.extend(s_y)
    #s_x.extend(s_z)

    return (s_x, s_y, s_z)


file1 = open('/Users/preethi/Allclass/297/data1/ash/ash1.json')

f1 = json.load(file1)

y = []
speed1_x, speed1_y, speed1_z = construct_speed(f1)

'''
speed2 = construct_speed(f2)
speed3 = construct_speed3(f3)

for s1 in speed1:
    y.append(0)
for s1 in speed2:
    y.append(0)
for s1 in speed3:
    y.append(1)

speed1.extend(speed2)
speed1.extend(speed3)

clf = svm.
X = np.array(speed1)
my_y = np.array(y)
X = X.reshape(-1,1)
clf.fit(X, my_y)
pprint(cross_validate(clf, X, my_y))
'''

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

def ploting(x, y, class_num):
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'r', 'b', 'g', 'c', 'y', 'm', 'darkblue']
    users = ['intruder', 'user']
    plt.plot(x, y,'ro', color=colors[class_num])


def run_pca():
    x_val = []
    y_val = []
    pca = PCA(n_components=3)
    # users = ["preethi_s","ash", "kar", "avinash", "ansu", "nive_n","deepthi", "shreya_z", "kaushikc"]
    users = ["preethi_s", "ash_s", "kar_s", "avinash_s", "ansu_s", "nive_s", "deepthi_s", "shreya_s", "kaushik_s"]
    for idx in range(9):
        files = list_files('/Users/preethi/Allclass/297/data_s/*', users[idx])
        for f in files:
            file1 = open(f)
            f1 = json.load(file1)
            x, y, z = get_axes(f1)
            l_v = []
            l_v.append(x)
            l_v.append(y)
            l_v.append(z)
            x_val.append(l_v)
            y_val.append(idx)
    X = np.asarray(x_val)
    Y = np.asarray(y_val)
    pca.fit(X, Y)
    print(pca.score(X, Y))


principal_comps_train = []
principal_comps_test = []
pca_y_train = []
pca_y_test = []

def plot_pca():
    #users = ["preethi_s", "kar", "avinash", "ansu", "nive_n","deepthi", "shreya_z", "kaushikc","sushmita_m", "ash"]
    users = ["preethi_s", "kar_s", "avinash_s", "ansu_s", "nive_s","deepthi_s", "shreya_s", "kaushik_s","sushmita_s", "ash_s"]
    for idx in range(9):
        files = list_files('/Users/preethi/Allclass/297/data_s/*', users[idx])
        sample = 0
        for f in files:
            sample += 1
            file1 = open(f)
            f1 = json.load(file1)
            x, y, z = get_axes(f1)
            pca = PCA(n_components=3)
            compone = pca.fit([x, y, z])
            speeds = pca.explained_variance_
            if(sample < 15):
                principal_comps_train.append([speeds[0], speeds[1]])
                pca_y_train.append(idx)
                ploting(speeds[0], speeds[1], idx)
                print(pca.explained_variance_)
            elif sample<20:
                principal_comps_test.append([speeds[0], speeds[1]])
                pca_y_test.append(idx)
                ploting(speeds[0], speeds[1], idx)
                print(pca.explained_variance_)
    plt.xlabel("Principal Component 1 (eigenvalue[0])")
    plt.ylabel("Principal Component 2 (eigenvalue[1])")
    plt.show()


svm2 = svm.SVC(kernel='poly', degree=3)


def plotting_stats():
    user_means = []
    ax = plt.subplot(111)
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'r', 'b', 'g', 'c', 'y', 'm', 'darkblue']
    users = ["preethi_s","ash", "kar", "avinash", "ansu", "nive_n","deepthi", "shreya_z", "kaushikc"]
    #users = ["preethi_s","ash_s", "kar_s", "avinash_s", "ansu_s", "nive_s","deepthi_s", "shreya_s", "kaushik_s"]
    for idx in range(9):
        #user_means_curr = []
        files = list_files('/Users/preethi/Allclass/297/data1/*', users[idx])
        pnt = 0
        for f in files:
            if pnt < 20:
                file1 = open(f)
                f1 = json.load(file1)
                x, y, z = get_axes(f1)
                (x_m,y_m,z_m) = (statistics.mean(x), statistics.mean(y),statistics.mean(z))
                plt.plot(pnt, x_m, 'ro', color=colors[idx])
                #user_means_curr.append(x_m,y_m,z_m)
                print(x_m,y_m,z_m,idx)
                pnt +=1
            else:
                break

        #user_means.append(user_means_curr)
    plt.ylabel('Mean')
    plt.xlabel('Sample id')
    plt.show()

def construct_svm():
    users = ["preethi_s", "kar_s", "avinash_s", "ansu_s", "nive_s", "deepthi_s", "shreya_s", "kaushik_s", "sushmita_s",
             "ash_s"]
    for idx in range(9):
        files = list_files('/Users/preethi/Allclass/297/data_s/*', users[idx])
        sample = 0
        for f in files:
            sample += 1
            file1 = open(f)
            f1 = json.load(file1)
            x, y, z = get_axes(f1)
            x = []
            x.extend(x)
            x.extend(y)
            x.extend(z)
            print(len(x))
            if (sample < 15):
                principal_comps_train.append(x[:50])
                pca_y_train.append(idx)
            elif sample < 20:
                principal_comps_test.append(x[:50])
                pca_y_test.append(idx)


def construct_svm_android():

    s2_users = ["1","4","8","9","10","11","12","14","17","19","20","21","22","23","24","28","29","30","31","34"]
    for idx in range(len(s2_users)):
        files = list_files('/Users/preethi/Allclass/297/all_data/adrdata_2_s/*', s2_users[idx])
        sample = 0
        for f in files:
            sample += 1
            file1 = open(f)
            f1 = json.load(file1)
            x, y, z = get_axes_android(f1)
            x = []
            x.extend(x)
            x.extend(y)
            x.extend(z)
            print(len(x))
            if len(x) <= 150:
                continue
            if (sample < 12):
                principal_comps_train.append(x[:150])
                pca_y_train.append(idx)
            elif sample < 18:
                principal_comps_test.append(x[:150])
                pca_y_test.append(idx)


def fit_svm(x,y):
    svm2.fit(x, y)
    scores = svm2.score(x, y)
    print(scores)
    return scores


def run_svm():
    users = ["preethi_s", "ash", "kar", "avinash", "ansu", "nive_n", "deepthi", "shreya_z", "kaushikc"]
    #users = ["preethi_s","ash_s", "kar_s", "avinash_s", "ansu_s", "nive_s","deepthi_s", "shreya_s", "kaushik_s"]
    x_means = []
    Y = []
    for idx in range(len(users)):
        # user_means_curr = []
        files = list_files('/Users/preethi/Allclass/297/data1/*', users[idx])
        pnt = 0
        for f in files:
            if pnt < 20:
                file1 = open(f)
                f1 = json.load(file1)
                x, y, z = get_axes(f1)
                x_m, y_m, z_m = statistics.mean(x), statistics.mean(y), statistics.mean(z)
                cur = []
                cur.append(x_m)
                cur.append(y_m)
                cur.append(z_m)
                x_means.append(cur)
                Y.append(idx)

                pnt += 1
    X = np.asarray(x_means)
    Y = np.asarray(Y)
    svm2.fit(X,Y)
    print(svm2.score(X, Y))


def find_magnitude():
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'r', 'b', 'g', 'c', 'y', 'm', 'darkblue']
    #users = ["preethi_s", "ash", "kar", "avinash", "ansu", "nive_n", "deepthi", "shreya_z", "kaushikc"]
    users = ["preethi_s","ash_s", "kar_s", "avinash_s", "ansu_s", "nive_s","deepthi_s", "shreya_s", "kaushik_s"]
    for idx in range(9):
        # user_means_curr = []
        user_means = []
        files = list_files('/Users/preethi/Allclass/297/data_s/*', users[idx])
        pnt = 0
        for f in files:
            if pnt < 20:
                file1 = open(f)
                f1 = json.load(file1)
                x, y, z = get_axes(f1)
                x = np.square(x)
                y = np.square(y)
                z = np.square(z)
                s = x + y + z
                user_means.append(np.sqrt(np.sum(s))/x.size)
                pnt += 1
        um = np.array(user_means)
        sums = np.sum(um, axis=0)
        print(sums)

# shreya : [1.35324831e+01 4.24435468e-01 9.53888320e-32]
# kaushik: [7.72291799e+00 4.02926737e-01 9.23248345e-32]
# av1: [1.73868352e+01 2.13308587e+00 1.62279126e-31]
# av2: [1.91828586e+01 1.72145168e+00 1.07821973e-31],
# [1.76906765e+01 2.23122621e+00 8.63143560e-31],
# [1.49041939e+01 2.59471812e+00 5.36216826e-31]
#run_pca()



#plot_pca()
construct_svm_android()
print("SVM scores = ")
y_pred = fit_svm(principal_comps_train, pca_y_train)
process_confusion_mat(principal_comps_train, pca_y_train, principal_comps_test, pca_y_test,  np.asarray(list(range(20))))


#plotting_stats()
#plot_pca()

print("\n")
#find_magnitude()