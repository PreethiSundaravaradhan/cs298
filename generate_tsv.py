import json

#from svm.svm_roc import get_axes
import os


def tsv_from_json():
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'r', 'b', 'g', 'c', 'y', 'm', 'darkblue']
    users = ["preethi_s", "ash", "kar", "avinash", "ansu", "nive_n", "deepthi", "shreya_z", "kaushikc"]
    #users = ["preethi_s","ash_s", "kar_s", "avinash_s", "ansu_s", "nive_s","deepthi_s", "shreya_s", "kaushik_s"]
    files = open('/Users/preethi/Allclass/297/data1/pree2.json')
    f1 = json.load(files)
    #    x, y, z = get_axes(f1)
    print(x)
    print(y)
    print(z)


def parse_android_data():
    may_15_epoch = 1557883783000
    with open('/Users/preethi/Allclass/297/all_data/trialguesture-export.json') as f:
        android_data = json.load(f)
        uid = 1
        for user in android_data:
            dir1 = '/Users/preethi/Allclass/297/all_data/adrdata_2/'
            dir2 = '/Users/preethi/Allclass/297/all_data/adrdata_2_s/'
            dir3 = '/Users/preethi/Allclass/297/all_data/adrdata_2_s2/'
            if 'Test2' not in user and \
                            'Ravali' not in user and \
                            'ximin chen' not in user and \
                            'aakash_unique_1557960971140' not in user:
                #print(user)
                user_details = user.split("_")
                timestamp = int(user_details[-1])
                #print(timestamp)

                if timestamp < may_15_epoch:
                    user_dir = str(uid)
                    umame = user_dir
                    if user_details[1] == 'forged':
                        user_dir = dir2 + umame
                    else:
                        user_dir = dir1 + umame

                    if not os.path.exists(user_dir):
                        os.mkdir(user_dir)
                        if 'unique' in user:
                            print(umame)

                    for sample in android_data[user]:
                        try:
                            android_data[user][sample][0] = android_data[user][sample][1]
                        except Exception:
                            #print("unknown error")
                            pass
                        with open(user_dir+'/'+str(sample)+'.json', 'w') as f:
                            json.dump(android_data[user][sample], f)

                    if 'unique' in user:
                        uid += 1

                else:
                    user_dir = str(uid)
                    umame = user_dir
                    if user_details[1] == 'unique':
                        user_dir = dir1 + umame

                    if not os.path.exists(user_dir):
                        os.mkdir(user_dir)

                    if 'unique' in user:
                        for sample in android_data[user]:
                            try:
                                android_data[user][sample][0] = android_data[user][sample][1]
                            except Exception:
                                #print("unknown error")
                                pass
                            with open(user_dir + '/' + str(sample) + '.json', 'w') as f:
                                json.dump(android_data[user][sample], f)
                    else:
                        sample_id = 1
                        user_dir2 = dir2 + umame
                        if not os.path.exists(user_dir2):
                            os.mkdir(user_dir2)
                        for sample in android_data[user]:
                            try:
                                android_data[user][sample][0] = android_data[user][sample][1]
                            except Exception:
                                # print("unknown error")
                                pass
                            if sample_id > 10:
                                user_dir2 = dir3+umame
                                if not os.path.exists(user_dir2):
                                    os.mkdir(user_dir2)
                            with open(user_dir2 + '/' + str(sample) + '.json', 'w') as f:
                                json.dump(android_data[user][sample], f)
                            sample_id += 1

                    if 'unique' in user:
                        uid += 1

#tsv_from_json()

parse_android_data()