import json
from pprint import pprint
import os
from shutil import copyfile

import numpy as np
from PIL import Image
from sklearn.cross_validation import train_test_split


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

    return [s_x, s_y, s_z]


def convert_rgb(norm_vect):
    OldMin = 0
    OldMax = 1
    NewMin = 0
    NewMax = 255
    new_vector = []
    OldRange = (OldMax - OldMin)
    NewRange = (NewMax - NewMin)

    for axis in norm_vect:
        t_ax = []
        for val in axis:
            NewValue = int((((val - OldMin) * NewRange) / OldRange) + NewMin)
            t_ax.append(NewValue)
        new_vector.append(t_ax)
    return new_vector


def normalize(real_vector):
    avg = []
    min = 9999999
    max = 0
    new_vector = []
    for axis in real_vector:
        count = 0
        total = 0
        for val in axis:
            if val > max:
                max = val
            if val < min:
                min = val
            total += val
            count += 1
        cur_avg = total/count
        avg.append(cur_avg)
    for axis in real_vector:
        t_ax = []
        for val in axis:
            t_ax.append((val - min)/(max - min))
        new_vector.append(t_ax)
    return new_vector
#
#
#
#
#
# file1 = open('/Users/preethi/Allclass/297/data1/kar/kar5.json')
# f1 = json.load(file1)
#
#
# speeds = construct_speed(f1)
#
# norm_speeds = normalize(speeds)
# rgb_mappings = convert_rgb(norm_speeds)
# speed1 = np.transpose(rgb_mappings)
# speed1 = np.uint8(speed1)
# img = Image.fromarray(speed1, 'L')
# img = img.resize((100,300), Image.ANTIALIAS)
# img.save('test11.jpg')
#

import glob



def parse_name(file, image_dir):
    dirs = file.split("/")
    img_name = dirs[-2]
    dirs[-3] = image_dir
    path = ""
    for d in dirs[:-1]:
        path += d + "/"
    if not os.path.exists(path):
        os.mkdir(path)
    return path+img_name


def create_images(user_dirs, image_dir):
    users = glob.glob(user_dirs)#"/Users/preethi/Allclass/297/data1/*")
    for user in users:
        user_data = glob.glob(user+"/*")
        print("renaming..")
        i_1 = 0
        try:
            for file in user_data:
                i_1 += 1
                if os.path.isfile(file) and not file.endswith(".jpg"):
                    print(file)
                    with open(file) as f:
                        jsonx = json.load(f)
                        spd = construct_speed(jsonx)
                        norm_speeds = normalize(spd)
                        rgb_mappings = convert_rgb(norm_speeds)
                        speed1 = np.transpose(rgb_mappings)
                        speed1 = np.uint8(speed1)
                        img = Image.fromarray(speed1, 'L')
                        img = img.resize((100, 300), Image.ANTIALIAS)
                        img_name = parse_name(file, image_dir)
                        img.save(img_name+str(i_1)+'.jpg')
                else:
                    s_pattern = glob.glob(file+"/*")
                    for files in s_pattern:
                        i_1 += 1
                        if os.path.isfile(files) and not files.endswith(".jpg"):
                            with open(files) as f:
                                jsonx = json.load(f)
                                spd = construct_speed(jsonx)
                                norm_speeds = normalize(spd)
                                rgb_mappings = convert_rgb(norm_speeds)
                                speed1 = np.transpose(rgb_mappings)
                                speed1 = np.uint8(speed1)
                                img = Image.fromarray(speed1, 'L')
                                img = img.resize((100, 300), Image.ANTIALIAS)
                                img_name = parse_name(files, image_dir)
                                img.save(img_name + str(i_1) + 's.jpg')
        except Exception:
            print("something went wrong..")



        print(user_data)


def create_train_validations():
    img_dir = '/Users/preethi/Allclass/297/data_image/*/*'
    rasterList = glob.glob(os.path.join(img_dir))
    # Splitting data into training and testing

    train_samples, validation_samples = train_test_split(rasterList, test_size=0.2)
    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data

    train_dir = '/Users/preethi/Allclass/297/data_training/'
    valid_dir = '/Users/preethi/Allclass/297/data_validation/'


    for ts in train_samples:
        ts_path = ts.split("/")
        cur_dir = train_dir+ts_path[-2]
        if not os.path.exists(cur_dir):
            os.mkdir(cur_dir)
        copyfile(ts, train_dir+ts_path[-2]+"/"+ts_path[-1])


    for vs in validation_samples:
        ts_path = vs.split("/")
        cur_dir = valid_dir + ts_path[-2]
        if not os.path.exists(cur_dir):
            os.mkdir(cur_dir)
        copyfile(vs, valid_dir + ts_path[-2] + "/" + ts_path[-1])


if __name__ == "__main__":
    #create_images("/Users/preethi/Allclass/297/data_s/*", "data_image_s2")
    #create_images("/Users/preethi/Allclass/297/data_s/*", "data_image_s")
    create_train_validations()

