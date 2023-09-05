import torch.utils.data as data
import torch
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import random

class AgeGenderDataset(torch.utils.data.Dataset):
    def __init__(self, config, dataset=["IMDB", "WIKI", "Adience", "MORPH"], transform=None):

        self.root_path = config.age_gender_data_path

        self.image_paths = []
        self.labels = []
        
        self.suffix = ""

        self.sample_num = config.num_image // config.recognition_bz * config.age_gender_bz

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize([config.img_size, config.img_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        self.weight = dict()
        self.weight["Adience"] = 0.25
        self.weight["MORPH"] = 0.25
        self.weight["IMDB+WIKI"] = 0.5

        flag = True
        self.labels_1 = []
        while flag:

            if "Adience" in dataset and flag:
                # Adience
                self.Adience_path = os.path.join(self.root_path, "Adience")
                self.Adience_data = os.path.join(self.Adience_path, "data")
                self.Adience_label = os.path.join(self.Adience_path, ("label" + self.suffix + ".txt"))

                with open(self.Adience_label, "r") as f:
                    data = f.readlines()

                for i in range(1, len(data)):
                    line = data[i].split()
                    if len(line) > 0:

                        image_path = os.path.join(self.Adience_data, line[0])

                        if not (line[1] == "nan" or line[2] == "nan"):
                            label = [0.0, 0]  # age, gender
                            self.image_paths.append(image_path)
                            label[0] = np.asarray([float(line[1])])

                            if line[2] == "1.0":
                                label[1] = 1

                            self.labels_1.append(label)

                    if len(self.labels_1) >= self.sample_num * self.weight["Adience"]:
                        flag = False
                        break
        
        flag = True
        self.labels_2 = []
        while flag:

            if "MORPH" in dataset and flag:
                # MORPH
                self.MORPH_path = os.path.join(self.root_path, "MORPH")
                self.MORPH_data = os.path.join(self.MORPH_path, "data")
                self.MORPH_label = os.path.join(self.MORPH_path, ("label" + self.suffix + ".txt"))

                with open(self.MORPH_label, "r") as f:
                    data = f.readlines()

                for i in range(1, len(data)):
                    line = data[i].split()
                    if len(line) > 0:

                        image_path = os.path.join(self.MORPH_data, line[0])

                        if not (line[1] == "nan" or line[2] == "nan"):
                            label = [0.0, 0]  # age, gender
                            self.image_paths.append(image_path)
                            label[0] = np.asarray([float(line[1])])

                            if line[2] == "1.0":
                                label[1] = 1

                            self.labels_2.append(label)

                    if len(self.labels_2) >= self.sample_num * self.weight["MORPH"]:
                        flag = False
                        break

        self.labels.extend(self.labels_1)
        self.labels.extend(self.labels_2)

        flag = True
        while flag:

            if "WIKI" in dataset and flag:
                # WIKI
                self.WIKI_path = os.path.join(self.root_path, "WIKI")
                self.WIKI_data = os.path.join(self.WIKI_path, "data")
                self.WIKI_label = os.path.join(self.WIKI_path, ("label" + self.suffix + ".txt"))

                with open(self.WIKI_label, "r") as f:
                    data = f.readlines()

                for i in range(1, len(data)):
                    line = data[i].split()
                    if len(line) > 0:

                        image_path = os.path.join(self.WIKI_data, line[0])

                        if not (line[1] == "nan" or line[2] == "nan"):
                            label = [0.0, 0]  # age, gender
                            self.image_paths.append(image_path)
                            label[0] = np.asarray([float(line[1])])

                            if line[2] == "1.0":
                                label[1] = 1

                            self.labels.append(label)

                    if len(self.labels) == self.sample_num:
                        flag = False
                        break

            if "IMDB" in dataset and flag:
                # IMDB
                self.IMDB_path = os.path.join(self.root_path, "IMDB")
                self.IMDB_data = os.path.join(self.IMDB_path, "data")
                self.IMDB_label = os.path.join(self.IMDB_path, ("label" + self.suffix + ".txt"))
                with open(self.IMDB_label, "r") as f:
                    data = f.readlines()

                for i in range(1, len(data)):

                    line = data[i].split()
                    if len(line) > 0:

                        image_path = os.path.join(self.IMDB_data, line[0])

                        if not (line[1] == "nan" or line[2] == "nan"):
                            label = [0.0, 0]  # age, gender
                            self.image_paths.append(image_path)
                            label[0] = np.asarray([float(line[1])])

                            if line[2] == "1.0":
                                label[1] = 1

                            self.labels.append(label)

                    if len(self.labels) == self.sample_num:
                        flag = False
                        break


    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        img = torch.tensor(np.asarray(img))
        label = self.labels[idx]

        return img, label

    def __len__(self):
        return len(self.labels)

class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, config, choose, transform=None):

        self.image_names = []
        self.labels = []
        random.seed(config.seed)

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize([config.img_size, config.img_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        if choose == "train":
            self.CelebA_data = config.CelebA_train_data
            self.CelebA_label = config.CelebA_train_label

            self.sample_num = config.num_image // config.recognition_bz * config.CelebA_bz

            with open(self.CelebA_label, "r") as f:
                data = f.readlines()

            flag = True
            while flag:
                for i in range(1, len(data)):
                    r = random.random()
                    if r < 0.5:
                        line = data[i].split()

                        self.image_names.append(line[0])

                        label = [0 for j in range(40)]
                        for j in range(1, 41):
                            if line[j] == "1":
                                label[j-1] = 1

                        self.labels.append(label)

                        if len(self.labels) == self.sample_num:
                            flag = False
                            break
        elif choose == "val":
            self.CelebA_data = config.CelebA_val_data
            self.CelebA_label = config.CelebA_val_label

            with open(self.CelebA_label, "r") as f:
                data = f.readlines()

            for i in range(1, len(data)):
                line = data[i].split()

                self.image_names.append(line[0])

                label = [0 for j in range(40)]
                for j in range(1, 41):
                    if line[j] == "1":
                        label[j - 1] = 1

                self.labels.append(label)

        elif choose == "test":
            self.CelebA_data = config.CelebA_test_data
            self.CelebA_label = config.CelebA_test_label

            with open(self.CelebA_label, "r") as f:
                data = f.readlines()

            for i in range(1, len(data)):
                line = data[i].split()

                self.image_names.append(line[0])

                label = [0 for j in range(40)]
                for j in range(1, 41):
                    if line[j] == "1":
                        label[j - 1] = 1

                self.labels.append(label)

    def __getitem__(self, idx):

        img_path = os.path.join(self.CelebA_data, self.image_names[idx])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        img = torch.tensor(np.asarray(img))
        label = self.labels[idx]

        return img, label

    def __len__(self):
        return len(self.image_names)

class ExpressionDataset(torch.utils.data.Dataset):
    def __init__(self, config, transform=None):

        self.image_paths = []
        self.labels = []

        self.RAF_data = config.RAF_data
        self.RAF_label = config.RAF_label
        self.AffectNet_data = config.AffectNet_data
        self.AffectNet_label = config.AffectNet_label

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize([config.img_size, config.img_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        self.sample_num = config.num_image // config.recognition_bz * config.expression_bz

        self.weight = dict()
        self.weight["RAF"] = 0.5
        self.weight["AffectNet"] = 0.5

        with open(self.AffectNet_label, "r") as f:
            data = f.readlines()

        flag = True
        while flag:

            for i in range(1, len(data)):
                line = data[i].split()

                image_path = os.path.join(self.AffectNet_data, line[0])
                self.image_paths.append(image_path)
                self.labels.append(int(line[2]))
                if len(self.labels) >= self.sample_num * self.weight["AffectNet"]:
                    flag = False
                    break

        with open(self.RAF_label, "r") as f:
            data = f.readlines()

        flag = True
        while flag:
            for i in range(0, len(data)):
                line = data[i].strip('\n').split(" ")

                image_name = line[0]
                sample_temp = image_name.split("_")[0]

                if sample_temp == "train":
                    image_path = os.path.join(self.RAF_data, image_name)
                    self.image_paths.append(image_path)
                    self.labels.append(int(line[1]) - 1)

                if len(self.labels) == self.sample_num:
                    flag = False
                    break

        self.labels = np.asarray(self.labels)

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        img = torch.tensor(np.asarray(img))
        label = self.labels[idx]

        return img, label

    def __len__(self):
        return len(self.labels)

class RAFDataset(torch.utils.data.Dataset):
    def __init__(self, config, choose, transform=None):

        self.image_names = []
        self.labels = []

        self.RAF_data = config.RAF_data
        self.RAF_label = config.RAF_label

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize([config.img_size, config.img_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        self.train = True if choose == "train" else False

        if choose == "train":
            self.sample_num = config.num_image // config.recognition_bz * config.RAF_bz

            with open(self.RAF_label, "r") as f:
                data = f.readlines()

            flag = True
            while flag:
                for i in range(0, len(data)):
                    line = data[i].strip('\n').split(" ")

                    image_name = line[0]
                    sample_temp = image_name.split("_")[0]

                    if self.train and sample_temp == "train":
                        self.image_names.append(image_name)
                        self.labels.append(int(line[1]) - 1)

                    if len(self.labels) == self.sample_num:
                        flag = False
                        break

        elif choose == "test":

            with open(self.RAF_label, "r") as f:
                data = f.readlines()

            for i in range(0, len(data)):
                line = data[i].strip('\n').split(" ")

                image_name = line[0]
                sample_temp = image_name.split("_")[0]

                if not self.train and sample_temp == "test":
                    self.image_names.append(image_name)
                    self.labels.append(int(line[1]) - 1)

        self.labels = np.asarray(self.labels)

    def __getitem__(self, idx):

        img_path = os.path.join(self.RAF_data, self.image_names[idx])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        img = torch.tensor(np.asarray(img))
        label = torch.tensor(self.labels[idx])

        return img, label

    def __len__(self):
        return len(self.image_names)


class FGnetDataset(torch.utils.data.Dataset):
    def __init__(self, config, choose="all", id=0, transform=None):

        self.FGnet_data = config.FGnet_data
        self.FGnet_label = config.FGnet_label
        self.leave_out_file_name = ""

        self.image_names = []
        self.labels = []

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize([config.img_size, config.img_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        with open(self.FGnet_label, "r") as f:
            data = f.readlines()

        cut = len(data) // 10 * 9
        if choose == "remove_one":
            for i in range(1, len(data)):
                line = data[i].split()
                if i == id:
                    self.leave_out_file = os.path.join(self.FGnet_data, line[0])
                else:
                    self.image_names.append(line[0])
                    label = np.asarray([float(line[1])])
                    self.labels.append(label)
        elif choose == "all":
            for i in range(1, len(data)):
                line = data[i].split()
                self.image_names.append(line[0])
                label = np.asarray([float(line[1])])
                self.labels.append(label)
        elif choose == "9_fold":
            for i in range(1, cut):
                line = data[i].split()
                self.image_names.append(line[0])
                label = np.asarray([float(line[1])])
                self.labels.append(label)
        elif choose == "1_fold":
            for i in range(cut, len(data)):
                line = data[i].split()
                self.image_names.append(line[0])
                label = np.asarray([float(line[1])])
                self.labels.append(label)

    def __getitem__(self, idx):

        img_path = os.path.join(self.FGnet_data, self.image_names[idx])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        img = torch.tensor(np.asarray(img))
        label = torch.tensor(self.labels[idx])

        return img, label

    def __len__(self):
        return len(self.image_names)

    def get_leave_out_file_name(self):
        return self.leave_out_file_name
        
class LAPDataset(torch.utils.data.Dataset):
    def __init__(self, config, choose="", transform=None):

        if choose == "train":
            self.LAP_data = config.LAP_train_data
            self.LAP_label = config.LAP_train_label
        elif choose == "test":
            self.LAP_data = config.LAP_test_data
            self.LAP_label = config.LAP_test_label

        self.image_names = []
        self.labels = []

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize([config.img_size, config.img_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])


        with open(self.LAP_label, "r") as f:
            data = f.readlines()
        

        for i in range(0, len(data)):
            line = data[i].split(";")
    
            self.image_names.append(line[0])
    
            label = [np.asarray([float(line[1])]), np.asarray([float(line[2])])]
    
            self.labels.append(label)

    def __getitem__(self, idx):

        img_path = os.path.join(self.LAP_data, self.image_names[idx])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        img = torch.tensor(np.asarray(img))
        label = self.labels[idx]

        return img, label

    def __len__(self):
        return len(self.image_names)
