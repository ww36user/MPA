import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import ImageFile, Image
from PIL import ImageEnhance
import re
from ast import literal_eval
from io_utils import parse_args_test

ImageFile.LOAD_TRUNCATED_IMAGES = True
identity = lambda x:x
transformtypedict=dict(Brightness=ImageEnhance.Brightness, Contrast=ImageEnhance.Contrast, Sharpness=ImageEnhance.Sharpness, Color=ImageEnhance.Color)

class ImageJitter(object):
    def __init__(self, transformdict):
        self.transforms = [(transformtypedict[k], transformdict[k]) for k in transformdict]
        
    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms))
        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha*(randtensor[i]*2.0 -1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')
        return out


class SetDataset:
    def __init__(self, data_path, num_class, batch_size, transform):
        self.sub_meta = {}
        self.data_path = data_path
        self.num_class = num_class
        self.cl_list = range(self.num_class)
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        d = ImageFolder(self.data_path)
        for i, (data, label) in enumerate(d):
            self.sub_meta[label].append(data)
        self.d = d
        self.sub_dataloader = []
        sub_data_loader_params = dict(batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=0,  # use main thread only or may receive multiple batches
                                      pin_memory=False)
        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform=transform)
            self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))

        self.classes_datasets = self.d.classes
        self.cars_dict = {1: 'AM General Hummer SUV 2000', 2: 'Acura RL Sedan 2012', 3: 'Acura TL Sedan 2012', 4: 'Acura TL Type-S 2008', 5: 'Acura TSX Sedan 2012', 6: 'Acura Integra Type R 2001', 7: 'Acura ZDX Hatchback 2012', 8: 'Aston Martin V8 Vantage Convertible 2012', 9: 'Aston Martin V8 Vantage Coupe 2012', 10: 'Aston Martin Virage Convertible 2012', 11: 'Aston Martin Virage Coupe 2012', 12: 'Audi RS 4 Convertible 2008', 13: 'Audi A5 Coupe 2012', 14: 'Audi TTS Coupe 2012', 15: 'Audi R8 Coupe 2012', 16: 'Audi V8 Sedan 1994', 17: 'Audi 100 Sedan 1994', 18: 'Audi 100 Wagon 1994', 19: 'Audi TT Hatchback 2011', 20: 'Audi S6 Sedan 2011', 21: 'Audi S5 Convertible 2012', 22: 'Audi S5 Coupe 2012', 23: 'Audi S4 Sedan 2012', 24: 'Audi S4 Sedan 2007', 25: 'Audi TT RS Coupe 2012', 26: 'BMW ActiveHybrid 5 Sedan 2012', 27: 'BMW 1 Series Convertible 2012', 28: 'BMW 1 Series Coupe 2012', 29: 'BMW 3 Series Sedan 2012', 30: 'BMW 3 Series Wagon 2012', 31: 'BMW 6 Series Convertible 2007', 32: 'BMW X5 SUV 2007', 33: 'BMW X6 SUV 2012', 34: 'BMW M3 Coupe 2012', 35: 'BMW M5 Sedan 2010', 36: 'BMW M6 Convertible 2010', 37: 'BMW X3 SUV 2012', 38: 'BMW Z4 Convertible 2012', 39: 'Bentley Continental Supersports Conv. Convertible 2012', 40: 'Bentley Arnage Sedan 2009', 41: 'Bentley Mulsanne Sedan 2011', 42: 'Bentley Continental GT Coupe 2012', 43: 'Bentley Continental GT Coupe 2007', 44: 'Bentley Continental Flying Spur Sedan 2007', 45: 'Bugatti Veyron 16.4 Convertible 2009', 46: 'Bugatti Veyron 16.4 Coupe 2009', 47: 'Buick Regal GS 2012', 48: 'Buick Rainier SUV 2007', 49: 'Buick Verano Sedan 2012', 50: 'Buick Enclave SUV 2012', 51: 'Cadillac CTS-V Sedan 2012', 52: 'Cadillac SRX SUV 2012', 53: 'Cadillac Escalade EXT Crew Cab 2007', 54: 'Chevrolet Silverado 1500 Hybrid Crew Cab 2012', 55: 'Chevrolet Corvette Convertible 2012', 56: 'Chevrolet Corvette ZR1 2012', 57: 'Chevrolet Corvette Ron Fellows Edition Z06 2007', 58: 'Chevrolet Traverse SUV 2012', 59: 'Chevrolet Camaro Convertible 2012', 60: 'Chevrolet HHR SS 2010', 61: 'Chevrolet Impala Sedan 2007', 62: 'Chevrolet Tahoe Hybrid SUV 2012', 63: 'Chevrolet Sonic Sedan 2012', 64: 'Chevrolet Express Cargo Van 2007', 65: 'Chevrolet Avalanche Crew Cab 2012', 66: 'Chevrolet Cobalt SS 2010', 67: 'Chevrolet Malibu Hybrid Sedan 2010', 68: 'Chevrolet TrailBlazer SS 2009', 69: 'Chevrolet Silverado 2500HD Regular Cab 2012', 70: 'Chevrolet Silverado 1500 Classic Extended Cab 2007', 71: 'Chevrolet Express Van 2007', 72: 'Chevrolet Monte Carlo Coupe 2007', 73: 'Chevrolet Malibu Sedan 2007', 74: 'Chevrolet Silverado 1500 Extended Cab 2012', 75: 'Chevrolet Silverado 1500 Regular Cab 2012', 76: 'Chrysler Aspen SUV 2009', 77: 'Chrysler Sebring Convertible 2010', 78: 'Chrysler Town and Country Minivan 2012', 79: 'Chrysler 300 SRT-8 2010', 80: 'Chrysler Crossfire Convertible 2008', 81: 'Chrysler PT Cruiser Convertible 2008', 82: 'Daewoo Nubira Wagon 2002', 83: 'Dodge Caliber Wagon 2012', 84: 'Dodge Caliber Wagon 2007', 85: 'Dodge Caravan Minivan 1997', 86: 'Dodge Ram Pickup 3500 Crew Cab 2010', 87: 'Dodge Ram Pickup 3500 Quad Cab 2009', 88: 'Dodge Sprinter Cargo Van 2009', 89: 'Dodge Journey SUV 2012', 90: 'Dodge Dakota Crew Cab 2010', 91: 'Dodge Dakota Club Cab 2007', 92: 'Dodge Magnum Wagon 2008', 93: 'Dodge Challenger SRT8 2011', 94: 'Dodge Durango SUV 2012', 95: 'Dodge Durango SUV 2007', 96: 'Dodge Charger Sedan 2012', 97: 'Dodge Charger SRT-8 2009', 98: 'Eagle Talon Hatchback 1998', 99: 'FIAT 500 Abarth 2012', 100: 'FIAT 500 Convertible 2012', 101: 'Ferrari FF Coupe 2012', 102: 'Ferrari California Convertible 2012', 103: 'Ferrari 458 Italia Convertible 2012', 104: 'Ferrari 458 Italia Coupe 2012', 105: 'Fisker Karma Sedan 2012', 106: 'Ford F-450 Super Duty Crew Cab 2012', 107: 'Ford Mustang Convertible 2007', 108: 'Ford Freestar Minivan 2007', 109: 'Ford Expedition EL SUV 2009', 110: 'Ford Edge SUV 2012', 111: 'Ford Ranger SuperCab 2011', 112: 'Ford GT Coupe 2006', 113: 'Ford F-150 Regular Cab 2012', 114: 'Ford F-150 Regular Cab 2007', 115: 'Ford Focus Sedan 2007', 116: 'Ford E-Series Wagon Van 2012', 117: 'Ford Fiesta Sedan 2012', 118: 'GMC Terrain SUV 2012', 119: 'GMC Savana Van 2012', 120: 'GMC Yukon Hybrid SUV 2012', 121: 'GMC Acadia SUV 2012', 122: 'GMC Canyon Extended Cab 2012', 123: 'Geo Metro Convertible 1993', 124: 'HUMMER H3T Crew Cab 2010', 125: 'HUMMER H2 SUT Crew Cab 2009', 126: 'Honda Odyssey Minivan 2012', 127: 'Honda Odyssey Minivan 2007', 128: 'Honda Accord Coupe 2012', 129: 'Honda Accord Sedan 2012', 130: 'Hyundai Veloster Hatchback 2012', 131: 'Hyundai Santa Fe SUV 2012', 132: 'Hyundai Tucson SUV 2012', 133: 'Hyundai Veracruz SUV 2012', 134: 'Hyundai Sonata Hybrid Sedan 2012', 135: 'Hyundai Elantra Sedan 2007', 136: 'Hyundai Accent Sedan 2012', 137: 'Hyundai Genesis Sedan 2012', 138: 'Hyundai Sonata Sedan 2012', 139: 'Hyundai Elantra Touring Hatchback 2012', 140: 'Hyundai Azera Sedan 2012', 141: 'Infiniti G Coupe IPL 2012', 142: 'Infiniti QX56 SUV 2011', 143: 'Isuzu Ascender SUV 2008', 144: 'Jaguar XK XKR 2012', 145: 'Jeep Patriot SUV 2012', 146: 'Jeep Wrangler SUV 2012', 147: 'Jeep Liberty SUV 2012', 148: 'Jeep Grand Cherokee SUV 2012', 149: 'Jeep Compass SUV 2012', 150: 'Lamborghini Reventon Coupe 2008', 151: 'Lamborghini Aventador Coupe 2012', 152: 'Lamborghini Gallardo LP 570-4 Superleggera 2012', 153: 'Lamborghini Diablo Coupe 2001', 154: 'Land Rover Range Rover SUV 2012', 155: 'Land Rover LR2 SUV 2012', 156: 'Lincoln Town Car Sedan 2011', 157: 'MINI Cooper Roadster Convertible 2012', 158: 'Maybach Landaulet Convertible 2012', 159: 'Mazda Tribute SUV 2011', 160: 'McLaren MP4-12C Coupe 2012', 161: 'Mercedes-Benz 300-Class Convertible 1993', 162: 'Mercedes-Benz C-Class Sedan 2012', 163: 'Mercedes-Benz SL-Class Coupe 2009', 164: 'Mercedes-Benz E-Class Sedan 2012', 165: 'Mercedes-Benz S-Class Sedan 2012', 166: 'Mercedes-Benz Sprinter Van 2012', 167: 'Mitsubishi Lancer Sedan 2012', 168: 'Nissan Leaf Hatchback 2012', 169: 'Nissan NV Passenger Van 2012', 170: 'Nissan Juke Hatchback 2012', 171: 'Nissan 240SX Coupe 1998', 172: 'Plymouth Neon Coupe 1999', 173: 'Porsche Panamera Sedan 2012', 174: 'Ram C/V Cargo Van Minivan 2012', 175: 'Rolls-Royce Phantom Drophead Coupe Convertible 2012', 176: 'Rolls-Royce Ghost Sedan 2012', 177: 'Rolls-Royce Phantom Sedan 2012', 178: 'Scion xD Hatchback 2012', 179: 'Spyker C8 Convertible 2009', 180: 'Spyker C8 Coupe 2009', 181: 'Suzuki Aerio Sedan 2007', 182: 'Suzuki Kizashi Sedan 2012', 183: 'Suzuki SX4 Hatchback 2012', 184: 'Suzuki SX4 Sedan 2012', 185: 'Tesla Model S Sedan 2012', 186: 'Toyota Sequoia SUV 2012', 187: 'Toyota Camry Sedan 2012', 188: 'Toyota Corolla Sedan 2012', 189: 'Toyota 4Runner SUV 2012', 190: 'Volkswagen Golf Hatchback 2012', 191: 'Volkswagen Golf Hatchback 1991', 192: 'Volkswagen Beetle Hatchback 2012', 193: 'Volvo C30 Hatchback 2012', 194: 'Volvo 240 Sedan 1993', 195: 'Volvo XC90 SUV 2007', 196: 'smart fortwo Convertible 2012'}


    def __getitem__(self, i):
        data_list = next(iter(self.sub_dataloader[i]))
        data = data_list[0]
        label = data_list[1]
        single_label = label[0].item()

        class_dataset = self.classes_datasets[single_label] # 类名
        # 构建返回的list
        list_ = []

        # print(self.classes_datasets)
        # euroSAT 数据集
        dir_path = "/data_disk/ww/project/LDP"
        model = "/gpt3.5"
        # 1. euroSAT
        if 'AnnualCrop' in self.classes_datasets:
            target_dataset = "/euroSAT.txt"
            file_path = dir_path+model+target_dataset;

            with open(file_path, 'r') as file:
                content = file.read()

            # 用正则匹配所有列表
            matches = re.findall(r'euroSAT_des\d+\s*=\s*(\[.*?\])', content, re.DOTALL)

            # 安全地将字符串转换为列表
            euroSAT_des1 = literal_eval(matches[0])
            euroSAT_des2 = literal_eval(matches[1])
            euroSAT_des3 = literal_eval(matches[2])
            euroSAT_des4 = literal_eval(matches[3])
            euroSAT_des5 = literal_eval(matches[4])
            # euroSAT 数据集
            classes_euroSAT_name = [
    "This is a satellite image of AnnualCrop, showing expansive agricultural fields used for seasonal crops, often displaying varying colors and patterns due to different stages of growth and harvest.",
    "This is a satellite image of Forest, capturing dense areas of tree cover, with a canopy that appears as rich green expanses or seasonal color variations, depending on the type of forest.",
    "This is a satellite image of HerbaceousVegetation, depicting open areas dominated by grasses and shrubs, presenting a patchwork of green, yellow, or brown tones depending on the season and health of the vegetation.",
    "This is a satellite image of Highway, highlighting major roads and transportation arteries with visible lanes, intersections, and vehicles that may appear as tiny dots or lines in high-traffic areas.",
    "This is a satellite image of Industrial, showing clusters of factories, warehouses, or infrastructure, often characterized by large buildings, machinery, and expansive paved or cleared areas.",
    "This is a satellite image of Pasture, illustrating fields used for grazing livestock, often seen as open green spaces or divided parcels, sometimes bordered by fences or natural boundaries.",
    "This is a satellite image of PermanentCrop, focusing on fields used for perennial crops like orchards or vineyards, usually arranged in organized rows or plots with distinctive planting patterns.",
    "This is a satellite image of Residential, showcasing areas with clusters of homes, streets, and local amenities, often arranged in neighborhoods or city blocks with visible infrastructure.",
    "This is a satellite image of River, depicting a flowing water body that winds through the landscape, with visible banks, bends, and sometimes tributaries feeding into it.",
    "This is a satellite image of SeaLake, capturing large bodies of water with clearly defined boundaries, often showing variations in water color, nearby coastlines, and surrounding terrain."
]
            class_dataset2 = classes_euroSAT_name[single_label]  # 类名扩展
            classes_des1 = euroSAT_des1[single_label]  # 类名描述1
            classes_des2 = euroSAT_des2[single_label]  # 类名描述2
            classes_des3 = euroSAT_des3[single_label]  # 类名描述3
            classes_des4 = euroSAT_des4[single_label]  # 类名描述4
            classes_des5 = euroSAT_des5[single_label]  # 类名描述5

            list_.append(data)
            list_.append(label)  # 类标签
            list_.append(class_dataset)  # 类名1
            list_.append(class_dataset2)  # 描述类名2
            list_.append(classes_des1)  # 描述1
            list_.append(classes_des2)  # 描述2
            list_.append(classes_des3)  # 描述3
            list_.append(classes_des4)  # 描述4
            list_.append(classes_des5)  # 描述5
        # 2. CropDiesese 数据集
        elif 'Apple___Apple_scab' in self.classes_datasets:
            # CropDiesese 数据集
            cropDiesese_name1 = [
                'Apple: Apple scab', 'Apple: Black rot', 'Apple: Cedar apple rust', 'Apple: healthy',
                'Blueberry: healthy', 'Cherry (including sour): Powdery mildew', 'Cherry (including sour): healthy',
                'Corn (maize): Cercospora leaf spot Gray leaf spot', 'Corn (maize): Common rust',
                'Corn (maize): Northern Leaf Blight',
                'Corn (maize): healthy', 'Grape: Black rot', 'Grape: Esca (Black Measles)',
                'Grape: Leaf blight (Isariopsis Leaf Spot)',
                'Grape: healthy', 'Orange: Haunglongbing (Citrus greening)', 'Peach: Bacterial spot', 'Peach: healthy',
                'Pepper, bell: Bacterial spot', 'Pepper, bell: healthy', 'Potato: Early blight', 'Potato: Late blight',
                'Potato: healthy', 'Raspberry: healthy', 'Soybean: healthy', 'Squash: Powdery mildew',
                'Strawberry: Leaf scorch',
                'Strawberry: healthy', 'Tomato: Bacterial spot', 'Tomato: Early blight', 'Tomato: Late blight',
                'Tomato: Leaf Mold',
                'Tomato: Septoria leaf spot', 'Tomato: Spider mites Two-spotted spider mite', 'Tomato: Target Spot',
                'Tomato: Tomato Yellow Leaf Curl Virus', 'Tomato: Tomato mosaic virus', 'Tomato: healthy'
            ]
            cropDiesese_name2 = [
                'A picture of Apple with Apple scab', 'A picture of Apple with Black rot',
                'A picture of Apple with Cedar apple rust', 'A picture of healthy Apple',
                'A picture of healthy Blueberry', 'A picture of Cherry (including sour) with Powdery mildew',
                'A picture of healthy Cherry (including sour)',
                'A picture of Corn (maize) with Cercospora leaf spot Gray leaf spot',
                'A picture of Corn (maize) with Common rust', 'A picture of Corn (maize) with Northern Leaf Blight',
                'A picture of healthy Corn (maize)', 'A picture of Grape with Black rot',
                'A picture of Grape with Esca (Black Measles)',
                'A picture of Grape with Leaf blight (Isariopsis Leaf Spot)',
                'A picture of healthy Grape', 'A picture of Orange with Haunglongbing (Citrus greening)',
                'A picture of Peach with Bacterial spot', 'A picture of healthy Peach',
                'A picture of Pepper, bell with Bacterial spot', 'A picture of healthy Pepper, bell',
                'A picture of Potato with Early blight', 'A picture of Potato with Late blight',
                'A picture of healthy Potato', 'A picture of healthy Raspberry', 'A picture of healthy Soybean',
                'A picture of Squash with Powdery mildew',
                'A picture of Strawberry with Leaf scorch', 'A picture of healthy Strawberry',
                'A picture of Tomato with Bacterial spot', 'A picture of Tomato with Early blight',
                'A picture of Tomato with Late blight', 'A picture of Tomato with Leaf Mold',
                'A picture of Tomato with Septoria leaf spot',
                'A picture of Tomato with Spider mites (Two-spotted spider mite)',
                'A picture of Tomato with Target Spot', 'A picture of Tomato with Tomato Yellow Leaf Curl Virus',
                'A picture of Tomato with Tomato mosaic virus', 'A picture of healthy Tomato'
            ]
            target_dataset = "/CropDisease.txt"
            file_path = dir_path + model + target_dataset;

            with open(file_path, 'r') as file:
                content = file.read()

            # 用正则匹配所有列表
            matches = re.findall(r'cropDisease_des\d+\s*=\s*(\[.*?\])', content, re.DOTALL)

            # 安全地将字符串转换为列表
            cropDisease_des1 = literal_eval(matches[0])
            cropDisease_des2 = literal_eval(matches[1])
            cropDisease_des3 = literal_eval(matches[2])
            cropDisease_des4 = literal_eval(matches[3])
            cropDisease_des5 = literal_eval(matches[4])

            class_dataset = cropDiesese_name1[single_label]  # 类名扩展
            class_dataset2 = cropDiesese_name2[single_label]  # 类名扩展
            classes_des1 = cropDisease_des1[single_label]  # 类名描述1
            classes_des2 = cropDisease_des2[single_label]  # 类名描述2
            classes_des3 = cropDisease_des3[single_label]  # 类名描述3
            classes_des4 = cropDisease_des4[single_label]  # 类名描述4
            classes_des5 = cropDisease_des5[single_label]  # 类名描述5

            list_.append(data)
            list_.append(label)  # 类标签
            list_.append(class_dataset)  # 类名1
            list_.append(class_dataset2)  # 描述类名2
            list_.append(classes_des1)  # 描述1
            list_.append(classes_des2)  # 描述2
            list_.append(classes_des3)  # 描述3
            list_.append(classes_des4)  # 描述4
            list_.append(classes_des5)  # 描述5
        # 3. ISIC 数据集
        elif 'AKIEC' in self.classes_datasets:
            # ISIC 数据集
            ISIC_name1 = ['Actinic Keratosis', 'Basal Cell Carcinoma', 'Benign Keratosis', 'Dermatofibroma', 'Melanoma', 'Nevus', 'Vascular Skin Lesion']
            ISIC_name2 = [
    'This is Actinic Keratosis (AKIEC) skin disease.',
    'This is Basal Cell Carcinoma (BCC) skin disease.',
    'This is Benign Keratosis (BKL) skin disease.',
    'This is Dermatofibroma (DF) skin disease.',
    'This is Melanoma (MEL) skin disease.',
    'This is Nevus (NV) skin disease.',
    'This is Vascular Skin Lesion (VASC) skin disease.'
]

            target_dataset = "/ISIC.txt"
            file_path = dir_path + model + target_dataset;

            with open(file_path, 'r') as file:
                content = file.read()

            # 用正则匹配所有列表
            matches = re.findall(r'ISIC_des\d+\s*=\s*(\[.*?\])', content, re.DOTALL)

            # 安全地将字符串转换为列表
            ISIC_des1 = literal_eval(matches[0])
            ISIC_des2 = literal_eval(matches[1])
            ISIC_des3 = literal_eval(matches[2])
            ISIC_des4 = literal_eval(matches[3])
            ISIC_des5 = literal_eval(matches[4])
            class_dataset = ISIC_name1[single_label]  # 类名扩展
            class_dataset2 = ISIC_name2[single_label]  # 类名扩展
            classes_des1 = ISIC_des1[single_label]  # 类名描述1
            classes_des2 = ISIC_des2[single_label]  # 类名描述2
            classes_des3 = ISIC_des3[single_label]  # 类名描述3
            classes_des4 = ISIC_des4[single_label]  # 类名描述4
            classes_des5 = ISIC_des5[single_label]  # 类名描述5

            list_.append(data)
            list_.append(label)  # 类标签
            list_.append(class_dataset)  # 类名1
            list_.append(class_dataset2)  # 描述类名2
            list_.append(classes_des1)  # 描述1
            list_.append(classes_des2)  # 描述2
            list_.append(classes_des3)  # 描述3
            list_.append(classes_des4)  # 描述4
            list_.append(classes_des5)  # 描述5
        # 4. cars 数据集
        elif '99' in self.classes_datasets:
            # cars 数据集
            classes_datasets = [int(cls) for cls in self.classes_datasets]
            mapped_values = [self.cars_dict[cls] for cls in classes_datasets if cls in self.cars_dict] # 所有类别的名称

            cars_name1 = mapped_values #['Ferrari 458 Italia Convertible 2012', 'Ford Mustang Convertible 2007', 'Aston Martin Virage Coupe 2012', 'Ford Ranger SuperCab 2011', 'Ford Focus Sedan 2007', 'GMC Savana Van 2012', 'Geo Metro Convertible 1993', 'Honda Odyssey Minivan 2007', 'Hyundai Santa Fe SUV 2012', 'Hyundai Elantra Sedan 2007', 'Hyundai Elantra Touring Hatchback 2012', 'Isuzu Ascender SUV 2008', 'Jeep Liberty SUV 2012', 'Audi R8 Coupe 2012', 'Lamborghini Aventador Coupe 2012', 'Land Rover LR2 SUV 2012', 'Mazda Tribute SUV 2011', 'Mercedes-Benz SL-Class Coupe 2009', 'Mitsubishi Lancer Sedan 2012', 'Nissan 240SX Coupe 1998', 'Rolls-Royce Phantom Drophead Coupe Convertible 2012', 'Spyker C8 Convertible 2009', 'Suzuki SX4 Hatchback 2012', 'Toyota Camry Sedan 2012', 'Audi TT Hatchback 2011', 'Volkswagen Golf Hatchback 1991', 'Volvo XC90 SUV 2007', 'Audi S4 Sedan 2012', 'BMW 1 Series Convertible 2012', 'Acura TL Sedan 2012', 'BMW 6 Series Convertible 2007', 'BMW M5 Sedan 2010', 'Bentley Continental Supersports Conv. Convertible 2012', 'Bentley Continental GT Coupe 2007', 'Buick Regal GS 2012', 'Cadillac CTS-V Sedan 2012', 'Chevrolet Corvette Convertible 2012', 'Chevrolet Camaro Convertible 2012', 'Chevrolet Sonic Sedan 2012', 'Chevrolet Malibu Hybrid Sedan 2010', 'Acura ZDX Hatchback 2012', 'Chevrolet Express Van 2007', 'Chevrolet Silverado 1500 Regular Cab 2012', 'Chrysler 300 SRT-8 2010', 'Dodge Caliber Wagon 2012', 'Dodge Ram Pickup 3500 Quad Cab 2009', 'Dodge Dakota Club Cab 2007', 'Dodge Durango SUV 2007', 'FIAT 500 Abarth 2012']
            cars_name2 = [
    "This is a picture of a Ferrari 458 Italia Convertible 2012 car.",
    "This is a picture of a Ford Mustang Convertible 2007 car.",
    "This is a picture of an Aston Martin Virage Coupe 2012 car.",
    "This is a picture of a Ford Ranger SuperCab 2011 car.",
    "This is a picture of a Ford Focus Sedan 2007 car.",
    "This is a picture of a GMC Savana Van 2012 car.",
    "This is a picture of a Geo Metro Convertible 1993 car.",
    "This is a picture of a Honda Odyssey Minivan 2007 car.",
    "This is a picture of a Hyundai Santa Fe SUV 2012 car.",
    "This is a picture of a Hyundai Elantra Sedan 2007 car.",
    "This is a picture of a Hyundai Elantra Touring Hatchback 2012 car.",
    "This is a picture of an Isuzu Ascender SUV 2008 car.",
    "This is a picture of a Jeep Liberty SUV 2012 car.",
    "This is a picture of an Audi R8 Coupe 2012 car.",
    "This is a picture of a Lamborghini Aventador Coupe 2012 car.",
    "This is a picture of a Land Rover LR2 SUV 2012 car.",
    "This is a picture of a Mazda Tribute SUV 2011 car.",
    "This is a picture of a Mercedes-Benz SL-Class Coupe 2009 car.",
    "This is a picture of a Mitsubishi Lancer Sedan 2012 car.",
    "This is a picture of a Nissan 240SX Coupe 1998 car.",
    "This is a picture of a Rolls-Royce Phantom Drophead Coupe Convertible 2012 car.",
    "This is a picture of a Spyker C8 Convertible 2009 car.",
    "This is a picture of a Suzuki SX4 Hatchback 2012 car.",
    "This is a picture of a Toyota Camry Sedan 2012 car.",
    "This is a picture of an Audi TT Hatchback 2011 car.",
    "This is a picture of a Volkswagen Golf Hatchback 1991 car.",
    "This is a picture of a Volvo XC90 SUV 2007 car.",
    "This is a picture of an Audi S4 Sedan 2012 car.",
    "This is a picture of a BMW 1 Series Convertible 2012 car.",
    "This is a picture of an Acura TL Sedan 2012 car.",
    "This is a picture of a BMW 6 Series Convertible 2007 car.",
    "This is a picture of a BMW M5 Sedan 2010 car.",
    "This is a picture of a Bentley Continental Supersports Conv. Convertible 2012 car.",
    "This is a picture of a Bentley Continental GT Coupe 2007 car.",
    "This is a picture of a Buick Regal GS 2012 car.",
    "This is a picture of a Cadillac CTS-V Sedan 2012 car.",
    "This is a picture of a Chevrolet Corvette Convertible 2012 car.",
    "This is a picture of a Chevrolet Camaro Convertible 2012 car.",
    "This is a picture of a Chevrolet Sonic Sedan 2012 car.",
    "This is a picture of a Chevrolet Malibu Hybrid Sedan 2010 car.",
    "This is a picture of an Acura ZDX Hatchback 2012 car.",
    "This is a picture of a Chevrolet Express Van 2007 car.",
    "This is a picture of a Chevrolet Silverado 1500 Regular Cab 2012 car.",
    "This is a picture of a Chrysler 300 SRT-8 2010 car.",
    "This is a picture of a Dodge Caliber Wagon 2012 car.",
    "This is a picture of a Dodge Ram Pickup 3500 Quad Cab 2009 car.",
    "This is a picture of a Dodge Dakota Club Cab 2007 car.",
    "This is a picture of a Dodge Durango SUV 2007 car.",
    "This is a picture of a FIAT 500 Abarth 2012 car."
]
            target_dataset = "/cars.txt"
            file_path = dir_path + model + target_dataset;

            with open(file_path, 'r') as file:
                content = file.read()

            # 用正则匹配所有列表
            matches = re.findall(r'cars_des\d+\s*=\s*(\[.*?\])', content, re.DOTALL)

            # 安全地将字符串转换为列表
            cars_des1 = literal_eval(matches[0])
            cars_des2 = literal_eval(matches[1])
            cars_des3 = literal_eval(matches[2])
            cars_des4 = literal_eval(matches[3])
            cars_des5 = literal_eval(matches[4])

            class_dataset = cars_name1[single_label]  # 类名扩展
            class_dataset2 = cars_name2[single_label]  # 类名扩展
            classes_des1 = cars_des1[single_label]  # 类名描述1
            classes_des2 = cars_des2[single_label]  # 类名描述2
            classes_des3 = cars_des3[single_label]  # 类名描述3
            classes_des4 = cars_des4[single_label]  # 类名描述4
            classes_des5 = cars_des5[single_label]  # 类名描述5

            list_.append(data)
            list_.append(label)  # 类标签
            list_.append(class_dataset)  # 类名1
            list_.append(class_dataset2)  # 描述类名2
            list_.append(classes_des1)  # 描述1
            list_.append(classes_des2)  # 描述2
            list_.append(classes_des3)  # 描述3
            list_.append(classes_des4)  # 描述4
            list_.append(classes_des5)  # 描述5
        # 5. CUB数据集
        elif '004.Groove_billed_Ani' in self.classes_datasets:
            #CUB数据集
            mapped_values = ['Groove_billed_Ani', 'Rhinoceros_Auklet', 'Yellow_headed_Blackbird', 'Painted_Bunting', 'Yellow_breasted_Chat', 'Red_faced_Cormorant', 'Brown_Creeper', 'Mangrove_Cuckoo', 'Northern_Flicker', 'Olive_sided_Flycatcher', 'Frigatebird', 'European_Goldfinch', 'Pied_billed_Grebe', 'Pine_Grosbeak', 'Glaucous_winged_Gull', 'Ring_billed_Gull', 'Ruby_throated_Hummingbird', 'Pomarine_Jaeger', 'Dark_eyed_Junco', 'Green_Kingfisher', 'Red_legged_Kittiwake', 'Western_Meadowlark', 'Nighthawk', 'Hooded_Oriole', 'Brown_Pelican', 'American_Pipit', 'White_necked_Raven', 'Great_Grey_Shrike', 'Chipping_Sparrow', 'Fox_Sparrow', 'Le_Conte_Sparrow', 'Seaside_Sparrow', 'White_crowned_Sparrow', 'Barn_Swallow', 'Summer_Tanager', 'Common_Tern', 'Green_tailed_Towhee', 'Blue_headed_Vireo', 'White_eyed_Vireo', 'Black_throated_Blue_Warbler', 'Cerulean_Warbler', 'Kentucky_Warbler', 'Nashville_Warbler', 'Prairie_Warbler', 'Wilson_Warbler', 'Louisiana_Waterthrush', 'Pileated_Woodpecker', 'Downy_Woodpecker', 'House_Wren', 'Common_Yellowthroat']

            CUB_name1 = mapped_values
            CUB_name2 = [
    "This is a photo of Groove-billed Ani",
    "This is a photo of Rhinoceros Auklet",
    "This is a photo of Yellow-headed Blackbird",
    "This is a photo of Painted Bunting",
    "This is a photo of Yellow-breasted Chat",
    "This is a photo of Red-faced Cormorant",
    "This is a photo of Brown Creeper",
    "This is a photo of Mangrove Cuckoo",
    "This is a photo of Northern Flicker",
    "This is a photo of Olive-sided Flycatcher",
    "This is a photo of Frigatebird",
    "This is a photo of European Goldfinch",
    "This is a photo of Pied-billed Grebe",
    "This is a photo of Pine Grosbeak",
    "This is a photo of Glaucous-winged Gull",
    "This is a photo of Ring-billed Gull",
    "This is a photo of Ruby-throated Hummingbird",
    "This is a photo of Pomarine Jaeger",
    "This is a photo of Dark-eyed Junco",
    "This is a photo of Green Kingfisher",
    "This is a photo of Red-legged Kittiwake",
    "This is a photo of Western Meadowlark",
    "This is a photo of Nighthawk",
    "This is a photo of Hooded Oriole",
    "This is a photo of Brown Pelican",
    "This is a photo of American Pipit",
    "This is a photo of White-necked Raven",
    "This is a photo of Great Grey Shrike",
    "This is a photo of Chipping Sparrow",
    "This is a photo of Fox Sparrow",
    "This is a photo of Le Conte's Sparrow",
    "This is a photo of Seaside Sparrow",
    "This is a photo of White-crowned Sparrow",
    "This is a photo of Barn Swallow",
    "This is a photo of Summer Tanager",
    "This is a photo of Common Tern",
    "This is a photo of Green-tailed Towhee",
    "This is a photo of Blue-headed Vireo",
    "This is a photo of White-eyed Vireo",
    "This is a photo of Black-throated Blue Warbler",
    "This is a photo of Cerulean Warbler",
    "This is a photo of Kentucky Warbler",
    "This is a photo of Nashville Warbler",
    "This is a photo of Prairie Warbler",
    "This is a photo of Wilson's Warbler",
    "This is a photo of Louisiana Waterthrush",
    "This is a photo of Pileated Woodpecker",
    "This is a photo of Downy Woodpecker",
    "This is a photo of House Wren",
    "This is a photo of Common Yellowthroat"
]
            target_dataset = "/CUB.txt"
            file_path = dir_path + model + target_dataset;

            with open(file_path, 'r') as file:
                content = file.read()

            # 用正则匹配所有列表
            matches = re.findall(r'CUB_des\d+\s*=\s*(\[.*?\])', content, re.DOTALL)

            # 安全地将字符串转换为列表
            CUB_des1 = literal_eval(matches[0])
            CUB_des2 = literal_eval(matches[1])
            CUB_des3 = literal_eval(matches[2])
            CUB_des4 = literal_eval(matches[3])
            CUB_des5 = literal_eval(matches[4])


            class_dataset = CUB_name1[single_label]  # 类名扩展
            class_dataset2 = CUB_name2[single_label]  # 类名扩展
            classes_des1 = CUB_des1[single_label]  # 类名描述1
            classes_des2 = CUB_des2[single_label]  # 类名描述2
            classes_des3 = CUB_des3[single_label]  # 类名描述3
            classes_des4 = CUB_des4[single_label]  # 类名描述4
            classes_des5 = CUB_des5[single_label]  # 类名描述5

            list_.append(data)
            list_.append(label)  # 类标签
            list_.append(class_dataset)  # 类名1
            list_.append(class_dataset2)  # 描述类名2
            list_.append(classes_des1)  # 描述1
            list_.append(classes_des2)  # 描述2
            list_.append(classes_des3)  # 描述3
            list_.append(classes_des4)  # 描述4
            list_.append(classes_des5)  # 描述5
        # Places数据集
        elif 'arena-hockey' in self.classes_datasets:
            # Places数据集
            mapped_values = ['alcove', 'amusement park', 'arcade', 'arena hockey', 'art gallery', 'assembly line', 'auditorium',
 'bakery shop', 'ballroom', 'bar', 'basement', 'bazaar outdoor', 'bedchamber', 'berth', 'boathouse',
 'bow window indoor', 'building facade', 'bus station indoor', 'cafeteria', 'canal urban', 'carrousel',
 'chalet', 'church outdoor', 'closet', 'coffee shop', 'construction site', 'cottage', 'crevasse',
 'department store', 'diner outdoor', 'doorway outdoor', 'driveway', 'elevator shaft', 'escalator indoor',
 'fastfood restaurant', 'fire escape', 'florist shop indoor', 'forest path', 'galley', 'gazebo exterior',
 'glacier', 'grotto', 'harbor', 'highway', 'hospital room', 'house', 'ice shelf', 'igloo', 'jacuzzi indoor',
 'junkyard', 'kitchen', 'landing deck', 'legislative chamber', 'living room', 'locker room', 'market outdoor',
 'medina', 'motel', 'movie theater indoor', 'natural history museum', 'ocean', 'oilrig', 'pagoda',
 'parking garage indoor', 'patio', 'phone booth', 'pizzeria', 'pond', 'racecourse', 'rainforest',
 'residential neighborhood', 'rice paddy', 'rope bridge', 'sauna', 'shed', 'shower', 'skyscraper', 'stable',
 'stage indoor', 'street', 'swamp', 'synagogue outdoor', 'throne room', 'toyshop', 'tree house', 'utility room',
 'viaduct', 'volleyball court outdoor', 'waterfall', 'wheat field', 'youth hostel']

            Places_name1 = mapped_values
            Places_name2 = [
    "This is a photo of alcove",
    "This is a photo of amusement park",
    "This is a photo of arcade",
    "This is a photo of arena hockey",
    "This is a photo of art gallery",
    "This is a photo of assembly line",
    "This is a photo of auditorium",
    "This is a photo of bakery shop",
    "This is a photo of ballroom",
    "This is a photo of bar",
    "This is a photo of basement",
    "This is a photo of bazaar outdoor",
    "This is a photo of bedchamber",
    "This is a photo of berth",
    "This is a photo of boathouse",
    "This is a photo of bow window indoor",
    "This is a photo of building facade",
    "This is a photo of bus station indoor",
    "This is a photo of cafeteria",
    "This is a photo of canal urban",
    "This is a photo of carrousel",
    "This is a photo of chalet",
    "This is a photo of church outdoor",
    "This is a photo of closet",
    "This is a photo of coffee shop",
    "This is a photo of construction site",
    "This is a photo of cottage",
    "This is a photo of crevasse",
    "This is a photo of department store",
    "This is a photo of diner outdoor",
    "This is a photo of doorway outdoor",
    "This is a photo of driveway",
    "This is a photo of elevator shaft",
    "This is a photo of escalator indoor",
    "This is a photo of fastfood restaurant",
    "This is a photo of fire escape",
    "This is a photo of florist shop indoor",
    "This is a photo of forest path",
    "This is a photo of galley",
    "This is a photo of gazebo exterior",
    "This is a photo of glacier",
    "This is a photo of grotto",
    "This is a photo of harbor",
    "This is a photo of highway",
    "This is a photo of hospital room",
    "This is a photo of house",
    "This is a photo of ice shelf",
    "This is a photo of igloo",
    "This is a photo of jacuzzi indoor",
    "This is a photo of junkyard",
    "This is a photo of kitchen",
    "This is a photo of landing deck",
    "This is a photo of legislative chamber",
    "This is a photo of living room",
    "This is a photo of locker room",
    "This is a photo of market outdoor",
    "This is a photo of medina",
    "This is a photo of motel",
    "This is a photo of movie theater indoor",
    "This is a photo of natural history museum",
    "This is a photo of ocean",
    "This is a photo of oilrig",
    "This is a photo of pagoda",
    "This is a photo of parking garage indoor",
    "This is a photo of patio",
    "This is a photo of phone booth",
    "This is a photo of pizzeria",
    "This is a photo of pond",
    "This is a photo of racecourse",
    "This is a photo of rainforest",
    "This is a photo of residential neighborhood",
    "This is a photo of rice paddy",
    "This is a photo of rope bridge",
    "This is a photo of sauna",
    "This is a photo of shed",
    "This is a photo of shower",
    "This is a photo of skyscraper",
    "This is a photo of stable",
    "This is a photo of stage indoor",
    "This is a photo of street",
    "This is a photo of swamp",
    "This is a photo of synagogue outdoor",
    "This is a photo of throne room",
    "This is a photo of toyshop",
    "This is a photo of tree house",
    "This is a photo of utility room",
    "This is a photo of viaduct",
    "This is a photo of volleyball court outdoor",
    "This is a photo of waterfall",
    "This is a photo of wheat field",
    "This is a photo of youth hostel"
]

            target_dataset = "/Places.txt"
            file_path = dir_path + model + target_dataset;

            with open(file_path, 'r') as file:
                content = file.read()

            # 用正则匹配所有列表
            matches = re.findall(r'Places_des\d+\s*=\s*(\[.*?\])', content, re.DOTALL)

            # 安全地将字符串转换为列表
            Places_des1 = literal_eval(matches[0])
            Places_des2 = literal_eval(matches[1])
            Places_des3 = literal_eval(matches[2])
            Places_des4 = literal_eval(matches[3])
            Places_des5 = literal_eval(matches[4])


            class_dataset = Places_name1[single_label]  # 类名扩展
            class_dataset2 = Places_name2[single_label]  # 类名扩展
            classes_des1 = Places_des1[single_label]  # 类名描述1
            classes_des2 = Places_des2[single_label]  # 类名描述2
            classes_des3 = Places_des3[single_label]  # 类名描述3
            classes_des4 = Places_des4[single_label]  # 类名描述4
            classes_des5 = Places_des5[single_label]  # 类名描述5


            list_.append(data)
            list_.append(label)  # 类标签
            list_.append(class_dataset)  # 类名1
            list_.append(class_dataset2)  # 描述类名2
            list_.append(classes_des1)  # 描述1
            list_.append(classes_des2)  # 描述2
            list_.append(classes_des3)  # 描述3
            list_.append(classes_des4)  # 描述4
            list_.append(classes_des5)  # 描述5
        # Plantae
        elif '5283' in self.classes_datasets:
            #Plantae
            Plantae_name1 = ['Cocos nucifera', 'Dichelostemma capitatum', 'Iris pseudacorus', 'Medeola virginiana', 'Achillea millefolium',
         'Ageratina altissima', 'Bellis perennis', 'Cirsium arvense', 'Cirsium vulgare', 'Lobelia cardinalis',
         'Claytonia virginica', 'Portulaca oleracea', 'Cornus florida', 'Cornus sericea', 'Lonicera japonica',
         'Kalmia latifolia', 'Anagallis arvensis', 'Lupinus texensis', 'Fagus grandifolia', 'Quercus alba',
         'Asclepias asperula', 'Asclepias incarnata', 'Galium aparine', 'Campsis radicans', 'Glechoma hederacea',
         'Fraxinus americana', 'Mimulus aurantiacus', 'Plantago lanceolata', 'Verbascum thapsus',
         'Callicarpa americana', 'Sassafras albidum', 'Oxalis pes-caprae', 'Platanus occidentalis',
         'Berberis thunbergii', 'Sanguinaria canadensis', 'Aquilegia formosa', 'Fragaria vesca', 'Potentilla indica',
         'Rubus ursinus', 'Rhus typhina', 'Acer saccharum', 'Hamamelis virginiana', 'Solanum dulcamara',
         'Thuja occidentalis', 'Abies balsamea', 'Osmunda regalis', 'Polystichum acrostichoides', 'Polystichum munitum',
         'Onoclea sensibilis', 'Adiantum pedatum']
            Plantae_name2 = [
                'This is a picture of plant Cocos nucifera', 'This is a picture of plant Dichelostemma capitatum', 'This is a picture of plant Iris pseudacorus', 'This is a picture of plant Medeola virginiana', 'This is a picture of plant Achillea millefolium', 'This is a picture of plant Ageratina altissima', 'This is a picture of plant Bellis perennis', 'This is a picture of plant Cirsium arvense', 'This is a picture of plant Cirsium vulgare', 'This is a picture of plant Lobelia cardinalis', 'This is a picture of plant Claytonia virginica', 'This is a picture of plant Portulaca oleracea', 'This is a picture of plant Cornus florida', 'This is a picture of plant Cornus sericea', 'This is a picture of plant Lonicera japonica', 'This is a picture of plant Kalmia latifolia', 'This is a picture of plant Anagallis arvensis', 'This is a picture of plant Lupinus texensis', 'This is a picture of plant Fagus grandifolia', 'This is a picture of plant Quercus alba', 'This is a picture of plant Asclepias asperula', 'This is a picture of plant Asclepias incarnata', 'This is a picture of plant Galium aparine', 'This is a picture of plant Campsis radicans', 'This is a picture of plant Glechoma hederacea', 'This is a picture of plant Fraxinus americana', 'This is a picture of plant Mimulus aurantiacus', 'This is a picture of plant Plantago lanceolata', 'This is a picture of plant Verbascum thapsus', 'This is a picture of plant Callicarpa americana', 'This is a picture of plant Sassafras albidum', 'This is a picture of plant Oxalis pes-caprae', 'This is a picture of plant Platanus occidentalis', 'This is a picture of plant Berberis thunbergii', 'This is a picture of plant Sanguinaria canadensis', 'This is a picture of plant Aquilegia formosa', 'This is a picture of plant Fragaria vesca', 'This is a picture of plant Potentilla indica', 'This is a picture of plant Rubus ursinus', 'This is a picture of plant Rhus typhina', 'This is a picture of plant Acer saccharum', 'This is a picture of plant Hamamelis virginiana', 'This is a picture of plant Solanum dulcamara', 'This is a picture of plant Thuja occidentalis', 'This is a picture of plant Abies balsamea', 'This is a picture of plant Osmunda regalis', 'This is a picture of plant Polystichum acrostichoides', 'This is a picture of plant Polystichum munitum', 'This is a picture of plant Onoclea sensibilis', 'This is a picture of plant Adiantum pedatum']


            target_dataset = "/Plantae.txt"
            file_path = dir_path + model + target_dataset;

            with open(file_path, 'r') as file:
                content = file.read()

            # 用正则匹配所有列表
            matches = re.findall(r'Plantae_des\d+\s*=\s*(\[.*?\])', content, re.DOTALL)

            # 安全地将字符串转换为列表
            Plantae_des1 = literal_eval(matches[0])
            Plantae_des2 = literal_eval(matches[1])
            Plantae_des3 = literal_eval(matches[2])
            Plantae_des4 = literal_eval(matches[3])
            Plantae_des5 = literal_eval(matches[4])
            class_dataset = Plantae_name1[single_label]  # 类名扩展
            class_dataset2 = Plantae_name2[single_label]  # 类名扩展
            classes_des1 = Plantae_des1[single_label]  # 类名描述1
            classes_des2 = Plantae_des2[single_label]  # 类名描述2
            classes_des3 = Plantae_des3[single_label]  # 类名描述3
            classes_des4 = Plantae_des4[single_label]  # 类名描述4
            classes_des5 = Plantae_des5[single_label]  # 类名描述5

            list_.append(data)
            list_.append(label)  # 类标签
            list_.append(class_dataset)  # 类名1
            list_.append(class_dataset2)  # 描述类名2
            list_.append(classes_des1)  # 描述1
            list_.append(classes_des2)  # 描述2
            list_.append(classes_des3)  # 描述3
            list_.append(classes_des4)  # 描述4
            list_.append(classes_des5)  # 描述5
        # cifar-FS 数据集
        elif 'bed' in self.classes_datasets:
            # cifar-FS 数据集
            classes_cifarfs_name = [
    "A photo of a baby",
    "A photo of a bed",
    "A photo of a bicycle",
    "A photo of a chimpanzee",
    "A photo of a fox",
    "A photo of a leopard",
    "A photo of a man",
    "A photo of a pickup truck",
    "A photo of a plain",
    "A photo of a poppy",
    "A photo of a rocket",
    "A photo of a rose",
    "A photo of a snail",
    "A photo of a sweet pepper",
    "A photo of a table",
    "A photo of a telephone",
    "A photo of a wardrobe",
    "A photo of a whale",
    "A photo of a woman",
    "A photo of a worm"
]
            class_dataset2 = classes_cifarfs_name[single_label]  # 类名扩展

            target_dataset = "/cifar-FS.txt"
            file_path = dir_path + model + target_dataset;

            with open(file_path, 'r') as file:
                content = file.read()

            # 用正则匹配所有列表
            matches = re.findall(r'cifarfs_des\d+\s*=\s*(\[.*?\])', content, re.DOTALL)

            # 安全地将字符串转换为列表
            cifarfs_des1 = literal_eval(matches[0])
            cifarfs_des2 = literal_eval(matches[1])
            cifarfs_des3 = literal_eval(matches[2])
            cifarfs_des4 = literal_eval(matches[3])
            cifarfs_des5 = literal_eval(matches[4])

            classes_des1 = cifarfs_des1[single_label]  # 类名描述1
            classes_des2 = cifarfs_des2[single_label]  # 类名描述2
            classes_des3 = cifarfs_des3[single_label]  # 类名描述3
            classes_des4 = cifarfs_des4[single_label]  # 类名描述4
            classes_des5 = cifarfs_des5[single_label]  # 类名描述5

            list_.append(data)
            list_.append(label)  # 类标签
            list_.append(class_dataset)  # 类名1
            list_.append(class_dataset2)  # 描述类名2
            list_.append(classes_des1)  # 描述1
            list_.append(classes_des2)  # 描述2
            list_.append(classes_des3)  # 描述3
            list_.append(classes_des4)  # 描述4
            list_.append(classes_des5)  # 描述5
        # FC-100 数据集#['baby', 'beaver', 'bee', 'beetle', 'boy', 'butterfly', 'caterpillar', 'cockroach', 'dolphin', 'fox', 'girl', 'man', 'otter', 'porcupine', 'possum', 'raccoon', 'seal', 'skunk', 'whale', 'woman']
        elif 'bee' in self.classes_datasets:
            classes_FC100_name = [
    "A photo of a baby",
    "A photo of a beaver",
    "A photo of a bee",
    "A photo of a beetle",
    "A photo of a boy",
    "A photo of a butterfly",
    "A photo of a caterpillar",
    "A photo of a cockroach",
    "A photo of a dolphin",
    "A photo of a fox",
    "A photo of a girl",
    "A photo of a man",
    "A photo of an otter",
    "A photo of a porcupine",
    "A photo of a possum",
    "A photo of a raccoon",
    "A photo of a seal",
    "A photo of a skunk",
    "A photo of a whale",
    "A photo of a woman"
]
            class_dataset2 = classes_FC100_name[single_label]  # 类名扩展
            target_dataset = "/FC-100.txt"
            file_path = dir_path + model + target_dataset;

            with open(file_path, 'r') as file:
                content = file.read()

            # 用正则匹配所有列表
            matches = re.findall(r'FC100_des\d+\s*=\s*(\[.*?\])', content, re.DOTALL)

            # 安全地将字符串转换为列表
            FC100_des1 = literal_eval(matches[0])
            FC100_des2 = literal_eval(matches[1])
            FC100_des3 = literal_eval(matches[2])
            FC100_des4 = literal_eval(matches[3])
            FC100_des5 = literal_eval(matches[4])

            classes_des1 = FC100_des1[single_label]  # 类名描述1
            classes_des2 = FC100_des2[single_label]  # 类名描述2
            classes_des3 = FC100_des3[single_label]  # 类名描述3
            classes_des4 = FC100_des4[single_label]  # 类名描述4
            classes_des5 = FC100_des5[single_label]  # 类名描述5

            list_.append(data)
            list_.append(label)  # 类标签
            list_.append(class_dataset)  # 类名1
            list_.append(class_dataset2)  # 描述类名2
            list_.append(classes_des1)  # 描述1
            list_.append(classes_des2)  # 描述2
            list_.append(classes_des3)  # 描述3
            list_.append(classes_des4)  # 描述4
            list_.append(classes_des5)  # 描述5
        # miniImagenet ['nematode, nematode worm, roundworm','king crab, Alaska crab, Alaskan king crab, Alaska king crab, Paralithodes camtschatica', 'golden retriever','malamute, malemute, Alaskan malamute', 'dalmatian, coach dog, carriage dog','African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus', 'lion, king of beasts, Panthera leo','ant, emmet, pismire', 'black-footed ferret, ferret, Mustela nigripes', 'bookshop, bookstore, bookstall','crate', 'cuirass', 'electric guitar', 'hourglass', 'mixing bowl', 'school bus', 'scoreboard','theater curtain, theatre curtain', 'vase', 'trifle']
        elif 'n01930112' in self.classes_datasets:
            classes_miniImagenet_name = ['nematode, nematode worm, roundworm',
                          'king crab, Alaska crab, Alaskan king crab, Alaska king crab, Paralithodes camtschatica',
                          'golden retriever',
                          'malamute, malemute, Alaskan malamute', 'dalmatian, coach dog, carriage dog',
                          'African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus',
                          'lion, king of beasts, Panthera leo',
                          'ant, emmet, pismire', 'black-footed ferret, ferret, Mustela nigripes',
                          'bookshop, bookstore, bookstall',
                          'crate', 'cuirass', 'electric guitar', 'hourglass', 'mixing bowl', 'school bus', 'scoreboard',
                          'theater curtain, theatre curtain', 'vase', 'trifle']
            class_dataset = classes_miniImagenet_name[single_label]
            class_dataset2 = [
    "A picture of a nematode, nematode worm, roundworm",
    "A picture of a king crab, Alaska crab, Alaskan king crab, Alaska king crab, Paralithodes camtschatica",
    "A picture of a golden retriever",
    "A picture of a malamute, malemute, Alaskan malamute",
    "A picture of a dalmatian, coach dog, carriage dog",
    "A picture of an African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus",
    "A picture of a lion, king of beasts, Panthera leo",
    "A picture of an ant, emmet, pismire",
    "A picture of a black-footed ferret, ferret, Mustela nigripes",
    "A picture of a bookshop, bookstore, bookstall",
    "A picture of a crate",
    "A picture of a cuirass",
    "A picture of an electric guitar",
    "A picture of an hourglass",
    "A picture of a mixing bowl",
    "A picture of a school bus",
    "A picture of a scoreboard",
    "A picture of a theater curtain, theatre curtain",
    "A picture of a vase",
    "A picture of a trifle"
][single_label]  # 类名扩展
            target_dataset = "/miniImagenet.txt"
            file_path = dir_path + model + target_dataset;

            with open(file_path, 'r') as file:
                content = file.read()

            # 用正则匹配所有列表
            matches = re.findall(r'miniImagenet_des\d+\s*=\s*(\[.*?\])', content, re.DOTALL)

            # 安全地将字符串转换为列表
            miniImagenet_des1 = literal_eval(matches[0])
            miniImagenet_des2 = literal_eval(matches[1])
            miniImagenet_des3 = literal_eval(matches[2])
            miniImagenet_des4 = literal_eval(matches[3])
            miniImagenet_des5 = literal_eval(matches[4])

            classes_des1 = miniImagenet_des1[single_label]  # 类名描述1
            classes_des2 = miniImagenet_des2[single_label]  # 类名描述2
            classes_des3 = miniImagenet_des3[single_label]  # 类名描述3
            classes_des4 = miniImagenet_des4[single_label]  # 类名描述4
            classes_des5 = miniImagenet_des5[single_label]  # 类名描述5

            list_.append(data)
            list_.append(label)  # 类标签
            list_.append(class_dataset)  # 类名1
            list_.append(class_dataset2)  # 描述类名2
            list_.append(classes_des1)  # 描述1
            list_.append(classes_des2)  # 描述2
            list_.append(classes_des3)  # 描述3
            list_.append(classes_des4)  # 描述4
            list_.append(classes_des5)  # 描述5
        # tiered-Imagenet
        elif 'n01440764' in self.classes_datasets:
            classes_tieredImagenet_name = ['tench, Tinca tinca', 'goldfish, Carassius auratus',
                                         'great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias',
                                         'tiger shark, Galeocerdo cuvieri', 'hammerhead, hammerhead shark',
                                         'electric ray, crampfish, numbfish, torpedo', 'stingray', 'kuvasz',
                                         'schipperke', 'groenendael', 'malinois', 'briard', 'kelpie', 'komondor',
                                         'Old English sheepdog, bobtail',
                                         'Shetland sheepdog, Shetland sheep dog, Shetland', 'collie', 'Border collie',
                                         'Bouvier des Flandres, Bouviers des Flandres', 'Rottweiler',
                                         'German shepherd, German shepherd dog, German police dog, alsatian',
                                         'Doberman, Doberman pinscher', 'miniature pinscher',
                                         'Greater Swiss Mountain dog', 'Bernese mountain dog', 'Appenzeller',
                                         'EntleBucher', 'boxer', 'bull mastiff', 'Tibetan mastiff', 'French bulldog',
                                         'Great Dane', 'Saint Bernard, St Bernard', 'Eskimo dog, husky',
                                         'malamute, malemute, Alaskan malamute', 'Siberian husky',
                                         'affenpinscher, monkey pinscher, monkey dog', 'tiger beetle',
                                         'ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle',
                                         'ground beetle, carabid beetle',
                                         'long-horned beetle, longicorn, longicorn beetle', 'leaf beetle, chrysomelid',
                                         'dung beetle', 'rhinoceros beetle', 'weevil', 'fly', 'bee',
                                         'ant, emmet, pismire', 'grasshopper, hopper', 'cricket',
                                         'walking stick, walkingstick, stick insect', 'cockroach, roach',
                                         'mantis, mantid', 'cicada, cicala', 'leafhopper', 'lacewing, lacewing fly',
                                         "dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk",
                                         'damselfly', 'admiral', 'ringlet, ringlet butterfly',
                                         'monarch, monarch butterfly, milkweed butterfly, Danaus plexippus',
                                         'cabbage butterfly', 'sulphur butterfly, sulfur butterfly',
                                         'lycaenid, lycaenid butterfly', 'barracouta, snoek', 'eel',
                                         'coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch',
                                         'rock beauty, Holocanthus tricolor', 'anemone fish', 'sturgeon',
                                         'gar, garfish, garpike, billfish, Lepisosteus osseus', 'lionfish',
                                         'puffer, pufferfish, blowfish, globefish',
                                         'bannister, banister, balustrade, balusters, handrail', 'barrel, cask',
                                         'bathtub, bathing tub, bath, tub', 'beaker', 'beer bottle',
                                         'breakwater, groin, groyne, mole, bulwark, seawall, jetty', 'bucket, pail',
                                         'caldron, cauldron', 'chainlink fence', 'coffee mug', 'coffeepot',
                                         'dam, dike, dyke', 'face powder', 'grille, radiator grille', 'ladle', 'mortar',
                                         'picket fence, paling', 'pill bottle', 'pitcher, ewer',
                                         'pop bottle, soda bottle', 'rain barrel', 'sliding door', 'stone wall',
                                         'teapot', 'tub, vat', 'turnstile', 'vase',
                                         'washbasin, handbasin, washbowl, lavabo, wash-hand basin', 'water bottle',
                                         'water jug', 'water tower', 'whiskey jug', 'wine bottle',
                                         'worm fence, snake fence, snake-rail fence, Virginia fence', 'menu', 'plate',
                                         'guacamole', 'consomme', 'hot pot, hotpot', 'trifle', 'ice cream, icecream',
                                         'ice lolly, lolly, lollipop, popsicle', 'cheeseburger',
                                         'hotdog, hot dog, red hot', 'head cabbage', 'broccoli', 'cauliflower',
                                         'zucchini, courgette', 'spaghetti squash', 'acorn squash', 'butternut squash',
                                         'cucumber, cuke', 'artichoke, globe artichoke', 'bell pepper', 'cardoon',
                                         'mushroom', 'Granny Smith', 'strawberry', 'orange', 'lemon', 'fig',
                                         'pineapple, ananas', 'banana', 'jackfruit, jak, jack', 'custard apple',
                                         'pomegranate', 'hay', 'carbonara', 'chocolate sauce, chocolate syrup', 'dough',
                                         'pizza, pizza pie', 'potpie', 'burrito', 'red wine', 'espresso', 'cup',
                                         'eggnog', 'alp', 'cliff, drop, drop-off', 'coral reef', 'geyser',
                                         'lakeside, lakeshore', 'promontory, headland, head, foreland',
                                         'sandbar, sand bar', 'seashore, coast, seacoast, sea-coast', 'valley, vale',
                                         'volcano']
            class_dataset = classes_tieredImagenet_name[single_label]

            class_dataset2 = [
                "A picture of a tench, Tinca tinca",
                "A picture of a goldfish, Carassius auratus",
                "A picture of a great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias",
                "A picture of a tiger shark, Galeocerdo cuvieri",
                "A picture of a hammerhead, hammerhead shark",
                "A picture of an electric ray, crampfish, numbfish, torpedo",
                "A picture of a stingray",
                "A picture of a kuvasz",
                "A picture of a schipperke",
                "A picture of a groenendael",
                "A picture of a malinois",
                "A picture of a briard",
                "A picture of a kelpie",
                "A picture of a komondor",
                "A picture of an Old English sheepdog, bobtail",
                "A picture of a Shetland sheepdog, Shetland sheep dog, Shetland",
                "A picture of a collie",
                "A picture of a Border collie",
                "A picture of a Bouvier des Flandres, Bouviers des Flandres",
                "A picture of a Rottweiler",
                "A picture of a German shepherd, German shepherd dog, German police dog, alsatian",
                "A picture of a Doberman, Doberman pinscher",
                "A picture of a miniature pinscher",
                "A picture of a Greater Swiss Mountain dog",
                "A picture of a Bernese mountain dog",
                "A picture of an Appenzeller",
                "A picture of an EntleBucher",
                "A picture of a boxer",
                "A picture of a bull mastiff",
                "A picture of a Tibetan mastiff",
                "A picture of a French bulldog",
                "A picture of a Great Dane",
                "A picture of a Saint Bernard, St Bernard",
                "A picture of an Eskimo dog, husky",
                "A picture of a malamute, malemute, Alaskan malamute",
                "A picture of a Siberian husky",
                "A picture of an affenpinscher, monkey pinscher, monkey dog",
                "A picture of a tiger beetle",
                "A picture of a ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle",
                "A picture of a ground beetle, carabid beetle",
                "A picture of a long-horned beetle, longicorn, longicorn beetle",
                "A picture of a leaf beetle, chrysomelid",
                "A picture of a dung beetle",
                "A picture of a rhinoceros beetle",
                "A picture of a weevil",
                "A picture of a fly",
                "A picture of a bee",
                "A picture of an ant, emmet, pismire",
                "A picture of a grasshopper, hopper",
                "A picture of a cricket",
                "A picture of a walking stick, walkingstick, stick insect",
                "A picture of a cockroach, roach",
                "A picture of a mantis, mantid",
                "A picture of a cicada, cicala",
                "A picture of a leafhopper",
                "A picture of a lacewing, lacewing fly",
                "A picture of a dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk",
                "A picture of a damselfly",
                "A picture of an admiral",
                "A picture of a ringlet, ringlet butterfly",
                "A picture of a monarch, monarch butterfly, milkweed butterfly, Danaus plexippus",
                "A picture of a cabbage butterfly",
                "A picture of a sulphur butterfly, sulfur butterfly",
                "A picture of a lycaenid, lycaenid butterfly",
                "A picture of a barracouta, snoek",
                "A picture of an eel",
                "A picture of a coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch",
                "A picture of a rock beauty, Holocanthus tricolor",
                "A picture of an anemone fish",
                "A picture of a sturgeon",
                "A picture of a gar, garfish, garpike, billfish, Lepisosteus osseus",
                "A picture of a lionfish",
                "A picture of a puffer, pufferfish, blowfish, globefish",
                "A picture of a bannister, banister, balustrade, balusters, handrail",
                "A picture of a barrel, cask",
                "A picture of a bathtub, bathing tub, bath, tub",
                "A picture of a beaker",
                "A picture of a beer bottle",
                "A picture of a breakwater, groin, groyne, mole, bulwark, seawall, jetty",
                "A picture of a bucket, pail",
                "A picture of a caldron, cauldron",
                "A picture of a chainlink fence",
                "A picture of a coffee mug",
                "A picture of a coffeepot",
                "A picture of a dam, dike, dyke",
                "A picture of a face powder",
                "A picture of a grille, radiator grille",
                "A picture of a ladle",
                "A picture of a mortar",
                "A picture of a picket fence, paling",
                "A picture of a pill bottle",
                "A picture of a pitcher, ewer",
                "A picture of a pop bottle, soda bottle",
                "A picture of a rain barrel",
                "A picture of a sliding door",
                "A picture of a stone wall",
                "A picture of a teapot",
                "A picture of a tub, vat",
                "A picture of a turnstile",
                "A picture of a vase",
                "A picture of a washbasin, handbasin, washbowl, lavabo, wash-hand basin",
                "A picture of a water bottle",
                "A picture of a water jug",
                "A picture of a water tower",
                "A picture of a whiskey jug",
                "A picture of a wine bottle",
                "A picture of a worm fence, snake fence, snake-rail fence, Virginia fence",
                "A picture of a menu",
                "A picture of a plate",
                "A picture of a guacamole",
                "A picture of a consomme",
                "A picture of a hot pot, hotpot",
                "A picture of a trifle",
                "A picture of an ice cream, icecream",
                "A picture of an ice lolly, lolly, lollipop, popsicle",
                "A picture of a cheeseburger",
                "A picture of a hotdog, hot dog, red hot",
                "A picture of a head cabbage",
                "A picture of a broccoli",
                "A picture of a cauliflower",
                "A picture of a zucchini, courgette",
                "A picture of a spaghetti squash",
                "A picture of an acorn squash",
                "A picture of a butternut squash",
                "A picture of a cucumber, cuke",
                "A picture of an artichoke, globe artichoke",
                "A picture of a bell pepper",
                "A picture of a cardoon",
                "A picture of a mushroom",
                "A picture of a Granny Smith",
                "A picture of a strawberry",
                "A picture of an orange",
                "A picture of a lemon",
                "A picture of a fig",
                "A picture of a pineapple, ananas",
                "A picture of a banana",
                "A picture of a jackfruit, jak, jack",
                "A picture of a custard apple",
                "A picture of a pomegranate",
                "A picture of a hay",
                "A picture of a carbonara",
                "A picture of a chocolate sauce, chocolate syrup",
                "A picture of a dough",
                "A picture of a pizza, pizza pie",
                "A picture of a potpie",
                "A picture of a burrito",
                "A picture of a red wine",
                "A picture of an espresso",
                "A picture of a cup",
                "A picture of an eggnog",
                "A picture of an alp",
                "A picture of a cliff, drop, drop-off",
                "A picture of a coral reef",
                "A picture of a geyser",
                "A picture of a lakeside, lakeshore",
                "A picture of a promontory, headland, head, foreland",
                "A picture of a sandbar, sand bar",
                "A picture of a seashore, coast, seacoast, sea-coast",
                "A picture of a valley, vale",
                "A picture of a volcano"
            ][single_label]  # 类名扩展
            target_dataset = "/tiered-Imagenet.txt"
            file_path = dir_path + model + target_dataset;

            with open(file_path, 'r') as file:
                content = file.read()

            # 用正则匹配所有列表
            matches = re.findall(r'tieredimagenet_des\d+\s*=\s*(\[.*?\])', content, re.DOTALL)

            # 安全地将字符串转换为列表
            tieredImagenet_des1 = literal_eval(matches[0])
            tieredImagenet_des2 = literal_eval(matches[1])
            tieredImagenet_des3 = literal_eval(matches[2])
            tieredImagenet_des4 = literal_eval(matches[3])
            tieredImagenet_des5 = literal_eval(matches[4])

            classes_des1 = tieredImagenet_des1[single_label]  # 类名描述1
            classes_des2 = tieredImagenet_des2[single_label]  # 类名描述2
            classes_des3 = tieredImagenet_des3[single_label]  # 类名描述3
            classes_des4 = tieredImagenet_des4[single_label]  # 类名描述4
            classes_des5 = tieredImagenet_des5[single_label]  # 类名描述5

            list_.append(data)
            list_.append(label)  # 类标签
            list_.append(class_dataset)  # 类名1
            list_.append(class_dataset2)  # 描述类名2
            list_.append(classes_des1)  # 描述1
            list_.append(classes_des2)  # 描述2
            list_.append(classes_des3)  # 描述3
            list_.append(classes_des4)  # 描述4
            list_.append(classes_des5)  # 描述5
        # medical
        elif '三血管气管切面' in self.classes_datasets:
            classes_medical_name = ['Three Vessel Trachea Cross-section',
     'Upper Abdominal Horizontal Cross-section',
     'Thalamus Horizontal Cross-section',
     'Lateral Ventricle Horizontal Cross-section',
     'Right Ventricular Outflow Tract Cross-section',
     'Four-Chamber Heart Cross-section',
     'Cerebellum Horizontal Cross-section',
     'Radius and Ulna Coronal Cross-section ',
     'Left Ventricular Outflow Tract Cross-section',
     'Femur Long Axis Cross-section',
     'Humerus Long Axis Cross-section',
     'Tibia and Fibula Coronal Cross-section',
     'Lumbosacral Spine Sagittal Cross-section',
     'Bladder Horizontal Cross-section', 'Cervical-Thoracic Spine Sagittal Cross-section']
            class_dataset = classes_medical_name[single_label]

            class_dataset2 = [
    'This is a Three Vessel Trachea Cross-section',
    'This is an Upper Abdominal Horizontal Cross-section',
    'This is a Thalamus Horizontal Cross-section',
    'This is a Lateral Ventricle Horizontal Cross-section',
    'This is a Right Ventricular Outflow Tract Cross-section',
    'This is a Four-Chamber Heart Cross-section',
    'This is a Cerebellum Horizontal Cross-section',
    'This is a Radius and Ulna Coronal Cross-section',
    'This is a Left Ventricular Outflow Tract Cross-section',
    'This is a Femur Long Axis Cross-section',
    'This is a Humerus Long Axis Cross-section',
    'This is a Tibia and Fibula Coronal Cross-section',
    'This is a Lumbosacral Spine Sagittal Cross-section',
    'This is a Bladder Horizontal Cross-section',
    'This is a Cervical-Thoracic Spine Sagittal Cross-section'
][single_label]  # 类名扩展
            target_dataset = "/medical.txt"
            file_path = dir_path + model + target_dataset;

            with open(file_path, 'r') as file:
                content = file.read()

            # 用正则匹配所有列表
            matches = re.findall(r'medical_des\d+\s*=\s*(\[.*?\])', content, re.DOTALL)

            # 安全地将字符串转换为列表
            medical_des1 = literal_eval(matches[0])
            medical_des2 = literal_eval(matches[1])
            medical_des3 = literal_eval(matches[2])
            medical_des4 = literal_eval(matches[3])
            medical_des5 = literal_eval(matches[4])


            classes_des1 = medical_des1[single_label]  # 类名描述1
            classes_des2 = medical_des2[single_label]  # 类名描述2
            classes_des3 = medical_des3[single_label]  # 类名描述3
            classes_des4 = medical_des4[single_label]  # 类名描述4
            classes_des5 = medical_des5[single_label]  # 类名描述5

            list_.append(data)
            list_.append(label)  # 类标签
            list_.append(class_dataset)  # 类名1
            list_.append(class_dataset2)  # 描述类名2
            list_.append(classes_des1)  # 描述1
            list_.append(classes_des2)  # 描述2
            list_.append(classes_des3)  # 描述3
            list_.append(classes_des4)  # 描述4
            list_.append(classes_des5)  # 描述5

        return list_

    def __len__(self):
        return len(self.sub_dataloader)


class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity):
        self.sub_meta = sub_meta
        self.cl = cl
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, i):
        image = self.sub_meta[i]

        img = self.transform(image)
        target = self.target_transform(self.cl)

        return img, target

    def __len__(self):
        return len(self.sub_meta)


class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes



    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]

    def _image_to_tensor(self, image):
        """将PIL图像转换为Tensor"""
        # 假设你有一个转换函数，或者直接将图片转换为Tensor
        # 这里用ToTensor()作为例子，如果你已经有其他转换，则根据需要修改
        transform = transforms.ToTensor()
        image_tensor = transform(image)
        return image_tensor


class TransformLoader:
    def __init__(self, image_size, 
                 normalize_param = dict(mean= [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                 jitter_param = dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param

    def parse_transform(self, transform_type):
        if transform_type == 'ImageJitter':
            method = ImageJitter(self.jitter_param)
            return method
        # 其他转换保持不变
        elif transform_type == 'RandomResizedCrop':
            return getattr(transforms, transform_type)(self.image_size)
        elif transform_type == 'CenterCrop':
            return getattr(transforms, transform_type)(self.image_size)
        elif transform_type == 'Scale':  # 修改这里

            return transforms.Resize([int(self.image_size * 1.15)])


        elif transform_type == 'Normalize':
            return getattr(transforms, transform_type)(**self.normalize_param)
        else:
            return getattr(transforms, transform_type)()
    def get_composed_transform(self, aug = True):
        if aug:
            transform_list = ['RandomResizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Scale','CenterCrop', 'ToTensor', 'Normalize']
        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

class Eposide_DataManager():
    def __init__(self, data_path, num_class, image_size, n_way=5, n_support=1, n_query=15, n_eposide=1):
        super(Eposide_DataManager, self).__init__()
        self.data_path = data_path
        self.num_class = num_class
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_eposide = n_eposide
        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, aug): #parameters that would change on train/val set

        transform = self.trans_loader.get_composed_transform(aug)
        #数据集
        dataset = SetDataset(self.data_path, self.num_class, self.batch_size, transform)
        #设置如何采样?
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide)
        data_loader_params = dict(batch_sampler=sampler, num_workers=0, pin_memory=True)

        #采样结果
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        # data_loader = torch.utils.data.DataLoader(dataset,batch_size=1, **data_loader_params)

        return data_loader




        

        
