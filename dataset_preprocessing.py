from torchvision import transforms, datasets
from torchvision.utils import save_image
import os

root_dir = 'D://Project//dataset'

dataset = datasets.ImageFolder(root = root_dir)
tensor_transform = transforms.Compose([transforms.ToTensor()])

def get_norm_values(image):
    mean, std = image.mean([1, 2]), image.std([1, 2])
    return mean, std

def apply_transformations(image, normalize = True):
    if normalize:
        image = tensor_transform(image)
        mean, std = get_norm_values(image)
        print(mean, std)
        my_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation((-45, 45)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        image = my_transforms(image)
        return image
    else:
        image = tensor_transform(image)
        my_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation((-45, 45)),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])
        image = my_transforms(image)
        return image

dest = 'D:\\Project\\new_dataset'
classes = dataset.classes

def completed(arr):
    for i in arr:
        if i != 0:
            return False
    return True

# D represents required number of samples of each expression
def process_dir(D = 600, normalize = True):
    imnum = 0
    cur_count = [0 for i in range(len(classes))]
    label = 0
    for folder in classes:
        path = root_dir + '\\' + folder
        for file in os.listdir(path):
            cur_count[label] += 1
        label += 1
    
    rem = [600 - cur_count[i] for i in range(len(classes))]
    
    for folder in classes:
        if not os.path.exists(dest + '\\' + folder):
            os.makedirs(dest + '\\' + folder)
    
    iteration = 0
    while True:
        iteration += 1
        if iteration % 10 == 0:
            print(rem)
        if completed(rem):
            break                     
        for image, label in dataset:
            if rem[label] == 0:
                continue
            image = apply_transformations(image, False)
            folder = classes[label]
            save_image(image, dest + '\\' + folder + '\\' + 'image' + str(imnum) + '.png')
            imnum += 1
            rem[label] -= 1
    print('saved {} images'.format(imnum))
    save_original_images()
       
    
def save_original_images():
    imnum = 0
    for image, label in dataset:
        image = tensor_transform(image)
        folder = classes[label]
        save_image(image, dest + '\\' + folder + '\\' + 'original_image' + str(imnum) + '.png')
        imnum += 1
    print('original images saved {}'.format(imnum))

process_dir(normalize = False)