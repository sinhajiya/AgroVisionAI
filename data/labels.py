import os

data_dir = './dataset/archive(1)/plantvillage dataset/color'
label = []
image_path = []
folds = os.listdir(data_dir)
for fold in folds :
    folder_path = os.path.join(data_dir , fold)
    imgs = os.listdir(folder_path)
    for img in imgs : 
        img_path = os.path.join(folder_path , img)
        label.append(fold)
        image_path.append(img_path)

print("Total images:", len(image_path))
print("Total labels:", len(label))
# print(label)