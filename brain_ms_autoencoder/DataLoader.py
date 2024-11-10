from sklearn.model_selection import train_test_split
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import nibabel as nib
import pathlib

voxeldir_path = pathlib.Path('C:\\Users\\gianp\\OneDrive\\Desktop\\DL-projects-repo\\brain-ms-autoencoder\\data\\raw\\FlairTrainSet\\Flair')
file_list = sorted([str(path) for path in voxeldir_path.glob('*.nii*')])

train_path_list, test_path_list = train_test_split(file_list, test_size=0.20, random_state=15)
train_path_list, valid_path_list = train_test_split(train_path_list, test_size=0.20, random_state=15)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: F.rotate(x, 270)),
])

class FlairDataset(Dataset):
    def __init__(self, path_list, min_slice, max_slice, transform):
        self.path_list = path_list
        self.transform = transform
        self.min_slice = min_slice
        self.max_slice = max_slice
        self.num_slices_per_voxel = max_slice - min_slice
        self.total_images = len(path_list) * self.num_slices_per_voxel

    def __getitem__(self, index):
        file_idx = index // self.num_slices_per_voxel
        slice_idx = index % self.num_slices_per_voxel + self.min_slice 

        voxel = nib.load(self.path_list[file_idx]).get_fdata()
        image_slice = voxel[:, :, slice_idx]

        if self.transform is not None:
            image_slice = self.transform(image_slice)

        return image_slice

    def __len__(self):
        return self.total_images
    
dim_voxel = 176
min_slice = 110
max_slice = 150

trainDataset = FlairDataset(train_path_list, min_slice, max_slice, transform)
trainLoader = DataLoader(trainDataset, batch_size=64, shuffle=True)

"""img_batch = next(iter(trainLoader))  # Retrieve a batch of 64 images

# Create an 8x8 grid
fig, axes = plt.subplots(8, 8, figsize=(12, 12))
axes = axes.ravel()  # Flatten the array of axes for easy iteration

# Plot each image in the grid
for i in range(64):
    axes[i].imshow(img_batch[i].squeeze(), cmap='gray')  # Use squeeze if there's a single color channel
    axes[i].axis('off')  # Turn off axis

plt.tight_layout()
plt.show()"""