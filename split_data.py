# Import required libraries
import kagglehub
import os
import shutil
from sklearn.model_selection import train_test_split

# Download the dataset
path = kagglehub.dataset_download("mukaffimoin/potato-diseases-datasets")
print("Path to dataset files:", path)  # Should be: /root/.cache/kagglehub/datasets/mukaffimoin/potato-diseases-datasets/versions/3

# Define source and destination paths
source_dir = os.path.join(path, '')  # Update if needed based on your dataset structure
output_dir = "C:\\Users\\chris\\OneDrive\\Desktop\\SDSU\\Pavement Distress\\PotatoCNN"
train_dir = os.path.join(output_dir, 'train')
test_dir = os.path.join(output_dir, 'test')

# Create train and test directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Get category folders
categories = os.listdir(source_dir)

# Loop over each category to split data
for category in categories:
    # Path for each category
    category_path = os.path.join(source_dir, category)
    if os.path.isdir(category_path):  # Confirm it’s a folder
        images = os.listdir(category_path)

        # Split data into train and test sets (80-20 split)
        train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)

        # Create directories for each category in train and test folders
        os.makedirs(os.path.join(train_dir, category), exist_ok=True)
        os.makedirs(os.path.join(test_dir, category), exist_ok=True)

        # Move files to train and test directories
        for image in train_images:
            shutil.copy(os.path.join(category_path, image), os.path.join(train_dir, category, image))
        for image in test_images:
            shutil.copy(os.path.join(category_path, image), os.path.join(test_dir, category, image))

print("Data split into training and testing sets successfully.")