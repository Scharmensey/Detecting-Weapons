##Scharmensey Fine

#---- Libraries ----------------------------------------------------------------------------------------------------------------------------------
from matplotlib import pyplot as plt
import cv2
import imghdr
import os
import numpy as np 

#pyTorch related
import torch
from torch import nn
import torchvision
from torchvision import transforms, datasets
import torch.optim as optim

#Evaulation
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
  

#--------------------------------------------------------------------------------------------------------------------------------------

#os.listdir*('data) --> list/ shows whats in that file
 

'''
for image_class in os.listdir(directory):
    for image in os.listdir(os.listdir(os.path.join(directory, image_class))):
        print(image)
   
img = cv2.imread(os.path.join(directory,'danger','304648681.jpg'))
plt.imshow(img)
#cve.ccvtColor(img, cv2.COLOR_BGR2RGB)  #print pic in color

os.listdir(os.path.join(directory, 'danger'))
'''


#----- Data Preparation (Removing weird file types ----------------------------------------------------------------------------------------------------------------------------------

directory = '/Users/scharmenseyfine/Documents/Projects/Detecting Weapons/data'
image_type = ['jpeg', 'jpg', 'bmp', 'png']     

#Removing weird image types accquired in data set
for image_class in os.listdir(directory): 
    class_directory = os.path.join(directory, image_class)
    
    if os.path.isdir(class_directory):  # Check if it's a directory
        for item in os.listdir(class_directory):
            item_path = os.path.join(class_directory, item)
            
            # Skip .DS_Store and non-image files
            if item == '.DS_Store' or not os.path.isfile(item_path):
                continue
            
            try:
                img = cv2.imread(item_path)
                tip = imghdr.what(item_path)
                if tip not in image_type:
                    print('Image not in image type list: {}'.format(item_path))
                    os.remove(item_path)
            except Exception as e:
                print('Issue with image: {}'.format(item_path))





#--- Loading/Processing Data ----------------------------------------------------------------------------------------------------------------------------------

# Define transformations to be applied to the images

# 1 - safe,   0 - weapon
data_transform = transforms.Compose([
    transforms.Resize((256, 256)),   # Resize images to 256x256
    transforms.CenterCrop(224),      # Crop the center 224x224 region
    transforms.ToTensor(),            # Convert images to PyTorch tensors, scales automatically (0-1)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image tensors
])

# Load the dataset using ImageFolder
dataset = datasets.ImageFolder(root=directory, transform=data_transform)

# Define the sizes of the training and testing sets
train_size = int(0.75 * len(dataset))  # 75% of the data for training
test_size = len(dataset) - train_size  # Remaining 25% for testing

# Split the dataset into training and testing sets
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

# Create data loaders for the training and testing sets
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)


#to view images
'''
import matplotlib.pyplot as plt
import numpy as np

count = 0
class_names = dataset.classes
fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(20, 20))  # Adjust ncols to print only 5 images
for batch_idx, (images, labels) in enumerate(train_loader):
    # Process your data here
    print(f'Batch {batch_idx}:')
    for i in range(len(images)):  # Iterate over the images in the batch
        ax[count].imshow(np.transpose(images[i], (1, 2, 0)))  # Transpose image tensor to (height, width, channels)
        class_name = class_names[labels[i]]  # Get class name from numerical label
        ax[count].set_title(f'{class_name}')  # Set the title as the class name
        ax[count].axis('off')
        count += 1
        if count == 5:
            break
    if count == 5:  # Stop iterating if 5 images have been printed
        break

# Remove extra subplots if fewer than 5 images are displayed
for i in range(count, 5):
    fig.delaxes(ax[i])

plt.show()
#Check by printing images out here:

'''
#----- Deep Learning Model  ---------------------------------------------------------------------------------------------------------------------------

class CNN(nn.Module):
        
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=0) # 1st convolutional lay=yer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) #1st pooling layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)# 2nd convolutional layyer
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) #2nd pooling layer
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=0)# 3rd convolutional layyer
        self.fc1 = nn.Linear(16 * 26 * 26, 256)  
        self.fc2 = nn.Linear(256, 1)  
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid() #output layer
        self.dropout = nn.Dropout(p=0.4)  # Dropout regularization layer 0.4 gave best results

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.pool2(x)
        #print(x.size())  
        x = x.view(-1, 16 * 26 * 26)  # Flattening
        x = self.relu(self.fc1(x))
        x = self.dropout(x) # help with the overfitting
        x = self.sigmoid(self.fc2(x)) #Output activation 
        return x


def train(model, epoch, train_loader, test_loader):
        
    criterion = nn.BCELoss()  # Loss function to evaluatve how off the true labels we are
   # optimizer = optim.SGD(model.parameters(), lr=0.005)  
    optimizer = optim.Adam(model.parameters(), lr=0.002) # Adam seems to work better
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)  # Decays learning rate by a factor of 0.1 every 5 epochs

    total_train_loss = []
    total_test_loss = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Iterate over batches of training data
        for inputs, labels in train_loader:
            
            optimizer.zero_grad()  # Zero the parameter gradients
            inputs, labels = inputs.to(device), labels.to(device)
            #print(inputs.shape)
            # Forward pass into model
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().unsqueeze(1))  
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        # Calculate average training loss for each epoch
        train_loss = running_loss / len(train_loader)
        total_train_loss.append(train_loss)

        
        # Validateing model
        model.eval()  
        test_loss = 0.0
        with torch.no_grad():  
            for inputs, labels in test_loader:
                
                #Forward pass into model
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                #Calculating loss
                loss2 = criterion(outputs, labels.float().unsqueeze(1)).item()
                test_loss += loss2
                
        # Calculate average validation loss for the epoch
        test_loss /= len(test_loader)
        total_test_loss.append(test_loss)
        
        # Print training and validation losses to view performance
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {test_loss:.4f}")
        
        # Adjusting learning rate
        scheduler.step()
    
    return model, total_train_loss, total_test_loss


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Set device model.to(device)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Defining model
model = CNN().to(device)

#Num of epochs to run
num_epochs = 6

model, train_loss, test_loss = train(model, num_epochs, train_loader, test_loader)
    

 #------ Performance Measures ---------------------------------------------------------------------------------------------------------------------------------

# predicting function 
def predict(model, data_loader):
    model.eval()  #
    predictions = []
    true_labels = []
    
    with torch.no_grad():  
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())  
            true_labels.append(labels.cpu().numpy())  
    predictions = np.concatenate(predictions)
    true_labels = np.concatenate(true_labels)
    return predictions, true_labels

# Train vs Test ROC curve to view over/underfitting
def plot_roc_curve(test_pred, test_labels, train_pred, train_labels):
    fpr_train, tpr_train, thresholds_train = roc_curve(train_labels, train_pred)
    roc_auc_train = auc(fpr_train, tpr_train)
    
    fpr_test, tpr_test, thresholds_test = roc_curve(test_labels, test_pred)
    roc_auc_test = auc(fpr_test, tpr_test)
    plt.figure()
    plt.plot(fpr_train, tpr_train, color='brown', lw=2, label='ROC training set curve %0.2f' % roc_auc_train)
    plt.plot(fpr_test, tpr_test, color='darkorange', lw=2, label='ROC testing set curve %0.2f' % roc_auc_test)
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()


def show_images_predictions(model, test_loader, num_images=5):
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # Get the predicted labels
            images = images.numpy()
            labels = labels.numpy()
            predicted = predicted.numpy()
            for j in range(num_images):
                plt.subplot(1, num_images, j+1)
                plt.imshow(np.transpose(images[j], (1, 2, 0)))
                plt.title(f"Real: {labels[j]}, Predicted: {predicted[j]}", size = 6.5)
                plt.axis('off')
            plt.show()
            break  # Display only the first batch


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# View sample image predictions
show_images_predictions(model, test_loader, num_images=5)


# Predictions 
train_predictions, train_true_labels = predict(model, train_loader)
test_predictions, test_true_labels = predict(model, test_loader)


# Confusion Matrix & classifcation Repoort
test_predicted_classes = (test_predictions > 0.5).astype(int)
conf_matrix = confusion_matrix(test_true_labels, test_predicted_classes)
class_report = classification_report(test_true_labels, test_predicted_classes)

print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Plotting ROC curve
plot_roc_curve(test_predictions, test_true_labels, train_predictions, train_true_labels)

# Plotting losses against iterations
plt.plot(train_loss,label='training set',c = 'Brown')
plt.plot(test_loss,label='validation set', c = 'orange')
plt.xlabel('Epoch',color = 'Black', fontweight='bold')
plt.ylabel("loss",color = 'Black', fontweight='bold')

plt.legend()
plt.show()





