import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn import metrics
from torchvision import transforms
import numpy as np
from PIL import Image
import os
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

random_state = 123435647


class HASYDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        """
        Args:
            dataframe (DataFrame): DataFrame with paths and labels.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.dataframe.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label


# Load data
root_dir = "/home/jules/Dokumente/symb-rec/"
data = pd.read_csv(os.path.join(root_dir, "hasy-data-labels.csv"))
# data = data[:1000]

symbols_df = pd.read_csv(os.path.join(root_dir, "symbols.csv"))

# Adjust symbol_id to start from 1
symbol_id_mapping = {
    old_id: new_id for new_id, old_id in enumerate(sorted(data["symbol_id"].unique()), 0)
}
symbol_id_back_mapping = {
    new_id: symbols_df[symbols_df["symbol_id"] == old_id]["latex"].iat[0]
    for new_id, old_id in enumerate(sorted(data["symbol_id"].unique()), 0)
}
# print(symbol_id_back_mapping)
data["symbol_id"] = data["symbol_id"].map(symbol_id_mapping)


label_list = data["symbol_id"].unique()
num_classes = len(label_list)
image_size = 24

batch_size = 128
log_interval = 100  # How often to display (batch) loss during training
epochs = 20  # Number of epochs
learningRate = 0.001  # learning rate
learningMomentum = 0.9  # momentum of SGD


# Split data
train_data, test_data = train_test_split(data, test_size=0.2, random_state=random_state)
test_data, val_data = train_test_split(test_data, test_size=0.5, random_state=random_state)


# Transformations / normalization
transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((image_size, image_size)),  # Resize
        transforms.ToTensor(),  # Convert to a PyTorch tensor
    ]
)

# Create dataset objects
train_dataset = HASYDataset(train_data, root_dir=root_dir, transform=transform)
val_dataset = HASYDataset(val_data, root_dir=root_dir, transform=transform)
test_dataset = HASYDataset(test_data, root_dir=root_dir, transform=transform)

# Create DataLoaders
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(image_size * image_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = Model()

# Create an instance of "torch.optim.SGD"
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=learningRate,
    momentum=learningMomentum,
)


def loss_fn(prediction, labels):
    """Returns softmax cross entropy loss."""
    return F.cross_entropy(input=prediction, target=labels)


def run_epoch(model, epoch, data_loader, optimizer, is_training, log_interval):
    """
    Args:
        model        (obj): The neural network model
        epoch        (int): The current epoch
        data_loader  (obj): A pytorch data loader "torch.utils.data.DataLoader"
        optimizer    (obj): A pytorch optimizer "torch.optim"
        is_training (bool): Whether to use train (update) the model/weights or not.
        log_interval (int): Interval to log

    Intermediate:
        totalLoss: (float): The accumulated loss from all batches.
                            Hint: Should be a numpy scalar and not a pytorch scalar

    Returns:
        loss_avg         (float): The average loss of the dataset
        accuracy         (float): The average accuracy of the dataset
        confusion_matrix (float): A 10x10 matrix
    """

    if is_training == True:
        model.train()
    else:
        model.eval()

    total_loss = 0
    correct = 0
    confusion_matrix = np.zeros(shape=(num_classes, num_classes))
    for batch_idx, data_batch in enumerate(data_loader):
        images = data_batch[0]
        labels = data_batch[1]

        if not is_training:
            with torch.no_grad():
                prediction = model.forward(images)
                loss = loss_fn(prediction, labels)
                total_loss += loss.item()

        elif is_training:
            prediction = model.forward(images)
            loss = loss_fn(prediction, labels)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update the number of correct classifications and the confusion matrix
        predicted_label = prediction.max(1, keepdim=True)[1][:, 0]
        correct += predicted_label.eq(labels).cpu().sum().numpy()
        confusion_matrix += metrics.confusion_matrix(
            labels.cpu().numpy(), predicted_label.cpu().numpy(), labels=label_list
        )

        # Print statistics
        # batchSize = len(labels)
        if batch_idx % log_interval == 0:
            print(f"Epoch={epoch} | {(batch_idx+1)/len(data_loader)*100:.2f}% | loss = {loss:.5f}")

    loss_avg = total_loss / len(data_loader)
    accuracy = correct / len(data_loader.dataset)
    confusion_matrix = confusion_matrix / len(data_loader.dataset)

    return loss_avg, accuracy, confusion_matrix


def plot_loss(train_loss, val_loss, train_acc, val_acc, root_dir):
    # Plot the loss and the accuracy in training and validation
    # plt.figure()
    plt.figure(figsize=(18, 16), dpi=80, facecolor="w", edgecolor="k")
    ax = plt.subplot(2, 1, 1)
    # plt.subplots_adjust(hspace=2)
    ax.plot(train_loss, "b", label="train loss")
    ax.plot(val_loss, "r", label="validation loss")
    ax.grid()
    plt.ylabel("Loss", fontsize=18)
    plt.xlabel("Epochs", fontsize=18)
    ax.legend(loc="upper right", fontsize=16)

    ax = plt.subplot(2, 1, 2)
    plt.subplots_adjust(hspace=0.4)
    ax.plot(train_acc, "b", label="train accuracy")
    ax.plot(val_acc, "r", label="validation accuracy")
    ax.grid()
    plt.ylabel("Accuracy", fontsize=18)
    plt.xlabel("Epochs", fontsize=18)
    val_acc_max = np.max(val_acc)
    val_acc_max_ind = np.argmax(val_acc)
    plt.axvline(x=val_acc_max_ind, color="g", linestyle="--", label="Highest validation accuracy")
    plt.title("Highest validation accuracy = %0.1f %%" % (val_acc_max * 100), fontsize=16)
    ax.legend(loc="lower right", fontsize=16)

    plt.savefig(os.path.join(root_dir, "loss.pdf"))


# train the model
train_loss = np.zeros(shape=epochs)
train_acc = np.zeros(shape=epochs)
val_loss = np.zeros(shape=epochs)
val_acc = np.zeros(shape=epochs)
train_confusion_matrix = np.zeros(shape=(num_classes, num_classes, epochs))
val_confusion_matrix = np.zeros(shape=(num_classes, num_classes, epochs))

for epoch in range(epochs):
    (
        train_loss[epoch],
        train_acc[epoch],
        train_confusion_matrix[:, :, epoch],
    ) = run_epoch(
        model,
        epoch,
        train_data_loader,
        optimizer,
        is_training=True,
        log_interval=log_interval,
    )

    val_loss[epoch], val_acc[epoch], val_confusion_matrix[:, :, epoch] = run_epoch(
        model,
        epoch,
        val_data_loader,
        optimizer,
        is_training=False,
        log_interval=log_interval,
    )
    plot_loss(train_loss, val_loss, train_acc, val_acc, root_dir)
    torch.save(model, os.path.join(root_dir, f"model-{epoch}"))


torch.save(model, os.path.join(root_dir, "model"))

label_list_back = [symbol_id_back_mapping[i] for i in label_list]

ind = np.argmax(val_acc)
class_accuracy = val_confusion_matrix[:, :, ind]
for ii in range(num_classes):
    acc = val_confusion_matrix[ii, ii, ind] / np.sum(val_confusion_matrix[ii, :, ind])
    print(f"Accuracy of {str(label_list_back[ii]).ljust(15)}: {acc*100:.01f}%")


epoch_step = 2
set_colorbar_max_percentage = 10

# Plot confusion matrices
ticks = list(range(0, num_classes))
gridspec_kwargs = dict(top=0.9, bottom=0.1, left=0.0, right=0.9, wspace=0.5, hspace=0.2)
for i in range(0, epochs, epoch_step):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 16), gridspec_kw=gridspec_kwargs)
    im = ax1.imshow(val_confusion_matrix[:, :, i] * 100)
    ax1.set_title(f"Validation: Epoch #{i}", fontsize=18)
    ax1.set_xticks(ticks=ticks)
    ax1.set_yticks(ticks=ticks)
    ax1.set_yticklabels(label_list_back)
    im.set_clim(0.0, set_colorbar_max_percentage)
    ax1.set_xticklabels(label_list_back, rotation=70)
    ax1.set_ylabel("Prediction", fontsize=16)
    ax1.set_xlabel("Ground truth", fontsize=16)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    f.colorbar(im, cax=cax, orientation="vertical")

    im = ax2.imshow(train_confusion_matrix[:, :, i] * 100)
    ax2.set_title(f"Train: Epoch #{i}", fontsize=18)
    ax2.set_xticks(ticks=ticks)
    ax2.set_yticks(ticks=ticks)
    ax2.set_yticklabels(label_list_back)
    im.set_clim(0.0, set_colorbar_max_percentage)
    ax2.set_xticklabels(label_list_back, rotation=70)
    ax2.set_ylabel("Prediction", fontsize=16)
    ax2.set_xlabel("Ground truth", fontsize=16)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    f.colorbar(im, cax=cax, orientation="vertical")
    plt.savefig(os.path.join(root_dir, f"confusion{i}.pdf"))
