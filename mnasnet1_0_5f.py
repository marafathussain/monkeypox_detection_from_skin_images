import sys
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import os
import shutil    
import torch.nn as nn

pin_memory = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    dtype = torch.cuda.FloatTensor

    for fold in range(5):
        root = f'/home/marafath/projects/rrg-hamarneh/marafath/data/pox_data/fold_{fold+1}'
        
        train_dir = os.path.join(root, 'training')
        val_dir = os.path.join(root, 'validation')

        # Define transforms
        train_transforms = transforms.Compose([
            transforms.ToTensor(),            
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
          ])

        val_transforms = transforms.Compose([
            transforms.ToTensor(),            
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


        # create a training data loader
        train_ds = ImageFolder(train_dir, transform=train_transforms)
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=1, pin_memory=pin_memory)

        # create a validation data loader
        val_ds = ImageFolder(val_dir, transform=val_transforms)
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=1, pin_memory=pin_memory)


        device = torch.device('cuda:0')
        model = models.mnasnet1_0(pretrained=True)
        num_classes = len(train_ds.classes)
        model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        model.type(dtype)
        loss_function = torch.nn.CrossEntropyLoss().type(dtype)

        for param in model.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(model.parameters(), 1e-3)

        # start a typical PyTorch training
        val_interval = 1
        best_metric = -1
        best_metric_epoch = -1
        epoch_loss_values = list()
        metric_values = list()
        writer = SummaryWriter()
        epc = 100 # Number of epoch
        for epoch in range(epc):
            print('-' * 10)
            print('epoch {}/{}'.format(epoch + 1, epc))
            model.train()
            epoch_loss = 0
            step = 0
            for batch_data in train_loader:
                step += 1
                inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_len = len(train_ds) // train_loader.batch_size
                print('{}/{}, train_loss: {:.4f}'.format(step, epoch_len, loss.item()))
                writer.add_scalar('train_loss', loss.item(), epoch_len * epoch + step)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            print('epoch {} average loss: {:.4f}'.format(epoch + 1, epoch_loss))

            if (epoch + 1) % val_interval == 0:
                model.eval()
                with torch.no_grad():
                    num_correct = 0.
                    metric_count = 0
                    best_predicted = np.array([])
                    actual_label = np.array([])
                    for val_data in val_loader:
                        val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                        val_outputs = model(val_images)
                        value = torch.eq(val_outputs.argmax(dim=1), val_labels)
                        metric_count += len(value)
                        num_correct += value.sum().item()
                        best_predicted = np.append(best_predicted, val_outputs.argmax(dim=1).detach().cpu().numpy()) 
                        actual_label = np.append(actual_label, val_labels.cpu().detach().numpy())     
                    metric = num_correct / metric_count
                    metric_values.append(metric)
                    if metric > best_metric:
                        best_metric = metric
                        best_metric_epoch = epoch + 1
                        torch.save(model.state_dict(), f'/home/marafath/projects/rrg-hamarneh/marafath/data/pox_data/pox_saved_model/mnasnet1_0_f{fold+1}.pth')
                        np.save(f'predicted_mnasnet1_0_f{fold+1}', best_predicted)
                        np.save(f'actual_mnasnet1_0_f{fold+1}', actual_label)
                        print('saved new best metric model')
                    print('current epoch: {} current accuracy: {:.4f} best accuracy: {:.4f} at epoch {}'.format(
                        epoch + 1, metric, best_metric, best_metric_epoch))
                    writer.add_scalar('val_accuracy', metric, epoch + 1)
        print('train completed, best_metric: {:.4f} at epoch: {}'.format(best_metric, best_metric_epoch))
        writer.close()

if __name__ == '__main__':
    main()