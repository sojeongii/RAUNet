import os
from data.dataset import ColorHintDataset
import torch.utils.data as data
import torch
import cv2
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from model.res_unet.res_unet import ResUnet
from model.res_unet.res_unet_plus import ResUnetPlusPlus
from model.res_unet.unet import UNet
from model.res_unet.ra_unet import ResAttdUNet
import matplotlib.image as img
import copy, time

def main():
    # Change to your data root directory
    root_path = "./cv_project"
    os.makedirs('checkpoints/', exist_ok=True)
    check_path = './checkpoints/'
    # Depend on runtime setting
    use_cuda = False

    train_dataset = ColorHintDataset(root_path, 256, "train")
    val_dataset = ColorHintDataset(root_path, 256, "val")

    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    dataloaders['valid'] = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=True)


    print('train dataset: ', len(train_dataset))
    print('validation dataset: ', len(val_dataset))

    models = {'ResUnet': ResUnet(3), 'ResUnetPlusPlus': ResUnetPlusPlus(3), 'UNet': UNet(), 'ResAttdUnet' : ResAttdUNet()}
    model = models['ResAttdUNet']
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 999
    train_loss, train_acc, valid_loss, valid_acc = [], [], [], []
    epochs = 1

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss, running_corrects, num_cnt = 0.0, 0, 0

            # Iterate over data.
            for i, data in enumerate(tqdm.tqdm(dataloaders[phase])):
                if use_cuda:
                    l = data["l"].to('cuda')
                    ab = data["ab"].to('cuda')
                    hint = data["hint"].to('cuda')
                else:
                    l = data["l"]
                    ab = data["ab"]
                    hint = data["hint"]

                gt_image = torch.cat((l, ab), dim=1)
                hint_image = torch.cat((l, hint), dim=1)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(hint_image)
                    loss = criterion(outputs, gt_image)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.detach().cpu().item()
                num_cnt += len(data)

            # if phase == 'train':
            #     exp_lr_scheduler.step()

            epoch_loss = float(running_loss / num_cnt)

            if phase == 'train':
                train_loss.append(epoch_loss)
            else:
                valid_loss.append(epoch_loss)
            print(' {} Loss: {:.2f} '.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'valid' and epoch_loss < best_loss:
                best_idx = epoch
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                best_model_wts = copy.deepcopy(model.module.state_dict())
                # Save model & checkpoint
                state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }

                torch.save(model, check_path + str(best_loss) + '_model.pt')
                torch.save(state, check_path + str(best_loss) + '_checkpoint.pt')

                print('==> best model saved - %d / %.1f' % (best_idx, best_loss))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: %d - %.1f' % (best_idx, best_loss))

    fig, ax1 = plt.subplots()
    ax1.plot(train_loss, 'g-', label='Train_loss')
    ax1.plot(valid_loss, 'k-', label='Valid_loss')
    plt.plot(best_idx, valid_loss[best_idx], 'ro')
    ax1.set_ylabel('loss', color='k')
    ax1.tick_params('y', colors='k')
    plt.legend(loc='best')
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()