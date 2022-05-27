from data.dataset import ColorHintDataset
import torch
import torch.utils.data as data
import cv2
import tqdm
import os
from data.transform import tensor2im
from model.res_unet.res_unet import ResUnet
from model.res_unet.res_unet_plus import ResUnetPlusPlus
from model.res_unet.unet import UNet
from model.res_unet.ra_unet import ResAttdUNet


def test():
    device = "cpu"
    if torch.cuda.is_available():
      device = "cuda:0"
      print('device 0 :', torch.cuda.get_device_name(0))
    # Change to your data root directory
    root_path = "./cv_project"
    check_point = './checkpoints/best_model.pt'

    # Depend on runtime setting
    use_cuda = True

    test_dataset = ColorHintDataset(root_path, 256, "test")

    dataloaders = {}
    dataloaders['test'] = torch.utils.data.DataLoader(test_dataset, shuffle=False)
    print('test dataset: ', len(test_dataset))


    # state_dict = torch.load(check_point)
    model = ResAttdUNet().to(device)

    model.load_state_dict(torch.load(check_point))

    os.makedirs('outputs/test', exist_ok=True)

    model.eval()
    for i, data in enumerate(tqdm.tqdm(dataloaders['test'])):
        if use_cuda:
            l = data["l"].to(device)
            hint = data["hint"].to(device)
            mask = data["mask"].to(device)
        else:
            l = data["l"]
            hint = data["hint"]
            mask = data["mask"]

        file_name = data["file_name"]

        hint_image = torch.cat((l, hint), dim=1)

        output_hint = model(hint_image)
        out_hint_np = tensor2im(output_hint)
        output_bgr = cv2.cvtColor(out_hint_np, cv2.COLOR_LAB2BGR)

        cv2.imwrite("outputs/test/"+file_name, output_bgr)
