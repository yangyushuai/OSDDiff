from diffusion_utils import *
from noise_estimation_unet import NEU
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from data_loader import SARDataLoader
from metrics import *


def main():
    test_data = SARDataLoader("./data/test", data ="palsar", train=False)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)

    model = NEU()
    model = model.to(device)
    model.load_state_dict(torch.load('msd/palsar_best.pth'))  # 加载训练好的模型

    DS_1, DS_2, DS_3, DS_4, DS_5, DS_6, DS_7, DS_8, DS_9, DS_10 = [], [], [], [], [], [], [], [], [], []
    HD95_1, HD95_2, HD95_3, HD95_4, HD95_5, HD95_6, HD95_7, HD95_8, HD95_9, HD95_10 = [], [], [], [], [], [], [], [], [], []
    Pre_1, Pre_2, Pre_3, Pre_4, Pre_5, Pre_6, Pre_7, Pre_8, Pre_9, Pre_10 = [], [], [], [], [], [], [], [], [], []
    ACC_1, ACC_2, ACC_3, ACC_4, ACC_5, ACC_6, ACC_7, ACC_8, ACC_9, ACC_10 = [], [], [], [], [], [], [], [], [], []

    with torch.no_grad():
        model.eval()
        for idx, (x, label) in enumerate(test_data_loader):
            x, label = x.to(device), label.to(device)

            img_sum = torch.zeros_like(x)
            for i in range(ensembles_num):
                img = torch.randn_like(x).to(device)
                for t in tqdm(reversed(range(0, all_time_steps))):
                    img = p_sample(model, img, x, torch.full((batch_size,), t).long(), t)

                img = torch.where(img > 0.5, 1., 0.)
                label = torch.where(label > 0.5, 1., 0.)

                diceScore = compute_dice_score(img, label)
                print("i={},Dice={}".format(i, diceScore))

                img_sum += img

                if i == 0:  # n=1
                    img = img_sum / (i + 1)
                    img = torch.where(img > 0.5, 1., 0.)
                    diceScore, hd95, precision, acc = calculate_metrics(img, label)
                    DS_1.append(diceScore)
                    HD95_1.append(hd95)
                    Pre_1.append(precision)
                    ACC_1.append(acc)
                    print("{},{},{},{},{} \n".format(idx, diceScore, hd95, precision, acc))

                if i == 1:  # n=2
                    img = img_sum / (i + 1)
                    img = torch.where(img > 0.5, 1., 0.)
                    diceScore, hd95, precision, acc = calculate_metrics(img, label)
                    DS_2.append(diceScore)
                    HD95_2.append(hd95)
                    Pre_2.append(precision)
                    ACC_2.append(acc)
                    print("{},{},{},{},{} \n".format(idx, diceScore, hd95, precision, acc))

                if i == 2:  # n=3
                    img = img_sum / (i + 1)
                    img = torch.where(img > 0.5, 1., 0.)
                    diceScore, hd95, precision, acc = calculate_metrics(img, label)
                    DS_3.append(diceScore)
                    HD95_3.append(hd95)
                    Pre_3.append(precision)
                    ACC_3.append(acc)
                    print("{},{},{},{},{} \n".format(idx, diceScore, hd95, precision, acc))

                if i == 3:  # n=4
                    img = img_sum / (i + 1)
                    img = torch.where(img > 0.5, 1., 0.)
                    diceScore, hd95, precision, acc = calculate_metrics(img, label)
                    DS_4.append(diceScore)
                    HD95_4.append(hd95)
                    Pre_4.append(precision)
                    ACC_4.append(acc)
                    print("{},{},{},{},{} \n".format(idx, diceScore, hd95, precision, acc))

                if i == 4:  # n=5
                    img = img_sum / (i + 1)
                    img = torch.where(img > 0.5, 1., 0.)
                    diceScore, hd95, precision, acc = calculate_metrics(img, label)
                    DS_5.append(diceScore)
                    HD95_5.append(hd95)
                    Pre_5.append(precision)
                    ACC_5.append(acc)
                    print("{},{},{},{},{} \n".format(idx, diceScore, hd95, precision, acc))

                if i == 5:  # n=6
                    img = img_sum / (i + 1)
                    img = torch.where(img > 0.5, 1., 0.)
                    diceScore, hd95, precision, acc = calculate_metrics(img, label)
                    DS_6.append(diceScore)
                    HD95_6.append(hd95)
                    Pre_6.append(precision)
                    ACC_6.append(acc)
                    print("{},{},{},{},{} \n".format(idx, diceScore, hd95, precision, acc))

                if i == 6:  # n=7
                    img = img_sum / (i + 1)
                    img = torch.where(img > 0.5, 1., 0.)
                    diceScore, hd95, precision, acc = calculate_metrics(img, label)
                    DS_7.append(diceScore)
                    HD95_7.append(hd95)
                    Pre_7.append(precision)
                    ACC_7.append(acc)
                    print("{},{},{},{},{} \n".format(idx, diceScore, hd95, precision, acc))

                if i == 7:  # n=8
                    img = img_sum / (i + 1)
                    img = torch.where(img > 0.5, 1., 0.)
                    diceScore, hd95, precision, acc = calculate_metrics(img, label)
                    DS_8.append(diceScore)
                    HD95_8.append(hd95)
                    Pre_8.append(precision)
                    ACC_8.append(acc)
                    print("{},{},{},{},{} \n".format(idx, diceScore, hd95, precision, acc))

                if i == 8:  # n=9
                    img = img_sum / (i + 1)
                    img = torch.where(img > 0.5, 1., 0.)
                    diceScore, hd95, precision, acc = calculate_metrics(img, label)
                    DS_9.append(diceScore)
                    HD95_9.append(hd95)
                    Pre_9.append(precision)
                    ACC_9.append(acc)
                    print("{},{},{},{},{} \n".format(idx, diceScore, hd95, precision, acc))

                if i == 9:  # n=10
                    img = img_sum / (i + 1)
                    img = torch.where(img > 0.5, 1., 0.)
                    diceScore, hd95, precision, acc = calculate_metrics(img, label)
                    DS_10.append(diceScore)
                    HD95_10.append(hd95)
                    Pre_10.append(precision)
                    ACC_10.append(acc)
                    print("{},{},{},{},{} \n".format(idx, diceScore, hd95, precision, acc))


if __name__ == '__main__':
    main()
