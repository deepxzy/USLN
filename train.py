import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
from model import USLN
from SegDataset import SegDataset
from loss import Combinedloss
########################################################
num_workers = 0 if sys.platform.startswith('win32') else 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
#############################################################
torch.cuda.set_device(1)  # 指定GPU运行
if __name__ == "__main__":



    Init_Epoch = 0
    Final_Epoch = 100
    batch_size = 10
    lr = 1e-2

    model = USLN()
    save_model_epoch = 1


    model = model.to(device)
    data_train = SegDataset('train')
    data_test = SegDataset('val')

    myloss = Combinedloss().to(device)
    if True:
        batch_size = batch_size
        start_epoch = Init_Epoch
        end_epoch = Final_Epoch

        optimizer       = optim.Adam(model.train().parameters(), lr=lr, weight_decay = 5e-4)

        lr_scheduler    = optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.94)




        for epo in range(start_epoch, end_epoch):
            train_loss = 0
            model.train()  # 启用batch normalization和drop out



            train_iter = torch.utils.data.DataLoader(data_train, batch_size, shuffle=True,
                                                     drop_last=True, num_workers=num_workers,pin_memory=True)
            test_iter = torch.utils.data.DataLoader(data_test, batch_size, drop_last=True,
                                                    num_workers=num_workers,pin_memory=True)

            for index, (bag, bag_msk) in enumerate(train_iter):

                bag = bag.to(device)
                bag_msk = bag_msk.to(device)
                optimizer.zero_grad()
                output = model(bag)

                loss = myloss(output, bag_msk)
                loss.backward()
                iter_loss = loss.item()

                train_loss += iter_loss
                optimizer.step()

                if np.mod(index, 15) == 0:
                    print('epoch {}, {}/{},train loss is {}'.format(epo, index, len(train_iter), iter_loss))

            # 验证
            test_loss = 0
            model.eval()
            with torch.no_grad():
                for index, (bag, bag_msk) in enumerate(test_iter):
                    bag = bag.to(device)
                    bag_msk = bag_msk.to(device)

                    optimizer.zero_grad()
                    output = model(bag)

                    loss = myloss(output, bag_msk)
                    # loss = criterion(output, torch.argmax(bag_msk, axis=1))
                    iter_loss = loss.item()

                    test_loss += iter_loss

            print('<---------------------------------------------------->')
            print('epoch: %f' % epo)
            print('epoch train loss = %f, epoch test loss = %f'
                  % (train_loss / len(train_iter), test_loss / len(test_iter)))

            lr_scheduler.step()
            # 每5个epoch存储一次模型
            if np.mod(epo, save_model_epoch) == 0:
                # 只存储模型参数
                torch.save(model.state_dict(), 'logs/ep%03d-loss%.3f-val_loss%.3f.pth' % (
                    (epo + 1), (100*train_loss / len(train_iter)), (100*test_loss / len(test_iter)))
                           )
                print('saveing checkpoints/model_{}.pth'.format(epo))



