
import torch
import torch.nn as nn
import torch.utils.data as dt
from carvana_dataset import CarvanaDataset
from model import SegmenterModel
from torch.autograd import Variable
import torch.optim as optim
from tensorboardX import SummaryWriter
import os
from tqdm import *
import numpy as np
import pickle

LR = 1e-3
useCuda =True
n_epoch = 100
log = './log/'
train = './data/train/'
train_masks = './data/train_masks/'
test = './data/test/'
test_masks = './data/test_masks'

if os.path.exists(log) == False:
    os.mkdir(log)
tb_writer = SummaryWriter(log_dir='log')

if __name__ == '__main__':
    m = SegmenterModel()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(m.parameters(), lr=LR)

    if useCuda == True:
        m = m.cuda()
        criterion= criterion.cuda()

    ds = CarvanaDataset(train, train_masks)
    ds_test = CarvanaDataset(test, test_masks)

    dl      = dt.DataLoader(ds, shuffle=True, num_workers=4, batch_size=5)
    dl_test = dt.DataLoader(ds_test, shuffle=False, num_workers=4, batch_size=5)

    global_iter = 0
    for epoch in range(0, n_epoch):
        print ("Current epoch: ", epoch)
        epoch_loss = 0
        m.train(True)
        for iter, (i, t) in enumerate(tqdm( dl)):
            if useCuda :
                i = i.cuda()
                t = t.cuda()
            o = m(i)
            tmp = torch.cat((t, 1.-t), dim=1)
            if useCuda :
                tmp = tmp.cuda()
            loss = criterion(o, tmp)
            loss.backward()
            optimizer.step()

            global_iter += 1
            epoch_loss += loss.item()

        epoch_loss = epoch_loss / float(len(ds))
        print ("Epoch loss", epoch_loss)
        tb_writer.add_scalar('Loss/Train', epoch_loss, epoch)
        '''
        if epoch==10:
            for g in optimizer.param_groups:
                g['lr'] = LR/5
        if epoch==70:
            for g in optimizer.param_groups:
                g['lr'] = LR/5
        '''
        print ("Make test")
        test_loss = 0
        m.eval()
        
       

        tb_out = np.random.choice(range(0, len(dl_test)), 3)
        for iter, (i, t) in enumerate(tqdm(dl_test)):
            with torch.no_grad():
                if useCuda :
                    i = i.cuda()
                    t = t.cuda()
                o = m(i)
                tmp = torch.cat((t, 1.-t), dim=1)
                if useCuda :
                    tmp = tmp.cuda()
                loss = criterion(o, tmp)
                test_loss += loss.item()

                for k, c in enumerate(tb_out):
                    if c == iter:
                        tb_writer.add_image('Image/Test_input_%d'%k,  i[0].cpu(), epoch)  # Tensor
                        tb_writer.add_image('Image/Test_target_%d'%k, t[0].cpu(), epoch)  # Tensor
                        tb_writer.add_image('Image/Test_output_%d'%k, o.chunk(2,dim=1)[0][0].cpu(), epoch)  # Tensor

        test_loss = test_loss / float(len(ds_test))
        print ("Test loss", test_loss)
        tb_writer.add_scalar('Loss/Test', test_loss, epoch)
        torch.save(m.state_dict(), 'model_state_dict_CrossEntropyLoss.pth.tar')