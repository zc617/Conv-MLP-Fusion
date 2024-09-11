import argparse
import torch
from models import fusion_model
from input_data import ImageDataset
from uitils import *
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(6)

plt.rcParams['font.family'] = ['serif']
plt.rcParams['font.sans-serif'] = ['Times New Roman']

parser = argparse.ArgumentParser()
parser.add_argument("--infrared_dataroot", default="../ir/", type=str)
parser.add_argument("--visible_dataroot", default="../vi/", type=str)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--image_size", type=int, default=(128, 128)) 
parser.add_argument("--epoch", type=int, default= 80)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/")
parser.add_argument('--loss_weight', default='[5, 25, 1]', type=str,metavar='N', help='loss weight')
   #  2 10 3
if __name__ == "__main__":
    opt = parser.parse_args()
    if not os.path.exists(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir)

    writer = SummaryWriter('./runs/logdir')

    net = fusion_model.FusionNet().to(device)

    optim = torch.optim.Adam(filter(lambda p: p.requires_grad,net.parameters()),lr=opt.lr)
    train_datasets = ImageDataset(opt.infrared_dataroot, opt.visible_dataroot, opt.image_size)
    lens = len(train_datasets)
    print('data lens', lens)
    log_file = './log_dir'
    dataloader = torch.utils.data.DataLoader(train_datasets,batch_size=opt.batch_size,shuffle=True)
    runloss = 0.
    total_params = sum(p.numel() for p in net.parameters())
    print('total parameters:', total_params)
    global_step = 0
    w1_vis = 1
    i = 0
    for epoch in range(opt.epoch):
        if epoch % 10 == 0 and epoch > 1:
           opt.lr=0.1*opt.lr
        net.train()
        num=0
    
        for index, data in enumerate(dataloader):
     
            nc, c, h, w = data[0].size()
            nc2, c2, h2, w2 = data[1].size()
        
            infrared = data[0].to(device)
            visible = data[1].to(device)
            fused_img, disp_ir_feature, disp_vis_feature = net(infrared, visible)
            fused_img = clamp(fused_img)
            int_loss = Int_Loss(fused_img, visible, infrared, w1_vis).to(device)
            gradient_loss = gradinet_Loss(fused_img, visible, infrared).to(device)
            t1, t2, t3 = eval(opt.loss_weight)
            loss = t1 * int_loss + t2 * gradient_loss # + t3 *(1-ssim_loss) 
            runloss += loss.item()
            if epoch == 0 and index == 0:
                writer.add_graph(net, (infrared, visible))
            if index % 200 == 0:  #
                writer.add_scalar('training loss', runloss / 200, epoch * len(dataloader) + index)
                runloss = 0.

            optim.zero_grad()
            loss.backward()
            optim.step()
        if epoch % 1 == 0:
            print('write_data, epoch=', epoch)
            print(
                'epoch [{}/{}], images [{}/{}], Int loss is {:.5}, gradient loss is {:.5}, total loss is  {:.5}, lr: {}'.
                format(epoch + 1, opt.epoch, (index + 1) * data[0].shape[0], lens, int_loss.item(),
                       gradient_loss.item(), loss.item(), opt.lr))
          
            writer.add_images('IR_images', infrared, dataformats='NCHW')
            writer.add_images('VIS_images', visible, dataformats='NCHW')
            writer.add_images('Fusion_images', fused_img, dataformats='NCHW')
    writer.close()
    torch.save(net.state_dict(), './checkpoints/fusion_mlp_n_'+str(epoch+1)+'.pth'.format(opt.lr, log_file[2:]))
    print('training is complete!')

    
    
    
    #fusion_grad_mlp_n_conv4_01   超参1 10 ,引入深度可分离卷积。重建的多次MLP先还原成一次，测试效果不佳。
    #fusion_grad_mlp_n_conv4_01   继续修改，修改mlp中的norm残差，效果不好，可能的原因是深度分离卷积加在MLP中间，还是对channel的操作，
    # fusion_grad_mlp_n_conv4_01  网络改成CNN+CNN+ （MLP+深度卷积）*num 的结构，其中MLP的残差再深度卷积前面,效果比较好。可以作为测试定稿之一。
    # fusion_grad_mlp_n_conv4_01  网络改成CNN+CNN+ （MLP+深度卷积）*num 的结构，目前发现一个问题，部分图像在黑色部分易出现白色（例如天空中的电线）