import torch.nn as nn


def conv_bn_relu(in_planes, out_planes, kernel=3, stride=1, padding=1):
     net = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=1),
                         nn.BatchNorm2d(num_features=out_planes),
                         nn.ReLU(True))
     return net;

# input size Bx3x224x224
class SegmenterModel(nn.Module):
    def __init__(self, in_size=3, num_classes=2, d1=16, d2=32, d3=64):
        super(SegmenterModel, self).__init__()
        #downsampling
        self.conv1 = nn.Conv2d(in_size, d1, kernel_size=11, stride=2, padding=5)
        self.bn1 = nn.BatchNorm2d(d1)
        self.dropout1 = nn.Dropout2d(p=0.2)
        self.high_res1_down = conv_bn_relu(d1, d1)
        self.dropout2 = nn.Dropout2d(p=0.2)
        self.high_res2_down = conv_bn_relu(d1, d1)
        self.dropout3 = nn.Dropout2d(p=0.2)
        
        self.conv2 = nn.Conv2d(d1, d2, kernel_size=11, stride=2, padding=5)
        self.bn2 = nn.BatchNorm2d(d2)
        self.dropout4 = nn.Dropout2d(p=0.2)
        self.middle_res1_down = conv_bn_relu(d2, d2)
        self.dropout5 = nn.Dropout2d(p=0.2)
        self.middle_res2_down = conv_bn_relu(d2, d2)
        self.dropout6 = nn.Dropout2d(p=0.2)
        
        self.conv3 = nn.Conv2d(d2, d3, kernel_size=11, stride=2, padding=5)
        self.bn3 = nn.BatchNorm2d(d3)
        self.dropout7 = nn.Dropout(p=0.2)
        self.low_res1_down = conv_bn_relu(d3, d3)
        self.dropout8 = nn.Dropout(p=0.2)
        self.low_res2_down = conv_bn_relu(d3, d3)
        self.dropout9 = nn.Dropout2d(p=0.2)
        
        #upsampling
        self.deconv1 = nn.ConvTranspose2d(in_channels=d3, out_channels=d2, kernel_size=11, stride=2, padding=5, output_padding=1)
        self.bn4 = nn.BatchNorm2d(d2)
        self.dropout10 = nn.Dropout2d(p=0.2)
        self.middle_res1_up = conv_bn_relu(d2, d2)
        self.dropout11 = nn.Dropout2d(p=0.2)
        self.middle_res2_up = conv_bn_relu(d2, d2)
        self.dropout12 = nn.Dropout2d(p=0.2)
        
        self.deconv2 = nn.ConvTranspose2d(in_channels=d2, out_channels=d1, kernel_size=11, stride=2, padding=5, output_padding=1)
        self.bn5 = nn.BatchNorm2d(d1)
        self.dropout13 = nn.Dropout2d(p=0.2)
        self.high_res1_up = conv_bn_relu(d1, d1)
        self.dropout14 = nn.Dropout2d(p=0.2)
        self.high_res2_up = conv_bn_relu(d1, d1)
        self.dropout15 = nn.Dropout2d(p=0.2)
        
        self.deconv3 = nn.ConvTranspose2d(in_channels=d1, out_channels=num_classes, kernel_size=11, stride=2, padding=5, output_padding=1)
        self.bn6 = nn.BatchNorm2d(num_classes)
        self.softmax2d = nn.Softmax2d()
        self.tanhshrink = nn.Tanhshrink()
        
    def forward(self, x):
        #downsampling
        x = self.dropout1(self.tanhshrink(self.bn1(self.conv1(x))))
        x = self.dropout3(self.high_res2_down(self.dropout2(self.high_res1_down(x))))
        x = self.dropout4(self.tanhshrink(self.bn2(self.conv2(x))))
        x = self.dropout6(self.middle_res2_down(self.dropout5(self.middle_res1_down(x))))
        x = self.dropout7(self.tanhshrink(self.bn3(self.conv3(x))))
        x = self.dropout9(self.low_res2_down(self.dropout8(self.low_res1_down(x))))
        
        #upsampling
        x = self.dropout10(self.tanhshrink(self.bn4(self.deconv1(x))))
        x = self.dropout12(self.middle_res2_up(self.dropout11(self.middle_res1_up(x))))
        x = self.dropout13(self.tanhshrink(self.bn5(self.deconv2(x))))
        x = self.dropout15(self.high_res2_up(self.dropout14(self.high_res1_up(x))))
        x = self.bn6(self.deconv3(x))
        return self.softmax2d(x)