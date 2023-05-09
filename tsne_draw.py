import torch
from TSNE import RunTsne
from networks.UNet import UNet
from setup_loader import setup_loader


if __name__ == '__main__':
    # 自己指定要进行t-SNE的类别（可以根据t-SNE的效果选择最好的几个类别即可）
    all_class = False   # t-SNE展示全部类别，还是部分类别
    if all_class:
        selected_cls = ['cup', 'disc', 'background']
    else:
        selected_cls = ['cup', 'disc']


    # 为每个数据集指定一个ID
    domId2name = {
        0:'DomainA',
        1:'DomainB',
        2:'DomainC',
        3:'DomainD'}


    # Two ways to use, we take the cross-domain Fundu dataset for an example(两种使用方式：这里以cross-domain Fundus dataset 为例子)
    # 1.默认使用cityscapes里面的标签类别
    # 2.直接定义trainId2name 和 trainId2color
    # 这里采用第2种方式

    # import cityscapes_labels
    # trainId2name = cityscapes_labels.trainId2name

    trainId2name = {0: 'cup',
                    1: 'disc',
                    2: 'background'}
    # trainId2name = {255: 'trailer',
    #                 0: 'road',
    #                 1: 'sidewalk',
    #                 2: 'building',
    #                 3: 'wall',
    #                 4: 'fence',
    #                 5: 'pole',
    #                 6: 'traffic light',
    #                 7: 'traffic sign',
    #                 8: 'vegetation',
    #                 9: 'terrain',
    #                 10: 'sky',
    #                 11: 'person',
    #                 12: 'rider',
    #                 13: 'car',
    #                 14: 'truck',
    #                 15: 'bus',
    #                 16: 'train',
    #                 17: 'motorcycle',
    #                 18: 'bicycle',
    #                 -1: 'license plate'}


    # trainId2color = cityscapes_labels.trainId2color
    trainId2color = {
                        0: (128, 64, 128),
                        1: (255, 0, 0),
                        2: (152, 251, 152)}

    output_dir = './'
    tsnecuda = True
    extention = '.png'
    duplication = 10
    plot_memory = False
    clscolor = True
    domains2draw = ['DomainA', 'DomainB', 'DomainC', 'DomainD'] #需要自定义修改
    # 指定需要进行t-SNE的域，即数据集

    tsne_runner = RunTsne(selected_cls=selected_cls,
                          domId2name=domId2name,
                          trainId2name=trainId2name,
                          trainId2color=trainId2color,
                          output_dir=output_dir,
                          tsnecuda=tsnecuda,
                          extention=extention,
                          duplication=duplication)

    ################ inference过程 ################
    # 注意这里是伪代码，根据自己的情况进行修改
    data_loaders = setup_loader()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = UNet(3,3)
    net.to(device=device)
    net= torch.load('C://Users//ZiweiNiu//Desktop//IRLSG//runner//Fundus-D-A.pth', map_location=device)
    net.eval()

    with torch.no_grad():
        for dataset, val_loader in data_loaders.items(): # data_loaders里面包含多个数据集的val_loader
            for val_idx, data in enumerate(val_loader):
                inputs, gt_image = data #[1, 3, 384, 384]
                B, C, H, W = inputs.shape
                gt_image = gt_image.view(-1, H, W) #[3, 384, 384]
                inputs, gt_cuda = inputs.cuda(), gt_image.cuda()
                features, logits = net(inputs)

                tsne_runner.input2basket(features, gt_cuda, dataset)
    ################ inference过程 ################

    # 如果网络中有每个类别的聚类中心，就执行下面的语句
    # m_items = net.module.memory.m_items.clone().detach()
    # tsne_runner.input_memory_item(m_items)

    # t-SNE可视化
    tsne_runner.draw_tsne(domains2draw, plot_memory=plot_memory, clscolor=clscolor)
