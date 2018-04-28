import brainchild
import brainchild.loader
import brainchild.reg_net as brn
import torch.nn as N
import torch

def main():
    data_dir="./data/Mindboggle"
    # m = brn.reg_net()
    # if torch.cuda.is_available():
    #     net = N.DataParallel(m, device_ids=range(0,len(torch.cuda.device_count())))
    # else:
    #     net = m
    # loss = N.KLDivLoss()
    # optim = torch.optim.Adam(net.parameters())
    # reg = brn.GenerateRegistration(net=net, loss=loss, optim=optim)
    # for i in range(1000):
    #     print(f'Epoch: {i+1}')
    #     if torch.cuda.is_available():
    #         loader = iter(brainchild.loader.load_dataset(data_dir, dataset='Mindboggle', goal='register', batch_size=torch.cuda.device_count()))
    #     else:
    #         loader = iter(brainchild.loader.load_dataset(data_dir, dataset='Mindboggle', goal='register'))
    #     x, y = next(loader)
    #     reg.partial_fit(x, y)
    #
    #     torch.save(net, "./network.pt")
    #     torch.save(reg, "./model.pt")
    model = torch.load('model.pt')
    load_test = iter(brainchild.loader.load_dataset(data_dir, dataset='Mindboggle', goal='register', batch_size=1, num_workers=1))
    for i in range(10):
        x_test, y_test = next(load_test)
        y_test = torch.log(y_test)
        score = model.score(x_test, y_test)
        print(score)


if __name__ == '__main__':
    main()
