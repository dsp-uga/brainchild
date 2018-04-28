import brainchild.loader
import brainchild.reg_net as brn
import torch.nn as N
import torch

def main():
    data_dir="/home/orcsy/brainchild/data/Mindboggle"
    m = brn.reg_net()

    if torch.cuda.is_available():
        net = N.DataParallel(m, device_ids=range(0,len(torch.cuda.device_count())))
        loader = iter(brainchild.loader.load_dataset(data_dir, dataset='Mindboggle', goal='register', batch_size=torch.cuda.device_count()))
    else:
        net = m
        loader = iter(brainchild.loader.load_dataset(data_dir, dataset='Mindboggle', goal='register'))

    loss = N.KLDivLoss()
    optim = torch.optim.Adam(net.parameters())
    reg = brn.GenerateRegistration(net=net, loss=loss, optim=optim)


    # print(f'Epoch: {i+1}')
    x, y = next(loader)
    reg.partial_fit(x, y)

    torch.save(net, "./network1.pt")
    torch.save(reg, "./model1.pt")

    load_test = iter(brainchild.loader.load_dataset(data_dir, dataset='Mindboggle', goal='register'))
    x_test, y_test = next(load_test)
    h_test = reg.predict(x_test)
    torch.save(net, "./network.pt")
    torch.save(reg, "./model.pt")
if __name__ == '__main__':
    main()
