import brainchild.loader

def test_mindboggle_dataset(capsys):
    data_dir = 'data/Mindboggle'
    ds = brainchild.loader.MindboggleData(data_dir, goal='register')
    x, y = ds[1]
    with capsys.disabled():
        print('dtype')
        print(' inputs:', x.type())
        print(' labels:', y.type())
        print('shape')
        print(' inputs:', x.shape)
        print(' labels:', y.shape)

def test_ppmi_dataset(capsys):
    data_dir = './data/PPMI_reg'
    ds = brainchild.loader.PPMIData(data_dir)
    with capsys.disabled():
        print(ds[0])
def test_load_dataset(capsys):
    data_dir = './data/Mindboggle'
    loader = iter(brainchild.loader.load_dataset(data_dir, dataset='Mindboggle', goal='register'))
    batch = next(loader)
    x, y = batch
    with capsys.disabled():
        print('dtype')
        print(' inputs:', x.type())
        print(' labels:', y.type())
        print('shape')
        print(' inputs:', x.shape)
        print(' labels:', y.shape)
