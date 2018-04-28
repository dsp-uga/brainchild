# Automatic Registration and Segmentation of Brain Images

Our project aims to use deep neural networks to automatically register and segment
magnetic resonance images (MRI). To this end we built two different networks. Our
registration network can be run by entering the repository folder and running the command:
 `python main.py`

The segmentation network can be run by calling  `python vnet_train.py`. The network might run into the error due to unhandled memory explosion. 

 If you wish to attempt running the program on the cluster, copy the `gcp-template.sh` file to the main repository folder and rename to `gcp.sh`. After ensuring that both the `setup.py` and `gcp.sh` and be executed, run `./gcp.sh` in your console.

 
