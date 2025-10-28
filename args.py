class args():

    # hyperparameters
    put_type = 'right'
    balance = 1

    # training args
    epochs = 20# "number of training epochs, default is 2"
    trainpath = "./train_image/"
    save_per_epoch = 1
    # trainG1_continue = "off"
    batch_size = 24# "batch size for training/testing, default is 4"
    dataset1 = "./train_visible.txt"
    dataset2 = "./train_lwir.txt"
    # HEIGHT = 1024
    # WIDTH = 672
    lr = 1e-4 # "Initial learning rate, default is 0.0001"
    lr_step = 10 # Learning rate is halved in 10 epochs
    #SSIM_resume = "./models/best.SSIM.ckpt_2.pt" # if you have, please put the path of the model like "./models/densefuse_gray.model"
    #tv_resume = "./models/best.TV.ckpt_2.pt"
    resume = None
    #resume = "./models/Wrap.ckpt_5.pt"
    #resume = "/media/lab/liulu/liulu/LRINet_Copy/models/fakegt28_0.5_1.5_150.pt"

    save_model_dir = "./models/" #"path to folder where trained model with checkpoints will be saved."
    workers = 16
    #train_continue = "on"
    pre_train = "on"
    train_continue = "off"
    st_eopch = 0
    mode = "train"
    dir_chck = "/media/media/6dbfb7d7-4857-4183-acea-f582f662b061/liulu/trainVVC/checkpoints/"
    dir_log_train = "./log/"

    # For GPU training
    world_size = -1
    rank = -1
    gpu = 0
    multiprocessing_distributed = False
    distributed = None

    # For testing
    test_save_dir = './media/lab/liulu/liulu/tt/test/result'
    test_visible = './media/lab/liulu/liulu/tt/test/rgb'
    test_lwir = './media/lab/liulu/liulu/tt/test/nir'

