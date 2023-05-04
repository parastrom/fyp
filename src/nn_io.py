import torch


def save(model, optimzer, save_path):
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    torch.save(model.state_dict(), save_path + '.params')
    torch.save(optimzer.state_dict(), save_path + '.opt')
