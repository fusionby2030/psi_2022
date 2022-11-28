import torch 
def save_model(model, hparams, dataset):
    model_name = hparams['model_name']
    save_dict = {'state_dict': model.state_dict(), 'hparams': hparams, 'dataset': dataset}
    torch.save(save_dict, './' + model_name + '.pth')

def load_model(model_name): 
    save_dict = torch.load(f'./{model_name}')
    return save_dict['state_dict'], save_dict['hparams'], save_dict['dataset']