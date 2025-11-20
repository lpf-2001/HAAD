import torch
def running_WalkieTalkie(model,x_test,y_test,x_train,y_train,device):
    from Defence_method.WalkieTakie.WT_Util import WalkieTalkie
    
    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    wt = WalkieTalkie(model, device=device)
    perturbed_x=wt.generate_walkie_talkie_samples(x_test,y_test,x_train,y_train)
    return perturbed_x
