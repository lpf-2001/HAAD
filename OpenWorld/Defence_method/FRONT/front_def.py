import torch

def running_front(model,test_x,device):
    from Defence_method.FRONT.Front_util import DirectionalFRONT
    
    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    front = DirectionalFRONT(
        model, 
        Nc=25, 
        Ns=4, 
        Wmin=0.1, 
        Wmax=0.2,
        device=device
    )  # Wmin=[0.01, 0.05 0.1 0.15 0.2], Wmax=[0.1 0.2 0.3 0.4]
    # AWF: Wmin=0.1 Wmax=0.1
    # DF: Wmin=0.1 Wmax=0.2

    perturbed_x=front.perturb_batch_torch(test_x,False)
    return perturbed_x.cpu().numpy()
