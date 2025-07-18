import torch.nn as nn


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)



def show_tensor_images(tensor, normalize=True, denormalize=False, title=None):
    """
    Display a PyTorch tensor as an image or batch of images.
    
    Args:
        tensor (torch.Tensor): Image tensor of shape (C,H,W) or (B,C,H,W)
        normalize (bool): Whether to normalize values to [0,1] range
        denormalize (bool): Whether to denormalize using ImageNet stats
        title (str): Optional title for the plot
        
    Returns:
        None - displays plot
    """
    import matplotlib.pyplot as plt
    import torch
    import torchvision.utils as vutils
    
    # Clone tensor to avoid modifying original
    images = tensor.clone().detach().cpu()
    
    # Add batch dim if needed
    if len(images.shape) == 3:
        images = images.unsqueeze(0)
        
    # Denormalize if requested (assumes ImageNet normalization)
    if denormalize:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        images = images * std + mean
    
    # Normalize to [0,1] if requested
    if normalize:
        images = (images - images.min()) / (images.max() - images.min())
    
    # Convert to grid
    grid = vutils.make_grid(images, nrow=4, padding=2, normalize=False)
    
    # Display
    plt.figure(figsize=(10,10))
    plt.axis('off')
    if title:
        plt.title(title)
    plt.imshow(grid.permute(1,2,0))
    plt.show()

