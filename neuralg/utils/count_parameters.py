def count_parameters(model):
    """Counts number of trainable parameters in torch model

    Args:
        model (torch.nn): Trained model

    Returns:
        int : Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

