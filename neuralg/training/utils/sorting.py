import torch


def real_sort(tensor, dim=-1):
    """Sorts the elements of a potentially complex valued tensor along a given dimension in ascending order by real part.
        If dim is not given, the last dimension of the input is chosen.
        If descending is True then the elements are sorted in descending order.
   Args:
        tensor (tensor): input tensor to sort.
        dim (int, optional): Dimension to sort along. Defaults to -1.

    Returns:
        tensor: Containing the sorted values of the elements in the original input tensor
    """
    indices = torch.sort(torch.real(tensor))[1]
    t = torch.gather(tensor, dim=dim, index=indices)
    return t


def abs_sort(tensor, dim=-1, descending=False):
    """Sorts the elements of a potentially complex valued tensor along a given dimension in ascending order by absolute value.
        If dim is not given, the last dimension of the input is chosen.
        If  descending is True then the elements are sorted in descending order.
    Args:
        tensor (tensor): input tensor to sort.
        dim (int, optional): Dimension to sort along. Defaults to -1.

    Returns:
        tensor: Containing the sorted values of the elements in the original input tensor
    """
    indices = torch.sort(tensor.abs(), descending=descending)[1]
    t = torch.gather(tensor, dim=dim, index=indices)
    return t
