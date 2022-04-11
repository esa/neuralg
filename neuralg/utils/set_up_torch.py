from .set_precision import set_precision
from .enable_cuda import enable_cuda


def set_up_torch(data_type=None, torch_enable_cuda=True):
    """ Configure device and potentially precision for neuralg backend.
    Calls neuralg.enable_cuda unless torch_enable_cuda is False.
    If a data type is passed, set the default floating point precision with neuralg.set_precision.
    Args:
        data_type ("float32", "float64" or None, optional): Data type which is passed to set_precision. If None, do not call set_precision except if CUDA is enabled for torch. Defaults to None.
        torch_enable_cuda (Bool, optional): If True and backend is "torch", call enable_cuda. Defaults to True.
    """
    if torch_enable_cuda:
        if data_type is None:
            enable_cuda()
        else:
            # Do not call set_precision twice.
            enable_cuda(data_type=None)
    if data_type is not None:
        set_precision(data_type)
