import torch
from loguru import logger


def set_precision(data_type="float32"):
    """Allows the user to set the default precision for floating point numbers for torch backend

    Args:
        data_type (str, optional): Data type to use, either "float32" or "float64". Defaults to "float32".
    """
    cuda_enabled = torch.cuda.is_initialized()
    tensor_dtype, tensor_dtype_name = {
        ("float32", True): (torch.cuda.FloatTensor, "cuda.Float32"),
        ("float64", True): (torch.cuda.DoubleTensor, "cuda.Float64"),
        ("float32", False): (torch.FloatTensor, "Float32"),
        ("float64", False): (torch.DoubleTensor, "Float64"),
    }[(data_type, cuda_enabled)]
    cuda_enabled_info = (
        "CUDA is initialized" if cuda_enabled else "CUDA not initialized"
    )
    logger.info(
        f"Setting Torch's default tensor type to {tensor_dtype_name} ({cuda_enabled_info})."
    )
    torch.set_default_tensor_type(tensor_dtype)
