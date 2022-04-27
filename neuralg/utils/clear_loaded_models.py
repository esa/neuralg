from .. import neuralg_ModelHandler


def clear_loaded_models():
    """Free allocated memory from loaded models"""
    neuralg_ModelHandler.clear_loaded_models()
