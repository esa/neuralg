import neuralg


def clear_loaded_models():
    """Free allocated memory from loaded models
    """
    neuralg.neuralg_ModelHandler.clear_loaded_models()

