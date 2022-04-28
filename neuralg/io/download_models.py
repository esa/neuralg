from loguru import logger
from .. import __version__
import requests
import zipfile
import io
import os


def download_models():
    """Downloads and unzips the folder of model state dicts attached to the latest GitHub release.
    Will save the folder as saved_models in the models subdirectory.
    """
    logger.info(
        f"Downloading models from the latest release for neuralg v{__version__}"
    )

    # Link to latest release
    zip_url = "https://github.com/esa/neuralg/releases/latest/download/saved_models.zip"

    r = requests.get(zip_url)
    logger.info("Download complete. Unzipping content")
    z = zipfile.ZipFile(io.BytesIO(r.content))
    saved_models_path = os.path.realpath(
        os.path.join(
            os.path.dirname(__file__),
            "../models/",
        )
    )
    z.extractall(saved_models_path)
