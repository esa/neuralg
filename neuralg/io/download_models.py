from loguru import logger
import requests
import zipfile
import io
import os


def download_models():
    """Downloads and unzips the folder of model state dicts attached to the latest GitHub release.
    Will save the folder as saved_models in the models subdirectory.
    """
    logger.info("Downloading models from the latest release")
    # This is currently pre-release version
    zip_url = "https://github.com/esa/neuralg/releases/download/v.0.0.1-alpha/saved_models.zip"
    # This should be enabled for real release. Remember to tag release with latest.
    # "https://github.com/esa/neuralg/releases/latest/download/saved_models.zip"
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
