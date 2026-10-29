# AIVC — AI Version Control

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module=".*pydantic_settings.*")
warnings.filterwarnings("ignore", message=".*lifespan.*")
warnings.filterwarnings("ignore", message=".*file_cache is only supported.*")

import logging
logging.getLogger("googleapiclient").setLevel(logging.ERROR)
logging.getLogger("googleapiclient.discovery_cache").setLevel(logging.ERROR)
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("aivc").setLevel(logging.WARNING)
