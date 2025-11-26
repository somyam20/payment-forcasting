import logging
import logging.config
import os
import yaml

CONFIG_PATH = os.path.join("config", "logging_config.yaml")

def setup_logging():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
            logging.config.dictConfig(config)
    else:
        # fallback config
        logging.basicConfig(
            level=logging.INFO,
            format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s"
        )

# Initialize logging at import
setup_logging()
logger = logging.getLogger("generative_ai_project")
