import confuse
import argparse


def load_config_from_yaml(file_path):
    config = confuse.Configuration('GAIL', __name__)
    config.set_file(file_path)

    return config['GAIL']


