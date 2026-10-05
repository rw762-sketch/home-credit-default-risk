"""Download Home Credit files after configuring Kaggle API authentication."""
from pathlib import Path

import kagglehub
from kagglehub.exceptions import UnauthenticatedError


def main():
    destination = Path(__file__).resolve().parents[1] / 'data' / 'raw'
    try:
        path = kagglehub.competition_download(
            'home-credit-default-risk', output_dir=str(destination))
    except UnauthenticatedError:
        raise SystemExit(
            'Kaggle API authentication is missing. Configure a Kaggle API token '
            'locally using kagglehub.login(); do not paste it into chat. '
            'Accept the Home Credit competition rules on Kaggle before downloading.')
    print('Path to competition files:', path)


if __name__ == '__main__':
    main()
