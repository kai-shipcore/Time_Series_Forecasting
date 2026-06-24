# Orchestrates the full forecasting run end to end
from src.ingest import ingest
from src.clean import clean
from src.profile import profile
from src.segment import segment
from src.backtest import backtest
from src.models import get_models
from src.selector import select
from src.forecast import forecast


def run():
    raise NotImplementedError


if __name__ == "__main__":
    run()
