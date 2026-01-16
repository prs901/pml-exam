from typing import Callable
from pathlib import Path
from csv import DictWriter
from logging import Logger

from DDPM.main import main as ddpm_main
from AE.main import main as ae_main
from FM.main import main as fm_main
from DataLoader import DataLoader


DATETIME_FORMAT = r'%Y-%m-%d %H:%M:%S'


class Result:
    def __init__(self, training_time: int, evaluation_time: int, fid_train: float, fid_test: float, inception: float):
        self.training_time = training_time
        self.evaluation_time = evaluation_time
        self.fid_train = fid_train
        self.fid_test = fid_test
        self.inception = inception


class A:
    def __init__(self):
        self.n_runs = 5
        self.logger = Logger('A')
        self.results = Path('results').resolve()
        self.models = {
            'ddpm': ddpm_main,
            'fm': fm_main,
            'ae': ae_main
        }

        self.dataloader = DataLoader()


    def log_result(self, model: str, run: int, result: Result):
        self.logger.info(f'Saving {model} run {run}')

        row = {
            'model': model,
            'run': run,
            'training_time': result.training_time,
            'evaluation_time': result.evaluation_time,
            'fid_train': result.fid_train,
            'fid_test' : result.fid_test,
            'inception' : result.inception
        }

        try:
            with open(self.results, 'a') as fd:
                fields = list(row.keys())
                writer = DictWriter(fd, fields)
                writer.writerow(row)
        except Exception as exc:
            self.logger.error(f'Failed to save {model} run {run}: {row}', exc_info=exc)
            return

        self.logger.info(f'Saved {model} run {run}')


    def execute_run(self, main_fn: Callable, run_number: int):
        result = None

        try:
            training_time, evaluation_time, fid_train, fid_test, inception = main_fn(run_number, self.dataloader)
            result = Result(training_time, evaluation_time, fid_train, fid_test, inception)
        except Exception as exc:
            self.logger.exception('run failed!', exc_info=exc)
        finally:
            return result


    def execute_model(self, model: str):
        print("\nExecuting Model: {}\n".format(model))
        self.logger.info(f'Starting {model}')

        main_fn = self.models[model]

        for run_number in range(self.n_runs):
            print("Run Number: {}".format(run_number))
            self.logger.info(f'.. run {run_number}')
            result = self.execute_run(main_fn, run_number)
            if result is None:
                self.logger.warning(f'Aborting {model} at run {run_number}')
                break
            self.log_result(model, run_number, result)

        self.logger.info(f'Finished {model}')


    def execute_all(self):
        self.logger.info('All models')
        for model in self.models:
            self.execute_model(model)
        self.logger.info('Finished all models')


if __name__ == '__main__':
    a = A()
    a.execute_all()
