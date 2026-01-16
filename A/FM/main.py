import sys

sys.path.append('FM/')

from solution import main as run_main

import os

def main(run_number, dataloader):
    os.makedirs("FM/Generations", exist_ok = True)
    os.makedirs("FM/fm_samples", exist_ok = True)       

    '''
        For the final run, num_epochs should be 100 and use_reporter set to False
    '''

    num_epochs = 100
    use_reporter = True

    return run_main(run_number, dataloader, num_epochs, use_reporter)

if __name__ == "__main__":
    main()