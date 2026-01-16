import sys

sys.path.append('DDPM/')

from mnist_ddpm_solution import main as result_main

import os

def main(run_number, dataloader):
    os.makedirs("DDPM/Generations", exist_ok=True)
    os.makedirs("DDPM/Train_Monitoring_DDIM", exist_ok = True)
    os.makedirs("DDPM/Train_Monitoring_DDPM", exist_ok = True)

    '''
        For the final run, num_epochs should be 100 and use_reporter set to False
    '''
    num_epochs = 100
    use_reporter = True
    return result_main(run_number, dataloader, num_epochs, use_reporter)

if __name__ == "__main__":
    main()