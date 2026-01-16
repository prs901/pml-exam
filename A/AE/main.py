import sys

sys.path.append('AE/')

from mnist_AE_solution import main as res_main

import os

def main(run_number, dataloader):
    os.makedirs("AE/Generations", exist_ok=True)
    os.makedirs("AE/AE_Models", exist_ok=True)
    os.makedirs("AE/AE_Training_Monitoring", exist_ok=True)
    os.makedirs("AE/DDPM_Training_Monitoring", exist_ok=True)

    '''
        For the final run, num_AE_epochs should be 5 and num_ddpm_epochs should be 100
        use_reporter should be False
    '''
    num_AE_epochs = 5
    num_ddpm_epochs = 100
    use_reporter = True

    return res_main(run_number, dataloader, num_AE_epochs, num_ddpm_epochs, use_reporter)

if __name__ == "__main__":
    main()