from cell_extractor.calculate_mean_cell_image import MeanImageCalculator
import argparse
if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--animal', type=str, help='Animal ID')
    # parser.add_argument('--disk', type=str, help='storage disk')
    # args = parser.parse_args()
    # braini = args.animal
    # disk = args.disk
    # MeanImageCalculator(braini,disk=disk)
    MeanImageCalculator('DK46',disk='scratch')

