import sys
import os
# To get the absolute path of the directory where the lgdcnn module is located
lgdcnn_dir = r"D:\deep\LGDCNN"
# Add the directory where the lgdcnn module is located to the module search path
sys.path.append(lgdcnn_dir)

from paper.utils import  save_results

import argparse
# Create an argument parser
parser = argparse.ArgumentParser()
# Add an argument to decide which import to use
parser.add_argument("--import_choose", choices=["LGDCNN", "LGDCNN_v1", "LGDCNN_onehot", "LDCNN"], required=True)

# Add an argument to decide which pth(saved models) to use
parser.add_argument("--choose_model", choices=["1", "2", "3", "4"], required=True)


# Parse the command line arguments
args = parser.parse_args()

# Import the desired module based on the provided argument
if args.import_choose == "LGDCNN":
    from lgdcnn.fusion_lstm_dcnn import LGDCNN
elif args.import_choose == "LGDCNN_v1":
    from lgdcnn.fusion_lstm_dcnn_v1 import LGDCNN
elif args.import_choose == "LGDCNN_onehot":
    from lgdcnn.fusion_lstm_dcnn_onehot import LGDCNN
elif args.import_choose == "LDCNN":
    from lgdcnn.fusion_ldcnn import LGDCNN
else: 
    pass

# choose model
if args.choose_model == "1":
    model_name = "L-G-DCNN"
elif args.choose_model == "2":
    model_name = "L-G-DCNN-v1"
elif args.choose_model == "3":
    model_name = "L-G-DCNN-onehot"
elif args.choose_model == "4":
    model_name = "LDCNN"
else: 
    pass 


if __name__ == '__main__':
    # To construct the path of the benchmark_data folder
    benchmark_data_dir = os.path.join(lgdcnn_dir,"data","benchmark_data")
    mat_props = os.listdir(benchmark_data_dir)
    classification_list = []
    print(f'training: {mat_props}')
    for mat_prop in mat_props:
        classification = False
        if mat_prop in classification_list:
            classification = True
        print(f'property: {mat_prop}')
        print('=====================================================')
        print('calculating test mae')
        model_test, t_mae = save_results(LGDCNN,lgdcnn_dir, model_name, mat_prop, classification, 'test.csv', verbose=False)