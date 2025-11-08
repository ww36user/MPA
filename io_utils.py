import argparse
def parse_args_test():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--image_size'  , default=224, type=int,  help='image size') 
    parser.add_argument('--feature_size' , default=512, type=int, help='feature_size')
    parser.add_argument('--list_of_out_dims', default=[64,128,256,512], help='every block output')
    parser.add_argument('--list_of_stride', default=[1,2,2,2], help='every block conv stride')
    parser.add_argument('--list_of_dilated_rate', default=[1,1,1,1], help='dilated conv') 
    parser.add_argument('--n_way', default=5, type=int,  help='class num to classify for every task')
    parser.add_argument('--n_support', default=5, type=int,  help='number of labeled data in each class, same as n_support') 
    parser.add_argument('--n_query', default=15, type=int,  help='number of test data in each class, same as n_query')
    parser.add_argument('--test_n_eposide', default=100, type=int, help ='total task every epoch')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--current_data_path', default='./target_domain/EuroSAT', help='EuroSAT_data_path')
    parser.add_argument('--current_class', default=10, type=int, help='total number of classes in EuroSAT')
    parser.add_argument('--pretrain_model_path', default='./pretrain/399.tar', help='pretrain_model_path')#正常
    parser.add_argument('--model_path', default='./checkpoint/crop5/100.tar', help='model_path')
    parser.add_argument('--fine_tune_epochs', default=5, type=int, help ='fine tune epoch')
    return parser.parse_args()










