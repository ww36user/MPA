import argparse
def parse_args_test():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--image_size'  , default=224, type=int,  help='image size')
    parser.add_argument('--n_way', default=5, type=int,  help='class num to classify for every task')
    parser.add_argument('--n_support', default=5, type=int,  help='number of labeled data in each class, same as n_support') 
    parser.add_argument('--n_query', default=15, type=int,  help='number of test data in each class, same as n_query')
    parser.add_argument('--test_n_eposide', default=100, type=int, help ='total task every epoch')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--current_data_path', default='./target_domain/EuroSAT', help='EuroSAT_data_path')
    parser.add_argument('--current_class', default=10, type=int, help='total number of classes in EuroSAT')
    return parser.parse_args()










