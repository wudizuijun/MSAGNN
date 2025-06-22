# used in shell 
import argparse


''' FCC debug '''
def get_args():
    parser = argparse.ArgumentParser(description='results! ')

    parser.add_argument('--time', type=str, required=False, default='day', help='runing time')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    a = get_args()
    print(a.time)

