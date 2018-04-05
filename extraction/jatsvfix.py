import sys
from tqdm import tqdm
if __name__ == '__main__':
    with open(sys.argv[2], 'w') as outfile:
        with open(sys.argv[1], 'r') as f:
            for line in tqdm(f):
                tmp = line.split('\t')
                tmp[0] = tmp[0].replace(' ', '')
                outfile.write('\t'.join(tmp))
                if '。' in tmp[0]:
                    outfile.write('\n')
