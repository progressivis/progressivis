"""
Example:
cat /tmp/mnist8m|python ../scripts/svmlight2csv.py --n_features=784 --n_samples=500000 --with_header=pixel >mnist500k.csv
"""
import click
from sys import stdin

@click.command()
@click.option('--n_samples', default=10, help='number of samples')
@click.option('--n_features', default=3, help='number of features')
@click.option('--with_labels', default="class", help='if not empty=>label col')
@click.option('--with_header', default="_", help='if not empty=>header pattern')
def main(n_samples, n_features, with_labels, with_header):
    if with_header:
        patt = f"{with_header}{{}}"
        header = ','.join([patt.format(i) for i in range(n_features)])
        if with_labels:
            header += f',{with_labels}'
        print(header)
    keys_ = [str(i) for i in range(n_features)]
    for i, line in zip(range(n_samples), stdin):
        lab, *kw = line.split(' ')
        pairs = [elt.strip().split(':') for elt in kw]
        dict_ = dict(pairs)
        s = ','.join([dict_.get(k, '0') for k in keys_])
        print(s, end='')
        if with_labels:
            print(f',{lab}')
        else:
            print('')
            #print(lab)
if __name__ == '__main__':
    main()

    
