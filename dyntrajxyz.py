#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy.constants import physical_constants as const
import argparse

# string and float format strings (only for slow option)
fmt_s = lambda x: '{:2}'.format(x)
fmt_f = lambda x: '{:10.6f}'.format(x)

def bohr2ang(x_bohr):
    x_ang = x_bohr * const['Bohr radius'][0] * 1e10
    return x_ang

def get_numbers(fin):
    '''get number of atoms and number of lines from file f
    returns two integers: atom number and line number
    '''
    with open(fin) as f:
        first = next(f)
        second = next(f)
        n = int(second.split('=')[-1])
        l = 2
        for line in f:
            l += 1
    return n, l

def get_df(fin, n, l):
    'return dataframe with data from file f (skipping all non-data lines)'
    # non-data lines
    i = [0] + list(range(1, l, n+1)) 
    
    columns = ['a', 'x', 'y', 'z']
    df = pd.read_csv(fin, skiprows=i, delim_whitespace=True, names=columns)
    
    # convert to angstrom
    df.loc[:,columns[1:]] = df.loc[:,columns[1:]].apply(bohr2ang)
           
    return df
    
    

def write_output(out, df, n, l):
    'write output file f'

    
    with open(out, 'w') as f:
        
        nf = int((l-1) / (n+1)) # number of frames 

        for i in range(nf):
            # two lines: 1) number of atoms 2) comment
            f.write('{}\n'.format(n))
            f.write('stuff\n')  

            # one frame at a time
            df_ = df.iloc[i*n:(i+1)*n, :]
            
            # better looking
#             string = df_.to_string(header=False, index=False, formatters=[fmt_s, fmt_f, fmt_f, fmt_f])
#             f.write(string)

            # faster
            df_.to_csv(f, mode='a', header=False, sep=' ', index=False, float_format='%.6f' )

            print('INFO: Done writing frame {}/{}'.format(i, nf), end='\r')
        print()

def run():
    parser = argparse.ArgumentParser(description='convert chemshell trajectory into xyz trajectory')
    parser.add_argument('chm_trj', metavar='TRJ', help='Chemshell trajectory file in Bohr')
    parser.add_argument('-o', '--output', metavar='XYZ', help='Name of xyz trajectory file in Ang. Default: TRJ.xyz')
    args = parser.parse_args()

    fin = args.chm_trj
    fout = args.output if args.output else fin + '.xyz'
    print('INFO: Reading {}'.format(fin))
    
    n, l = get_numbers(fin)
    print('INFO: Found {} atoms and {} lines'.format(n, l))

    print('INFO: Reading data from {}'.format(fin))
    df = get_df(fin, n, l)

    print('INFO: Writing to {}'.format(fout))
    write_output(fout, df, n, l)
    print('INFO: ... done!')

if __name__ == '__main__':
    run()
