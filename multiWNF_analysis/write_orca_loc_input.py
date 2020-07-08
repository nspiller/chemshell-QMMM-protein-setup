#!/usr/bin/env python3

import pandas as pd

import argparse
import re

def get_df_orb(output):
    ' From an output file return orbital block as pandas dataframe '

    df = pd.DataFrame(columns=['n', 'occ', 'e_ha', 'e_ev', 'spin'])
    parse, alpha = False, False


    with open(output) as f:
        for line in f:
            l = line.split()
            if 'SPIN UP ORBITALS' in line:
                parse, alpha = True, True
                continue
            elif 'SPIN DOWN ORBITALS' in line:
                parse, alpha = True, False
                continue
            elif len(l) == 0:
                parse = False
                continue
            elif l[0] == 'NO':
                continue

            if parse:
                n = int(l[0])
                occ = float(l[1])
                ha = float(l[2])
                ev = float(l[3])
                s = 0 if alpha else 1
                
                row = len(df)
                df.loc[row,:] = [ n, occ, ha, ev, s ]


    return df

def write_orca_loc(orb_i, orb_f, spin, fname, gbwin, gbwout):
    ' write orca_loc input file with initial, final orbital and spin'

    template = '''\
{gbwin:10}  # input gbw
{gbwout:10}  # output gbw
{orb_i:10}  # orbital window: first orbital to be localized 
{orb_f:10}  # orbital window: last orbital to be localized 
2           # localization method 1=PIPEK-MEZEY,2=FOSTER-BOYS,3=IAO-IBO,4=IAO-BOYS, 5=NEW-BOYS 6=FB AUGHESS
{spin:10}  # operator:0 for alpha, 1 for beta
1200        # maximum number of iterations
1e-6        # convergence tolerance of the localization functional value
0.0         # relative convergence tolerance of the localization functional value
100         # printing thresh to call an orbital strongly localized: all should be printed as deloc
100         # printing thresh to call an orbital bond-like: all should be printed as deloc
2           # print level
1           # use Cholesky Decomposition (0=false, 1=true, default is true,optional)
0           # Randomize seed for localization(optional)'''

    context = {
        'gbwin' : str(gbwin),
        'gbwout': str(gbwout),
        'orb_i' : str(orb_i),
        'orb_f' : str(orb_f),
        'spin'  : str(spin), }

    with open(fname, 'w') as f:
        f.write(template.format(**context))

def run():

    parser = argparse.ArgumentParser(description='determine orbital ranges from orca output and write orca_loc input files')
    parser.add_argument('output', help='give output file of orca calulationn')
    args = parser.parse_args()
    orcaout = args.output
    gbw = re.sub(r'(mpi\d*\.)?out$', 'gbw', orcaout)
    
    df = get_df_orb(orcaout)

    # only occupied with energy > -1 Ha
    df_a = df.loc[ ( df.loc[:,'occ'] != 0 ) & ( df.loc[:,'spin'] == 0 ) & (df.loc[:,'e_ha'] > -1)]
    df_b = df.loc[ ( df.loc[:,'occ'] != 0 ) & ( df.loc[:,'spin'] == 1 ) & (df.loc[:,'e_ha'] > -1)] 
    
    a_min = df.loc[ df_a.loc[:,'e_ha'].astype(float).idxmin(), 'n' ]
    a_max = df.loc[ df_a.loc[:,'e_ha'].astype(float).idxmax(), 'n' ]
    b_min = df.loc[ df_b.loc[:,'e_ha'].astype(float).idxmin(), 'n' ]
    b_max = df.loc[ df_b.loc[:,'e_ha'].astype(float).idxmax(), 'n' ]
    
    write_orca_loc(a_min, a_max, 0, 'orca_loc_a.input', gbw, 'loc_a.gbw')
    write_orca_loc(b_min, b_max, 1, 'orca_loc_b.input', 'loc_a.gbw', 'loc.gbw')

if __name__ == '__main__':
    run()
