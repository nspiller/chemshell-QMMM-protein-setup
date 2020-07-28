#!/usr/bin/env python3

################
## User Input ##
################

# this dict converts orca atom indices to sensible names (required, if orca1.out not present)
# fill here, if automatic determination does not work
orca2name = { 
#    16: 'Fe1',
#    17: 'Fe2',
#    18: 'Fe3',
#    19: 'Fe4',
#    20: 'Fe5',
#    21: 'Fe6',
#    22: 'Fe7',
#    15: 'Mo',
}

import numpy as np
import pandas as pd
import re

from pathlib import Path
import argparse
import sys

import matplotlib.pylab as plt
import seaborn as sns

from matplotlib import rcParams
rcParams['figure.dpi'] = 200

###############
## HIRSHFELD ##
###############
def get_hirsh_spin(output):
    '''get values of a fuzzy volume integration

    return values of block given below as df (here: spin)
    ATTENTION: MultiWFN counts from 0, returned df counts from 0 

   Atomic space        Value                % of sum            % of sum abs
     1(C )            0.00001069             0.000356             0.000054
     2(H )           -0.00000212            -0.000071            -0.000011
     3(H )            0.00000017             0.000006             0.000001
     4(C )            0.00009353             0.003118             0.000476
     5(H )            0.00002562             0.000854             0.000130
     6(H )            0.00000051             0.000017             0.000003
     7(C )            0.00059285             0.019762             0.003018
    '''

    df = pd.DataFrame() # empty dataframe
    parse = False # parsing switch

    with open(output) as f:
        for line in f:
            l = line.split()

            if 'Atomic space Value % of sum % of sum abs' in ' '.join(l): # start parsing
                parse = True
                continue
            elif 'Summing up above values:' in line: # stop parsing 
                parse = False
                continue

            if parse:
                i = int(l[0].split('(')[0]) - 1 # counting from 0
                c = float(l[-3])
                df.loc['Hirshfeld spin', i] = c

    return df

def get_hirsh_charge(output):
    '''return CM5 and Hirshfeld charges
    
    parses the block given below and return dataframe with Hirshfeld and CM5 charges
    ATTENTION: MultiWFN counts from 0, returned df counts from 0 

                         ======= Summary of CM5 charges =======
     Atom:    1C   CM5 charge:   -0.227214  Hirshfeld charge:   -0.077632
     Atom:    2H   CM5 charge:    0.079908  Hirshfeld charge:    0.027221
     Atom:    3H   CM5 charge:    0.092847  Hirshfeld charge:    0.041910
     Atom:    4C   CM5 charge:   -0.155276  Hirshfeld charge:   -0.048841'''

    df = pd.DataFrame() # initiate dataframe
    parse = False # parsing switch

    with open(output) as f:
        for line in f:

            if '======= Summary of CM5 charges =======' in line: # start parsing
                parse = True
                continue
            elif 'Summing up all CM5 charges:' in line: # stop parsing
                parse = False
                continue

            if parse:
                l = line.split()

                i = int(re.findall('^\d+', l[1])[0]) - 1 # counting from 0
                hirsh = float(l[-1])
                cm5 = float(l[-4])

                df.loc['Hirshfeld charge', i] = hirsh
                df.loc['CM5 charge', i] = cm5

    return df

##############################
## LIDI.txt and orbcomp.txt ##
##############################

def get_lidi(output):
    '''
    returns pandas dataframes from LIDI.txt (basin/atom numbering starts with 0)
    '''

    # six empty dataframes in total
    df_deloc_a, df_deloc_b, df_deloc_tot = [pd.DataFrame(dtype=float) for _ in range(3)]
    df_loc_a, df_loc_b, df_loc_tot       = [pd.DataFrame(dtype=float) for _ in range(3)]

    parse = None # flag if parsing

    # localization block needs complex parser to match both basins and atoms
    loc_regex = re.compile(
        '((?P<i1>\d+(\(\w+\s*\))?:)\s*(?P<c1>-?\d+\.\d+))'
        '\s*'
        '((?P<i2>\d+(\(\w+\s*\))?:)\s*(?P<c2>-?\d+\.\d+))?'
        '\s*'
        '((?P<i3>\d+(\(\w+\s*\))?:)\s*(?P<c3>-?\d+\.\d+))?'
        '\s*'
        '((?P<i4>\d+(\(\w+\s*\))?:)\s*(?P<c4>-?\d+\.\d+))?'
        '\s*'
        '((?P<i5>\d+(\(\w+\s*\))?:)\s*(?P<c5>-?\d+\.\d+))?'
        )

    with open(output) as f:
        for line in f:

            # change df and parse to which part of the file is parsed
            if 'Delocalization index matrix for alpha' in line:
                df = df_deloc_a
                parse = 'deloc'
            elif 'Localization index for alpha' in line: # LIDI.txt has spin or electron here
                df = df_loc_a
                parse = 'loc'
            elif 'Delocalization index matrix for beta' in line:
                df = df_deloc_b
                parse = 'deloc'
            elif 'Localization index for beta' in line: # LIDI.txt has spin or electron here
                df = df_loc_b
                parse = 'loc'
            elif 'Total delocalization index matrix' in line:
                df = df_deloc_tot
                parse = 'deloc'
            elif 'Total localization index:' in line or 'Localization index:' in line:
                df = df_loc_tot
                parse = 'loc'
            elif len(line.split()) == 0:
                parse = None
                
            # all other lines contain data
            else:
                l = line.split()

                if parse == 'deloc':  # parse delocalization indices
                    try: # deloc can be very irregular
                        x = l[1]
                    except IndexError:
                        x = ''
                    if '.' not in x:
                        i_col_range = [ int(i) - 1 for i in l ] # counting from 0
                    elif '.' in x:
                        i_row = int(l[0]) - 1 # counting from 0
                        c_range = [ float(c) for c in l[1:] ]
                        for i_col, c in zip(i_col_range, c_range):
                            df.loc[i_row, i_col] = c

                elif parse == 'loc': # parse localization indices
                    m = loc_regex.search(line).groupdict()
                    for n in range(1, 6):
                        try:
                            i_ = re.findall('\d+', m['i{}'.format(n)])[0] 
                            i = int(i_) - 1
                            c = float(m['c{}'.format(n)])
                            df.loc[i, i ] = c
                        except TypeError:
                            pass

        # return all six dataframes
        return df_deloc_a, df_deloc_b, df_deloc_tot, df_loc_a, df_loc_b, df_loc_tot

def read_orbcomp(output, orb_i=0, orb_f=np.inf, orb_0=0):
    '''creates pandas dataframe from orbcomp.txt, counting from 0

    optional
    orb_i       first orbital index (counting from 0)
    orb_f       last orbital index
    orb_0       index for start counting (default: 0), required for beta'''

    df = pd.DataFrame() # empty DataFrame

    with open(output) as f:
        for line in f: 
            
            if 'Orbital' in line: # get orbital number from line
                l = line.split()
                n = int(l[1]) - 1 # adjust to ORCA counting
                continue

            # dont parse if out of range
            if orb_i > n:
                continue
            elif orb_f < n:
                break

            l = line.split()
            i = int(l[0]) - 1 # adjust to ORCA counting
            c = float(l[1]) / 100 # convert to 0 < c < 1

            df.loc[n - orb_0, i] = c

    return df

def get_orb(multi, orbcomp, spin, orca2name, minerg, minsum, thresh):
    '''creates dataframe with orbitals that are localized on certain atoms

    requires
    multi       multiwfn output with "-1 # Print basic information of all orbitals" called
    orbcomp     location of 'ORBCOMP.txt' file
    spin        electron spin. 0 for alpha, 1 for beta
    orca2name   dictionary connection orca index with user defined name
    minerg      ignore orbitals with energies lower than this
    minsum      ignore orbitals with summed contributions (according to orca2name) lower than this
    thresh      remove single atomic contributions lower than this (to enhance readability)'''

    # first: get orbital numbers from multiwfn output
    # determine orbital ranges in multiwfn numbering
    df = get_occ_orbitals(multi)

    df_subset = df.loc[ (df.loc[:,'spin'] == spin) & (df.loc[:,'erg'] > minerg) ]

    orb_i, orb_f = df_subset.index.min(), df_subset.index.max()
    orb_0 = df.loc[ df.loc[:,'spin'] == 1 ].index.min() if spin == 1 else 0

    # second: only get specific orbitals from orbcomp.txt
    df = read_orbcomp(orbcomp, orb_i, orb_f, orb_0) # new dataframe df!
    
    # third: only keep orbitals with certain contributions
    df = rename_columns(df, orca2name)
    
    # delete uninteresting orbitals from dataframe
    df.loc[:, 'sum'] = df.loc[:, : ].sum(axis=1) # create sum column
    df = df.loc[ df['sum'] > minsum, : ]

    # remove small contributions
    df = df.apply( lambda x: [y if y > thresh else np.nan for y in x])
    
    # sort values into blocks with > 0.5
    sort_mask = df.where(df > 0.5).sort_values(by=list( df.columns ), ascending=False) # sort mask
    sort_idx = sort_mask.index # order of row for sorting
    df = df.loc[sort_idx, :] # apply

    # append 'a' or 'b' to number to distinguish spin
    df = df.rename(lambda x: '{}{}'.format(x, 'a' if spin == 0 else 'b'))

    return df


########################
## General processing ##
########################

def get_orca2name(output):
    '''find Mo or V in orca output file and define names accordingly

    requires
    orcaout     orca output file

    find index of either Mo or V, then defines Fe1, Fe2 etc. as following directly Mo or V
    returns dict with { orca index: name (e.g. 'M', 'Fe1') }'''

    with open(output) as f:

        parse = False
        for line in f:

            if r'CARTESIAN COORDINATES (A.U.)' in line:
                parse = True
                next(f)
                next(f)
                continue

            if parse:
                l = line.split()
                if l[1] == 'Mo' or l[1] == 'V':
                    idx = int(l[0])
                    break
    
    d = { idx + i: 'Fe{}'.format(i) for i in range(1, 8)}
    d[idx] = 'M'

    return d


def get_occ_orbitals(multiout):
    '''parse multiwfn output to get information on occupied alpha and beta orbitals
    ATTENTION: MultiWFN counts from 1, returned values count from 0 (df.index)
    returns df with columns=['spin', 'occ', 'erg']

    requires
    mulitout    output file of multiwfn with "Pirint basic information of all orbitals" called'''

    df = pd.DataFrame() # initiate dataframe

    spindict = { # translate strings in multiwfn to integers
        'Alpha':        0,
        'Beta':         1,
        'Alpha&Beta':   2,}

    with open(multiout) as f:
        for line in f:
            l = line.split()

            if len(l) == 8 and l[0] == 'Orbital:' and l[-2] == 'Type:':
                    n = int(l[1]) - 1 # adjust to counting from 0
                    e = float(l[3])
                    s = l[-1]
                    o = float(l[-3])

                    if o != 0.0: # only consider occupied orbitals
                        df.loc[n,'spin'] = spindict[s]
                        df.loc[n, 'occ'] = o
                        df.loc[n, 'erg'] = e

    return df


def get_qmatoms(qmatoms):
    '''
    returns a list of indices appearing in the same order of the orca xyz 
    reads the single line file qmatoms
    '''
    with open(qmatoms) as f:
        # get rid of anything but numbers

        line = f.readline().strip()
        line = line.lstrip('set qmatoms {')
        line = line.rstrip('}')

        # make list of chemshell indices
        indices = [ int(i) for i in line.split() ]
        indices.sort()

        return indices

def get_atom_psf(psf, aName, resName=None, resID=None, segName=None, fast=True):    
    '''find atom index in psf file based on atom information

    get atom serial number from psf file (XPLOR format)
    atom number as appearing in psf file (numbering starts with 1)

    first match for atom name [optional: + residue name, +residue id, + segName]

    required: 
    aName           atom name e.g. HE1
    optional: 
    resName         residue name such as MET
    resID           residue ID such as 356
    segName         segment name such as ENZ1
    fast            bool: parsing algorithm: False for regex, True for format string (faster)'''

    match = re.compile( # regex for slow matching
        '^\s*'
        '(?P<i>\d+)'
        '\s*'
        '(?P<sn>[\w\d]+)'
        '\s*'
        '(?P<ri>\d+)'
        '\s*'
        '(?P<rn>[\w\d]+)'
        '\s*'
        '(?P<an>[\w\d]+)'
        '\s*'
        '(?P<at>[\w\d]+)'
        '\s*'
        '(?P<c>-?\d+\.\d+)'
        '\s*'
        '(?P<w>\d+\.\d+)'
        '\s*'
        '(?P<n>\d+)'
        '\s*$'
        )


    with open(psf) as f:

        parse = False
        for line in f:
            if '!NATOM' in line: # start here
                parse = True
                continue
            elif len(line.split()) == 0: # stop on emtpy line
                parse = False
            
            if parse:
                if fast: # fast
                    an = line[38:47].strip() 
                    if an == aName: # only continue if atomname matches
                        i = int(line[0:10].strip())
                        sn = line[11:20].strip()
                        ri = int(line[20:29].strip())
                        rn = line[29:38].strip()
                        if ( rn == resName or resName == None ) and ( ri == resID or resID == None ) and ( sn == segName or segName == None ):
                            return i # terminate once found
                else: # regex
                    if aName in line:
                        d = match.search(line).groupdict()
                        i = int(d['i'])
                        sn = d['sn']
                        ri = int(d['ri'])
                        rn = d['rn']
                        an = d['an']
                        
                        if an == aName and ( rn == resName or resName == None ) and ( ri == resID or resID == None ) and ( sn == segName or segName == None ):
                            return i

        return None # if not found


def rename_columns(df, d):
    '''remove and rename columns in a dataframe

    all coloumns in df that are not keys in d are removed
    all columns in df that are keys in d are renamed according to d
    returns modified dataframe

    df      dataframe
    d       dict with oldcol: newcol'''

    df = df.loc[:,d.keys()]
    df.rename(columns=d, inplace=True)

    return df


##############
## Plotting ##
##############

def save_fig(fig, path):
    if path:
        fig.savefig(path, transparent=False)

def plt_charge_spin(df_charge, df_spin):
    '''plot charge and spin dataframe in one figure

    required
    df_charge   dataframe with charges
    df_spin     dataframe with spin populations
    path        file name to save plot

    returns nothing'''

    x = len(df_charge.columns) 
    fig, axarr = plt.subplots(nrows=2, figsize=(3*x, 8) )

    kw_args = { # settings for both subplots
        'linewidth': 0.5,
        'yticklabels': True,
        'annot': True, 
        'fmt': '.3f', 
        'annot_kws': { 'fontfamily': 'monospace'}, }

    # plot charge to first axis
    ax = axarr[0]
    ax.set_title('Charge')
    sns.heatmap( df_charge.iloc[0:1,:], ax=ax, cmap='viridis_r', **kw_args )

    # plot spin to second axis
    ax = axarr[1]
    ax.set_title('Spin')
    sns.heatmap( df_spin, ax=ax, cmap='seismic_r', **kw_args )

    for ax in axarr:
        ax.tick_params(axis='both', labelrotation=0)

    fig.tight_layout()

    return fig

def plt_orb(df):
    '''create plot for orbital composition
    takes dataframe with orbital compositions and creates a plot at path

    requires
    df          dataframe
    path        path to save figure

    returns nothing'''

    y, x = len(df.index), len(df.columns) # autogenerate figure size
    fig, ax = plt.subplots(figsize=(x*.7, y*.3)) # create figure and axis

    sns.heatmap( # plot seaborn heatmap
        df, 
        cmap='twilight', vmin=0, vmax=1, 
        xticklabels=True, yticklabels=True, linewidth=0.5,
        annot=True, fmt='.2f', annot_kws={ 'fontfamily': 'monospace'},)

    ax.tick_params(axis='y', which='major')
    ax.tick_params(axis='both', labelrotation=.5)
    ax.tick_params(top=True, labeltop=True)

    fig.tight_layout()

    return fig

def run(orca2name):
    'run from command line'

    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('multi_out', nargs='?', help='MultiWFN output file', default='multi_hirsh.out')
    parser.add_argument('orca_out', nargs='?', help='Orca output file', default='orca1.out')
    parser.add_argument('-e', '--orb_erg', metavar='ERG', default=-1, type=float, 
        help='Only show orbitals with energy (in Ha) higher than ERG. Default: -1')
    parser.add_argument('-s', '--orb_sum', metavar='SUM', default=.6, type=float, 
        help='Only show orbitals that have a summed contribution (for atoms defined in orca2name) higher than SUM. Default: 0.6')
    parser.add_argument('-t', '--orb_thresh', metavar='THRESH', default=.02, type=float, 
        help='Remove orbital contributions on a single atom below THRESH. Default: 0.02')
    args = parser.parse_args()

    # define input files
    multi = Path(args.multi_out)
    lidi = multi.parent / 'LIDI.txt' 
    orbcomp = multi.parent / 'orbcomp.txt'
    orca = Path(args.orca_out)

    # define output files
    charge_spin_sheet = Path('charge_spin.xlsx')
    charge_spin_fig = Path('charge_spin.png')
    orb_sheet = Path('orbital_composition.xlsx')
    orb_fig = Path('orbital_composition.png')

    # define thresholds
    minsum = args.orb_sum
    minerg = args.orb_erg
    thresh = args.orb_thresh

    # start processing
    if multi.is_file():
        print('Now processing {} ...'.format(multi.name))
        # get orca indices, if orca2name dict is empty and orca output exists
        if orca.is_file() and not orca2name:
            print('    ... found {}'.format(orca.name))
            orca2name = get_orca2name(orca)

        # charge
        print('    ... getting charges'.format(multi.name))
        df_charge = get_hirsh_charge(multi.name)
        df_charge = rename_columns(df_charge, orca2name) 

        #spin
        print('    ... getting spin populations'.format(multi.name))
        df_spin = get_hirsh_spin(multi.name)
        df_spin = rename_columns(df_spin, orca2name) 

        # write output
        print('    ... writing charges and spin population:\n        {}\n        {}'.format(charge_spin_sheet, charge_spin_fig))
        df_charge_spin = pd.concat([df_charge, df_spin]) # merge charge and spin
        df_charge_spin.to_excel(charge_spin_sheet)
        fig_charge_spin = plt_charge_spin( df_charge, df_spin)
        save_fig(fig_charge_spin, path=charge_spin_fig)

        print('    ... done!')

        if orbcomp.is_file():
            print('Now processing {} ...'.format(orbcomp.name))

            # orbital compositions
            print('    ... reading {}'.format(orbcomp.name))
            df_orba = get_orb(multi, orbcomp, spin=0, orca2name=orca2name, minerg=minerg, minsum=minsum, thresh=thresh)
            df_orbb = get_orb(multi, orbcomp, spin=1, orca2name=orca2name, minerg=minerg, minsum=minsum, thresh=thresh)

            # excel sheet and figure
            print('    ... writing orbital compositions:\n        {}\n        {}'.format(orb_sheet, orb_fig))
            df_orbab = pd.concat([ df_orba, pd.DataFrame(data=np.nan, index=[''], columns=df_orba.columns), df_orbb ]) # concatenate a and b 
            df_orbab.to_excel(orb_sheet)
            fig_orb = plt_orb(df=df_orbab)
            save_fig(fig_orb, path=orb_fig)

            print('    ... done!')

        else:
            print('{} NOT found. Orbitals will not be analyzed'.format(orbcomp.name))

    else: 
        print('{} NOT found. Cannot continue and will exit now'.format(multi.name))
        sys.exit()

if __name__ == '__main__':
    run(orca2name)
