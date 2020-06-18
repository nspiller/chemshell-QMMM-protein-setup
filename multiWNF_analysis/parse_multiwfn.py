#!/usr/bin/env python3

import numpy as np
import pandas as pd
import re

################
## User Input ##
################

idx2name = { # this dict converts orca atom indices to sensible names (required)
    15: 'Mo',
    16: 'Fe1',
    17: 'Fe2',
    18: 'Fe3',
    19: 'Fe4',
    20: 'Fe5',
    21: 'Fe6',
    22: 'Fe7',
}


# orbital analysis
minsum = 0.6 # consider only orbitals with at leats this much on above atoms
thresh = 0.02 # remove single contributions below this value

###############
## HIRSHFELD ##
###############
def get_fuzzy(output):
    '''
    ATTENTION: very unspecific parsing 
    return df of values of interation of some function in fuzzy volumes 
    (e.g. spin)

    ATTENTION: atom indices in MultiWFN start with 1. 

    matches first occurence of 

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

            if 'Atomic space Value % of sum % of sum abs' in ' '.join(line.split()): # start here
                parse = True
                continue
            elif 'Summing up above values:' in line: # stop here 
                parse = False
                continue

            if parse:
                l = line.split()
                i = int(l[0].split('(')[0]) - 1 # counting from 0
                c = float(l[-3])
                df.loc['Hirshfeld spin', i] = c

    df = change_columns(df, idx2name, idx2name.values())

    return df

def get_hirsh_cm5(output):
    '''
    return dataframe with CM5 and Hirshfeld 

    counting from 0
    matches first occurence of and terminates on empty line


                         ======= Summary of CM5 charges =======
     Atom:    1C   CM5 charge:   -0.227214  Hirshfeld charge:   -0.077632
     Atom:    2H   CM5 charge:    0.079908  Hirshfeld charge:    0.027221
     Atom:    3H   CM5 charge:    0.092847  Hirshfeld charge:    0.041910
     Atom:    4C   CM5 charge:   -0.155276  Hirshfeld charge:   -0.048841
     Atom:    5H   CM5 charge:    0.085117  Hirshfeld charge:    0.027582
     Atom:    6H   CM5 charge:    0.095197  Hirshfeld charge:    0.038745
     Atom:    7C   CM5 charge:   -0.037447  Hirshfeld charge:    0.010199
     Atom:    8H   CM5 charge:    0.088796  Hirshfeld charge:    0.019746
     Atom:    9H   CM5 charge:    0.104493  Hirshfeld charge:    0.034564
     Atom:   10N   CM5 charge:   -0.431012  Hirshfeld charge:   -0.072936
     Atom:   11H   CM5 charge:    0.293571  Hirshfeld charge:    0.097161
     Atom:   12C   CM5 charge:    0.401691  Hirshfeld charge:    0.197201
     Atom:   13N   CM5 charge:   -0.552805  Hirshfeld charge:   -0.106493
     Atom:   14H   CM5 charge:    0.352662  Hirshfeld charge:    0.164065
     Atom:   15H   CM5 charge:    0.359951  Hirshfeld charge:    0.171027
     Atom:   16N   CM5 charge:   -0.546935  Hirshfeld charge:   -0.105064
     '''

    df = pd.DataFrame()

    parse = False
    with open(output) as f:
        for line in f:
            if '======= Summary of CM5 charges =======' in line:
                parse = True
                continue
            elif 'Summing up all CM5 charges:' in line:
                parse = False
                continue

            if parse:
                l = line.split()
                i = int(re.findall('^\d+', l[1])[0]) - 1
                hirsh = float(l[-1])
                cm5 = float(l[-4])
                df.loc['Hirshfeld charge', i] = hirsh
                df.loc['CM5 charge', i] = cm5

    df = change_columns(df, idx2name, idx2name.values())

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


def get_orbcomp(output, orbrange=None):
    '''
    returns pandas dataframe from orbcomp.txt, counting from 0

    optional
    orbrange        tuple of first and last index (counting from 0)
    '''

    df = pd.DataFrame() # empty DataFrame

    with open(output) as f:
        for line in f: 
            
            if 'Orbital' in line: # get orbital number from line
                l = line.split()
                n = int(l[1]) - 1 # adjust to ORCA counting
                continue

            if orbrange: # dont parse if out of range
                if orbrange[0] > n:
                    continue
                elif orbrange[1] < n:
                    break

            l = line.split()
            i = int(l[0]) - 1 # adjust to ORCA counting
            c = float(l[1]) / 100 # convert to 0 < c < 1
            df.loc[n, i] = c # write in df: n = atom/basin (starts with 0) and i = contribution 

    return df

###########
## BADER ##
###########
def get_basin_dict(output):
    '''
    returns a dictionary with {basin no: atom index} 
    
    basin numbering and atom numbering start with 0

    expects this to be present:

    Integrating in trust sphere...
    Attractor     1 corresponds to atom    26 (S )
    The trust radius of attractor     1 is     2.268 Bohr

    Attractor     2 corresponds to atom    32 (S )
    The trust radius of attractor     2 is     2.121 Bohr

    Attractor     3 corresponds to atom    29 (S )
    ...
    '''
    d = {} # empty dict

    with open(output) as f:
        for line in f:

            if 'corresponds to atom' in line: # only these lines
                l = line.split()
                if l[0] == 'Attractor':
                    basin = int(l[1]) - 1  # counting from 0
                    atom = int(l[5]) - 1 # counting from 0
                    d[basin] = atom # write dict

    return d


def get_basin_charges(output):
    '''
    read multiwfn output and returns datafrme

    counting of atoms starts with 0

    function starts reading at
     The atomic charges after normalization and atomic volumes:
          1 (C )    Charge:    0.104633     Volume:    68.539 Bohr^3   
          2 (H )    Charge:   -0.037355     Volume:    51.015 Bohr^3   
          3 (H )    Charge:   -0.038537     Volume:    50.906 Bohr^3 
    ''' 
    df = pd.DataFrame()

    parse = False
    with open(output) as f:
        for line in f:
            if 'The atomic charges after normalization and atomic volumes:' in line:
                parse = True
                continue
            elif len(line.split()) == 0:
                parse = False
                continue

            if parse:
                l = line.split()
                i = int(l[0]) - 1 # start at 0
                c = float(l[-4])
                df.loc['Bader charge', i] = c 

    df = change_columns(df, idx2name, idx2name.values())

    return df 


def get_basin_spin(output):
    '''
    read multiwfn output and returns datafrme

    counting of atoms starts with 0

    function starts reading at

    Total result:
    Atom       Basin       Integral(a.u.)   Vol(Bohr^3)   Vol(rho>0.001)
    1 (C )      35          0.00943089       191.980        68.587
    2 (H )      29          0.00322007       664.040        50.814
    3 (H )      36          0.00118925       564.645        50.894
    ...
    ''' 
    df = pd.DataFrame()

    parse = False
    with open(output) as f:
        for line in f:
            if 'Atom       Basin       Integral(a.u.)   Vol(Bohr^3)   Vol(rho>0.001)' in line:
                parse = True
                continue
            elif 'Sum of above integrals:' in line:
                parse = False
                continue

            if parse:
                l = line.split()
                i = int(l[0]) - 1 # start at 0
                c = float(l[-3])
                df.loc['Bader spin', i] = c 

    df = change_columns(df, idx2name, idx2name.values())

    return df 

########################
## General processing ##
########################

def get_occ_orbitals(multiout):
    '''
    parse multiwfn output to get orbital range for occupied alpha and beta orbitals
    counting starts at 0

    return two tuples: (alpha_i, alpha_f), (beta_i, beta_f)
    '''

    alpha, beta = np.array([], dtype=int), np.array([], dtype=int)

    with open(multiout) as f:
        for line in f:
            l = line.split()
            if len(l) == 8: 
                if l[0] == 'Orbital:' and l[-2] == 'Type:':

                    print(line)
                    n = int(l[1])
                    s = l[-1]
                    o = float(l[-3])

                    if o == 1.0:
                        if s == 'Alpha':
                            alpha = np.append(alpha, [n])
                        elif s == 'Beta':
                            beta = np.append(beta, [n])
                        else:
                            raise Exception('Parsing error in line: {}'.format(line))


    alpha_range = ( alpha.min(), alpha.max() )              
    beta_range = ( beta.min(), beta.max() )

    return alpha_range, beta_range

    return (ai, af), (bi, bf)

def change_rows(df, d, l):
    '''
    convert the indices of dataframe

    df      dataframe
    d       dict with oldidx: newidx
    l       list of newidx

    returns df with only newidx as indices
    '''
    df.rename(index=d, inplace=True)
    df = df.loc[l]

    return df

def change_columns(df, d, l):
    '''
    convert the columns of dataframe

    df      dataframe
    d       dict with oldcol: newcol
    l       list of newcol

    returns df with only newcol as columns
    '''
    df.rename(columns=d, inplace=True)
    df = df.loc[:,l]

    return df

def nicefy_orbcomp(df, minsum, thresh):
        
    df = change_columns(df, idx2name, idx2name.values())
    
    # delete uninteresting orbitals from dataframe
    df.loc[:, 'sum'] = df.loc[:, : ].sum(axis=1) # create sum column
    df = df.loc[ df['sum'] > minsum, : ]

    df = df.apply( lambda x: [y if y > thresh else np.nan for y in x])
    
    df.sort_values([ i for i in idx2name.values()], inplace=True, ascending=False) # sort


    return df

if __name__ == '__main__':
    from pathlib import Path
    import argparse
    import sys

    parser = argparse.ArgumentParser()
#    parser.add_argument('method', help='specify either hirsh (for Hirshfeld) or bader (for Bader QTAIM)')
    #TODO: force overwrite function
    args = parser.parse_args()
    # TODO: logging function

    # files relevant here
    hirsh_out = Path('multi_hirsh.out')
    bader_out = Path('multi_bader.out')
    lidi_out = Path('LIDI.txt')
    orbcomp_out = Path('orbcomp.txt')

    # output
    xlsx_out = 'analysis.xlsx'

    # bools for parsing of each file
    hirsh, bader, lidi, orbcomp = [ False for _ in range(4) ] # set True later if file found

    if hirsh_out.is_file() and bader_out.is_file():
        print('Found both {} and {}: I do not know what to do'.format(hirsh_out.name, bader_out.name))
        sys.exit()
    elif hirsh_out.is_file():
        print('{} found, setting parsing environment to Hirshfeld'.format(hirsh_out.name))
        hirsh = True
    elif bader_out.is_file():
        print('{} found, setting parsing environment to Bader'.format(bader_out.name))
        bader = True
    else:
        print('Found neither {} and {}: I do not know what to do'.format(hirsh_out.name, bader_out.name))
        sys.exit()

    if lidi_out.is_file():
        print('{} found'.format(lidi_out.name))
        lidi = True
    else: 
        print('{} NOT found'.format(lidi_out.name))
        
    if orbcomp_out.is_file():
        print('{} found'.format(orbcomp_out.name))
        orbcomp = True
    else:
        print('{} NOT found'.format(orbcomp_out.name))


    with pd.ExcelWriter(xlsx_out, engine='xlsxwriter') as writer: # necessary to write to one xlsx document
        # Hirshfeld
        if hirsh:
#            # charge
#            df_charge = get_hirsh_cm5(hirsh_out.name)
#
#            #spin
#            df_spin = get_fuzzy(hirsh_out.name)
#
#            pd.concat([df_charge, df_spin]).to_excel(writer, sheet_name='Hirshfeld charge, spin')
#
#            # LIDI
#            if lidi:
#                df_lidi_deloca, df_lidi_delocb, df_lidi_delocab, df_lidi_loca, df_lidi_locb, df_lidi_locab = get_lidi(lidi_out) 
#
            # orbital analysis
            if orbcomp:
                arange, brange = get_occ_orbitals(hirsh_out)
                
                df_orbcompa = get_orbcomp(orbcomp_out, arange)
                df_orbcompb = get_orbcomp(orbcomp_out, brange)
                
                df_orbcompa = nicefy_orbcomp(df_orbcompa, minsum, thresh)
                df_orbcompb = nicefy_orbcomp(df_orbcompb, minsum, thresh)

                df_orbcompa.to_excel(writer, sheet_name='Hirshfeld alpha orb comp')
                df_orbcompb.to_excel(writer, sheet_name='Hirshfeld beta orb comp')


        # Bader
        elif bader:
            # dict to convert basin (starts at 0) to atoms (starts at 0)
            basin2atom = get_basin_dict(bader_out.name)

            # charge (in ATOM basis)
            df_charge = get_basin_charges(bader_out.name)
            
            # spin (in ATOM basis)
            df_spin = get_basin_spin(bader_out.name)

            # LIDI (in BASIN basis)
            if lidi:
                df_lidi_deloca, df_lidi_delocb, df_lidi_delocab, df_lidi_loca, df_lidi_locb, df_lidi_locab = get_lidi(lidi_out) 

            # orbital analysis (in BASIN basis)
            if orbcomp: 
                arange, brange = get_occ_orbitals(bader_out)
                df_orbcomp = get_orbcomp(orbcomp_out)
                # TODO: test orbcomp Bader