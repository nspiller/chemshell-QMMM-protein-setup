#!/usr/bin/env python3

import argparse
from pathlib import Path
import re
import sys

bohr_radius = 5.29177210903e-11
bohr2ang = lambda x: x * bohr_radius * 1e10
ang2bohr = lambda x: x / bohr_radius / 1e10

def get_atom_indices(chm_list_file):
    '''extract list of atoms from chm file such as qmatoms or act

    returns list of ints (atom indices)''' 

    with open(chm_list_file) as f:
        line = f.readline().strip()
        l1, l2 = line.split('{')

        region = l1.split()[1]
        indices = [ int(i) for i in l2.rstrip('}').split() ]
        
    return indices

def read_fragment(frag, atom_indices):
    '''extract coordinates for atoms (in ang) from fragment file'''

    with open(frag) as f:
        frag_data = f.readlines()
        coord_list = []

        for index in atom_indices:
            i = index + 3
            line = frag_data[i]
            l = line.split()
            a = l[0]
            x, y, z = [ float(i) for i in l[1:4] ]

            coord_list.append([a, x, y, z])

        return coord_list

def update_fragment(frag_old, frag_new, atom_indices, coord_list):
    '''write new frag with coordinates of atom_indices change
    expects the coord_list to be in the same order as atom_indices'''
                
    with open(frag_old) as f:
        frag_data = f.readlines()

    with open(frag_new, 'w') as f:
        atom_index_counter = 0
        for index in atom_indices:
            i = index + 3
            a, x, y, z = coord_list[atom_index_counter]
            frag_data[i] = '{:2} {:5.14e} {:5.14e} {:5.14e}\n'.format(a, x, y, z)  
            atom_index_counter += 1

        f.writelines(frag_data)
            

def read_xyz(xyz_file, convert=True):
    '''read xyz file
    return list of ['atom_name', float(x), float(y), float(y)]'''

    with open(xyz_file) as f:
        n = int(f.readline().strip()) # number of atoms
        f.readline() # comment line

        coord_list = []
        for line in f:
            l = line.split()
            a = l[0]
            x, y, z = [ float(i) for i in l[1:4] ]
            if convert:
                x, y, z = [ ang2bohr(i) for i in [x, y, z] ]
            
            coord_list.append([ a, x, y, z ])

        if len(coord_list) != n:
            sys.exit('Invalid xyz file: Header number {} does not match number of coordinate entries {}'.format(n, len(coord_list)))

        return coord_list


def write_xyz(coord_list, xyz_file, convert=True):
    '''write list of ['atom_name', float(x), float(y), float(y)]'''

    with open(xyz_file, 'w') as f:

        f.write(str(len(coord_list)) + '\n') # first line: number of atoms
        if convert: # second line: comment
            f.write('coordinates written in Ang with {} \n'.format(Path(__file__).name)) 
        else:
            f.write('coordinates written in Bohr with {} \n'.format(Path(__file__).name))

        for l in coord_list:
            a, x, y, z = l
            if convert:
                x, y, z = [ bohr2ang(i) for i in [x, y, z] ]
            f.write('{:4s}{:5.14e} {:5.14e} {:5.14e}\n'.format(a, x, y, z))


def run():
    parser = argparse.ArgumentParser(
        description='''Extract or update coordinates (in Ang) in a chemshell fragment file (in bohr).
        test''')
    parser.add_argument('fragment_file', metavar='FRAGMENT', 
        help='Chemshell fragment file')
    parser.add_argument('region_file', metavar='REGION', 
        help='chemshell file containing the list of atoms (e.g. act or qmatoms')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-u', '--update', action='store_true',
        help='update FRAGMENT from REGION.xyz')
    group.add_argument('-x', '--extract', action='store_true',
        help='extract REGION.xyz from FRAGMENT')
    parser.add_argument('-b', '--bohr', action='store_false', 
        help='Write and read REGION.xyz in Bohr)')
    args = parser.parse_args()

    frag = Path(args.fragment_file)
    region = Path(args.region_file)
    update = bool(args.update)
    extract = bool(args.extract)
    xyz = Path(region.name + '.xyz')
    convert = bool(args.bohr)

    atom_indices = get_atom_indices(region)

    if update:
        if convert:
            coord_list = read_xyz(xyz)
            print('INFO: read coordinates (in Ang) from {}'.format(xyz))
        else:
            coord_list = read_xyz(xyz, convert=False)
            print('INFO: read coordinates (in Bohr) from {}'.format(xyz))
        
        frag_new = Path(frag.name) 
        frag = frag.rename(frag.name + 'bak')
        print('INFO: backed up fragment file as: {}'.format(frag))

        update_fragment(frag, frag_new, atom_indices, coord_list)
        print('INFO: updated fragment file: {}'.format(frag_new))

        
    if extract:
        coord_list = read_fragment(frag, atom_indices)
        print('INFO: extracted coordinates for {} from {}'.format(region, frag))

        xyz_new = Path(xyz.name) 
        if xyz_new.exists():
            xyz = xyz.rename(xyz.name + 'bak')
            print('INFO: backed up xyz file as {}'.format(xyz))

        if convert:
            write_xyz(coord_list, xyz_new)
            print('INFO: written xyz file (in Ang) as {}'.format(xyz_new))
        else:
            write_xyz(coord_list, xyz_new, convert=False)
            print('INFO: written xyz file (in Bohr) as {}'.format(xyz_new))



if __name__ == '__main__':

    run()


