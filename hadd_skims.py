# originally from https://github.com/boostedsvj/svj_local_scripts

import glob, os, os.path as osp
import seutils
import svj_ntuple_processing as svj

def expand_wildcards(pats):
    expanded = []
    for pat in pats:
        if '*' in pat:
            if seutils.path.has_protocol(pat):
                expanded.extend(seutils.ls_wildcard(pat))
            else:
                expanded.extend(glob.glob(pat))
        else:
            expanded.append(pat)
    return expanded

def process_directory(tup):
    directory, outfile = tup
    npzfiles = expand_wildcards([directory+'*.npz'])[0]
    svj.logger.info(f'Processing {directory} -> {outfile} ({len(npzfiles)} files)')
    cols = []
    for f in npzfiles:
        try:
            cols.append(svj.Columns.load(f, encoding='latin1'))
        except Exception as e:
            svj.logger.error(f'Failed for file {f}, error:\n{e}')
    concatenated = svj.concat_columns(cols)
    concatenated.save(outfile)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stageout', type=str, required=True)
    parser.add_argument('-n', '--nthreads', default=10, type=int)
    parser.add_argument('directories', nargs='+', type=str)
    args = parser.parse_args()
    if args.stageout[-1]!='/':
        args.stageout += '/'

    directories = expand_wildcards(args.directories)

    fn_args = []
    for d in directories:
        outfile = args.stageout+'/'.join(path.split('/')[-3:-1])+'.npz'
        fn_args.append((d, outfile))

    import multiprocessing as mp
    p = mp.Pool(args.nthreads)
    p.map(process_directory, fn_args)
    p.close()
    p.join()
