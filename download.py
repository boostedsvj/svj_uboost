import os, os.path as osp
import seutils

from common import logger


DATADIR = osp.join(osp.dirname(osp.abspath(__file__)), 'data')


def copy(tup):
    src, dirname = tup
    logger.info('Copying %s -> %s', src, DATADIR)
    seutils.cp(src, DATADIR, recursive=True, parallel=4)
    wrong_named_dir = osp.join(DATADIR, osp.basename(src))
    correct_named_dir = osp.join(DATADIR, dirname)
    logger.info('Renaming %s -> %s', wrong_named_dir, correct_named_dir)
    os.rename(wrong_named_dir, correct_named_dir)


def main():
    if not osp.isdir(DATADIR): os.makedirs(DATADIR)

    signal_dir = 'root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/bdt_features/signal_nov02_truthcone'
    bkg_dir = 'root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/bdt_features/bkg_nov14/HADD'
    signal_dir_notruthcone = 'root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/bdt_features/signal_nov02'

    fn_args = [
        (signal_dir, 'signal'),
        (signal_dir_notruthcone, 'signal_notruthcone'),
        (bkg_dir, 'bkg'),
        ]

    import multiprocessing as mp
    p = mp.Pool(2)
    p.map(copy, fn_args)
    p.close()
    p.join()



if __name__ == '__main__':
    main()