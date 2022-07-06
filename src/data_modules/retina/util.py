'''Utility functions for data modules'''
import mmap
import hashlib

__all__ = [
    'verify_sha256',
]

def verify_sha256(path, checksum):
    '''Verify sha256 checksum of a file
    Parameteters
    -----------
    path : str
      Path to file to verify

    checksum : str
      Expected checksum as hexstring

    Returns
    -------
    bool
      True if checksum of file match `checksum`, False otherwise
    '''
    hasher = hashlib.sha256()
    with open(path, 'r+b') as infile:
        mmapped = mmap.mmap(infile.fileno(), 0)
        hasher.update(mmapped)
    return hasher.hexdigest() == checksum
