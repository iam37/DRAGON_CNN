import tarfile

from astropy.table import Table


def imgdwnldr_gen(args):
    df, num_start, num_stop, fltr = args

    t = Table()
    t["rerun"] = ["pdr3_wide"] * len(df[num_start:num_stop])
    t["filter"] = [fltr] * len(df[num_start:num_stop])
    t["ra"] = df["ra"][num_start:num_stop]
    t["dec"] = df["dec"][num_start:num_stop]
    t["type"] = ["coadd"] * len(df[num_start:num_stop])
    t["sh"] = ["8asec"] * len(df[num_start:num_stop])
    t["sw"] = ["8asec"] * len(df[num_start:num_stop])

    t["name"] = df["name"][num_start:num_stop]

    return t

