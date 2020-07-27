from sources.preprocess import read_csv
if __name__=='__main__':
    base_path1 = '/home/liyulian/data/CIDDS/CIDDS-001/traffic'
    base_path2 = '/home/liyulian/data/CIDDS/CIDDS-002/traffic'

    path = read_csv.read_ciddds_csvdata(base_path1, base_path2)