

path = '/home/liyulian/code/CIDDS/sources/utils/data_features_002.csv'
save_path = '/home/liyulian/code/CIDDS/repositories/xgb_model/'
log_path = '/home/liyulian/code/CIDDS/repositories/'
X, Y = get_features(path)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)