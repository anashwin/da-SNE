import pandas as pd

data_dir = '/scratch2/ashwinn/SNE/geosketch_v/bin/data/'

namespace = 'mouse_gastr_early/atlas/'

filename = 'meta.csv'


df = pd.read_csv(data_dir + namespace + filename)

column = 'stage'

out_dir = 'data/'

df[column].to_csv(out_dir + namespace[:namespace.find('/')] + '_label_' + column + '.csv')

