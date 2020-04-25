from pandas import read_csv, merge, concat, DataFrame
from tqdm import tqdm_notebook as T
from numpy import argsort,array


def load_labels(train_dir, val_dir):
    '''load table with directories'''
    train_df = read_csv('train_labels.csv')
    val_df = read_csv('val_labels.csv')
    train_df['path'] = train_dir+ train_df['img_name']
    val_df['path']  =  val_dir+ val_df['img_name']
    df = concat([train_df, val_df], ignore_index=True)
    valIndex = [i for i in range(len(train_df), len(df))]
    return df, valIndex

def load_prediction_to_top_3(preds,fnames, order_df):
  col = ['img_name']
  test_df = DataFrame(fnames, columns=col)
  test_df['label'] = ''
  predictions = array(preds).reshape(len(preds), 251)
  for i, pred in T(enumerate(predictions), total=len(predictions)):
    test_df.loc[i, 'label'] = ' '.join(str(int(i)) for i in argsort(pred)[::-1][:3])
  
  test_df = merge(order_df['img_name'], test_df )
  return test_df