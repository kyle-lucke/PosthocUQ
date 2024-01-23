import os
import argparse

import pandas as pd

pd.set_option('display.precision', 1)

def create_argparser():
  parser = argparse.ArgumentParser()

  parser.add_argument('input_dir')

  return parser
  
def main():
  parser = create_argparser()

  args = parser.parse_args()

  dataset_results = list(os.listdir(args.input_dir))
  
  ex_df = pd.read_csv(os.path.join(args.input_dir, dataset_results[0]))
  
  data = {"Metric": ex_df.columns}
  for ds_res in dataset_results:
    ds_name = ds_res.replace('.csv', '')

    res_path = os.path.join(args.input_dir, ds_res)
    
    df =  pd.read_csv(res_path)
    
    data[ds_name] = df.values[0].tolist()

  tbl = pd.DataFrame(data)

  tbl = tbl.rename(columns={"FashionMNIST": "FMNIST"})
  
  auroc = tbl[tbl["Metric"].str.contains("AUROC")]
  meta_model_auroc = tbl[~tbl["Metric"].str.contains("BASE")]

  meta_model = tbl[~tbl["Metric"].str.contains("BASE")]
  base_model = tbl[tbl["Metric"].str.contains("BASE")]

  auroc_meta = meta_model[meta_model["Metric"].str.contains("AUROC")]
  auroc_base = base_model[base_model["Metric"].str.contains("AUROC")]

  aupr_meta = meta_model[meta_model["Metric"].str.contains("AUPR")]
  aupr_base = base_model[base_model["Metric"].str.contains("AUPR")]

  if 'OOD' in args.input_dir:
  
    if 'CIFAR' in args.input_dir:
      out_col_order = ["Metric", "SVHN", "FMNIST", "LSUN", "TinyImage", "Corrupted"]
    else:
      out_col_order = ["Metric", "Omniglot", "FMNIST", "KMNIST", "CIFAR10", "Corrupted"]
  
    # reorder columns to match paper
    auroc_meta = auroc_meta[out_col_order]
    auroc_base = auroc_base[out_col_order]

    # reorder columns to match paper
    aupr_meta = aupr_meta[out_col_order]
    aupr_base = aupr_base[out_col_order]
  
  print('Meta model ARUOC scores')
  print(auroc_meta.to_string(index=False))
  print()

  print('Base model AUROC scores')
  print(auroc_base.to_string(index=False))
  print()

  print('Meta model AUPR scores')
  print(aupr_meta.to_string(index=False))
  print()

  print('Base model AUPR scores')
  print(aupr_base.to_string(index=False))
  print()
  
if __name__ == '__main__':
  main()
