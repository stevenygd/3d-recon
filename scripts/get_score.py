import os
import numpy as np
import csv
import argparse

parser = argparse.ArgumentParser('')
parser.add_argument('--prefix', type=str, default='chairs_pose0.5')
parser.add_argument('--categories', type=str, nargs='+', help='Categories to evaluate.')
args = parser.parse_args()
print(args.categories)

if args.categories is None:
    CATE = [
        # Train
        "airplanes",
        "cars",
        "chairs",
        "displays",
        "phones",
        "speakers",
        "tables",
        # VAL
        "benches",
        "vessels",
        "cabinets",
        "lamps",
        "sofas",
    ]
else:
    CATE = args.categories

METRICS = [
    'maxiou', 'avgprc', 'iout04', 'iout05'
]

data_val = {}  # model -> category -> score
data_test = {} # model -> category -> score

ROOT = 'log/train_gan_cyc_encdec_randomvp/'
for datasetting in os.listdir(ROOT):
    if args.prefix not in datasetting:
        continue
    print(datasetting)
    for dir_name in os.listdir(os.path.join(ROOT, datasetting)): # data setting or dataset
        if not os.path.isdir(os.path.join(ROOT, datasetting, dir_name)):
            continue
        key = "%s_%s" % (datasetting, dir_name)
        data_val[key] = {cate:{} for cate in CATE}
        data_test[key] = {cate:{} for cate in CATE}
        for fname in os.listdir(os.path.join(ROOT, datasetting, dir_name)):
            if '.txt' not in fname:
                continue
            data = data_test if 'test' in fname else data_val
            with open(os.path.join(ROOT, datasetting, dir_name, fname)) as f:
                is_header = True
                headers = None
                for l in f:
                    if is_header:
                        headers = l.strip().split()
                        is_header = False
                        continue

                    row = l.strip().split()
                    for cate in CATE:
                        if cate in row[0]:
                            if 'maxiou' in fname:
                                data[key][cate]['maxiou'] = row[1]
                            elif 'avgprc' in fname:
                                data[key][cate]['avgprc'] = row[2]
                            elif 'iout04' in fname:
                                data[key][cate]['iout04'] = row[3]
                            elif 'iout05' in fname:
                                data[key][cate]['iout05'] = row[4]
                            else: raise Exception("Invalid fname:%s"%fname)

out_dir = 'results'
if not os.path.exists('results'):
    os.makedirs('results')

for k in data_val.keys():
    for data, split in [(data_val, 'val'), (data_test, 'test')]:
        out_fname = os.path.join(out_dir, "%s-%s-scores.csv"%(k, split))
        with open(out_fname, 'w') as outf:
            outf.write("category\tsplit\tMaxIoU\tAP\tIoU(t>0.4)\tIou(t>0.5)\n")
            for cate in CATE:
                try:
                    scores = [data[k][cate][m] for m in METRICS]
                    row = [cate, split] + scores
                    outf.write("%s\n"%("\t".join(row)))
                except:
                    outf.write("%s\t%s\n"%(cate, split))
        print(out_fname)
