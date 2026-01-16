import re
import csv

with open('results', 'r') as fd:
    rows = fd.readlines()


clean_rows = [row for row in rows if len(row) > 2]
clean_rows = [row.replace('),', ')@') for row in clean_rows]
clean_rows = [row.replace('\',', '\'@') for row in clean_rows]
clean_rows = [row.replace(',', ';') for row in clean_rows]
clean_rows = [row.replace('@', ',') for row in clean_rows]

parsed_rows = list()
print(len(clean_rows))

for row in clean_rows:
    fields = row.split(';')
    model = fields[0]

    if model == 'ddpm':
        gen_time = fields[3]
        finds = re.findall(r'\d+\.\d+', gen_time)
        assert len(finds) == 2
        ddpm_gen_time = float(finds[0])
        ddim_gen_time = float(finds[1])

        fid_train = fields[4]
        finds = re.findall(r'\d+\.\d+', fid_train)
        assert len(finds) == 2
        ddpm_fid_train = float(finds[0])
        ddim_fid_train = float(finds[1])

        fid_test = fields[5]
        finds = re.findall(r'\d+\.\d+', fid_test)
        assert len(finds) == 2
        ddpm_fid_test = float(finds[0])
        ddim_fid_test = float(finds[1])

        is_test = fields[6]
        finds = re.findall(r'\d+\.\d+', is_test)
        assert len(finds) == 4
        ddpm_is_mu = float(finds[0])
        ddpm_is_sigma = float(finds[1])
        ddim_is_mu = float(finds[2])
        ddim_is_sigma = float(finds[3])

        parsed_row_ddpm = {
            'model': fields[0],
            'run': int(fields[1]),
            'train_time': float(fields[2]),
            'gen_time': ddpm_gen_time,
            'fid_train': ddpm_fid_train,
            'fid_gen': ddpm_fid_test,
            'is_mu': ddpm_is_mu,
            'is_sigma': ddpm_is_sigma,
        }

        parsed_row_ddim = {
            'model': fields[0].replace('p', 'i'),
            'run': int(fields[1]),
            'train_time': float(fields[2]),
            'gen_time': ddim_gen_time,
            'fid_train': ddim_fid_train,
            'fid_gen': ddim_fid_test,
            'is_mu': ddim_is_mu,
            'is_sigma': ddim_is_sigma,
        }

        parsed_rows.append(parsed_row_ddpm)
        parsed_rows.append(parsed_row_ddim)
    else:
        is_test = fields[6]
        finds = re.findall(r'\d+\.\d+', is_test)
        assert len(finds) == 2
        is_mu = float(finds[0])
        is_sigma = float(finds[1])

        parsed_row = {
            'model': fields[0],
            'run': int(fields[1]),
            'train_time': float(fields[2]),
            'gen_time': float(fields[3]),
            'fid_train': float(fields[4]),
            'fid_gen': float(fields[5]),
            'is_mu': is_mu,
            'is_sigma': is_sigma,
        }
        parsed_rows.append(parsed_row)

with open('results_clean.csv', 'w') as fd:
    writer = csv.DictWriter(fd, fieldnames=parsed_rows[0].keys())
    writer.writeheader()
    writer.writerows(parsed_rows)
