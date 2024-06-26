import os 

class config:
    cwd = os.getcwd()
    data_fp = os.path.abspath(os.path.join(cwd, 'data'))
    input_data_fp = os.path.join(data_fp, 'input_data')
    proc_data_fp = os.path.join(data_fp, 'proc_data')
    output_data_fp = os.path.join(data_fp, 'output_data')
    if not os.path.exists(input_data_fp):
        os.mkdir(input_data_fp)
    if not os.path.exists(proc_data_fp):
        os.mkdir(proc_data_fp)
    if not os.path.exists(output_data_fp):
        os.mkdir(output_data_fp)

if __name__ == '__main__':
    config