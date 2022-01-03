import os
import pickle5 as pickle
def create_dataframe_if_not_exists(dataframe, output_file):
    if os.path.isfile(output_file):
        print(f'file exists: {output_file}')
    else:
        dataframe.to_csv(output_file, index=False)
        print(f'file created: {output_file}')
def create_pickle_if_not_exists(pickle_obj, output_file):
    if os.path.isfile(output_file):
        print(f'file exists: {output_file}')
    else:
        with open(output_file, 'wb') as f:
            pickle.dump(pickle_obj, f, protocol=5)
        print(f'file created: {output_file}')