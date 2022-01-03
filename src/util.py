import os
def create_dataframe_if_not_exists(dataframe, output_file):
    if os.path.isfile(output_file):
        print(f'file exists: {output_file}')
    else:
        dataframe.to_csv(output_file, index=False)
        print(f'file created: {output_file}')