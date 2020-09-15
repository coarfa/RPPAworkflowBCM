import os
import pandas as pd

def main(exp_name, file_path):
    path = file_path
    experiment = exp_name  #experiment = "RPPA0028"
    print (experiment,'\n', path)
    
    dir_name = os.path.dirname(path)
    dirs = sorted(os.listdir(dir_name))
    
    barcode_data =  pd.read_table(path, sep="\t", index_col=0)
    print (barcode_data)
    
    for folder in dirs:
        if folder.startswith("Scanned_" + experiment):
            folder_with_path = os.path.join(dir_name, folder)
            path2 = folder_with_path
            files = sorted(os.listdir(path2))
            
            for filename in files:
                if filename.endswith('.tif'):
                    print (filename)
                    cols = filename.rstrip().split('_')
                    if len(cols) == 5:
                        barcode = int((cols[0])[4:])
                        #print barcode[1]
                        if barcode in barcode_data.index:
                            print ('0000' + str(barcode) + ':\t' + 'OK')
                            new_filename = cols[0] + "_" + cols[1] + "_" + cols[2] + "_" + barcode_data.loc[barcode, "name"] + "_" + cols[4]
                            
                            file_with_path =  os.path.join(folder_with_path, filename)
                            newfile_with_path =  os.path.join(folder_with_path, new_filename)
                            os.rename(file_with_path, newfile_with_path)
                        else:
                            print (filename, '\tNOT renamed: barcode not found')
                    else:
                        print (filename, '\tNOT renamed: name format invalid')
        # else:
        #     print ('Error: Image folder NOT found or bad folder name  ')
                  
if __name__ == '__main__':
    main(input1, input2)	