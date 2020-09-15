import numpy as np
import pandas as pd

# =============================================================================
# Discription: Array to Plate. Generate plates. This is the 6 plate version
# 				Each sample group starts with a new block.
# Input: Array assignment xlsx file, e.g 'output_sample_p6_assignment.xlsx' 
# Author: Dimuthu Perera 
# Year: 2020
# version: for GUI
# =============================================================================

def main(input_file):
    # well plate generation and output to excel
    DF = pd.read_excel(input_file)
    
    slide = pd.DataFrame()
    slide_no = 1
    for l in reversed(range(3)):
        for k in range(4):
            slide = pd.DataFrame()
            for m in range(10):
                list2 = []
                for n in range(4):
                    list2.append(DF.iloc[4*m + k, 3*n + l]) # this is kth row and the lth column on every block
                #print list
                slide = slide.append(pd.DataFrame([list2], columns=['D','C','B','A']), ignore_index=True)
    #        print 'ROW'
    #        print slide
            if np.remainder(k,2) == 0:
                slide1 = slide
                # ff = k/2
            else:
                slide1 = slide1.append(slide, ignore_index=True)
                slide1 = slide1.append(pd.DataFrame(np.full((4,4), '-'), columns=['D','C','B','A']), ignore_index=True)
                slide1 = slide1.sort_index(axis=1, ascending=True).transpose()
    #            offset = pd.DataFrame(np.full((4*ff,24), '-'))
    #            offset = offset.append(slide1, ignore_index=True)
                slide1.to_csv('RPPA_CSV_p6 P{}.csv'.format(slide_no), header=False, index=False)
    #            offset.to_csv('RPPA_CSV.v6 P{}.csv'.format(slide_no), header=False, index=False)
                slide_no += 1

    DF.to_csv('test_array6.csv', header=False, index=False)
    
if __name__ == '__main__':
    main(input_file)	