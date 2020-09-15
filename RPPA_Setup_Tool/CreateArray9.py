import numpy as np
import pandas as pd

# =============================================================================
# Discription: Create Array. Assigns the samples. This is the 9 plate version
# Input(s): use sample sheet as is
# 		 Each sample group starts one after another 
# Author: Dimuthu Perera 
# Year: 2020
# Version: 1.4-for gui
# =============================================================================

def Combined_cols(array, n):
    app_cols = pd.DataFrame(data=array[n, 0])
    for c2 in range(4)[1:]:
        temp = pd.DataFrame(data=array[n, c2])
        app_cols = pd.concat([app_cols, temp], axis=1, ignore_index=True)
    return app_cols

def assign2(loc, grp, pl):
    Plate = pl
    [x, y, m, n] = loc 
    a = 0
    row = m 
    label = grp[grp.columns[1]].tolist()
    
    while a < len(label):
        col = np.remainder(a + n, 3)      
        if col + 1 == 3:
            Plate[x, y][row, col] = label[a] # 1
            row += 1
        else:   
            Plate[x, y][row, col] = label[a] # 2
        a += 1
        if row + 1 == 7:
            y += 1
            row = 0
            if y == 4:
                y = 0
                x += 1
                row = 0
                if x == 10:
                    return

def df_append(df1, df2): 
    new_cols = {x: y for x, y in zip(df2.columns, df1.columns)}
    df = df1.append(df2.rename(columns=new_cols), ignore_index=True)
    return df

def fill_empty(index): 
    fill = pd.DataFrame(np.full((18*PI_spots.iloc[index]['blocks'] - PI_spots.iloc[index]['spots'],2), 'Empty'))
    return fill

def plateInit():
    Plate = np.empty([10,4], dtype=object)
    for r1 in range(10):
        for c1 in range(4):
            Plate[r1,c1] = np.zeros([6, 3], dtype='|S32') 
    return Plate
            
def main(input_file):
   master = pd.read_excel(input_file, sep = "\t", header = 1)
	
   Plate = plateInit()   
   data = master.drop(master.columns[[0,1]], axis=1)
   data[data.columns[0]] = data[data.columns[0]].str.split("_", n = 1, expand = True)[0]
   all = data[data.columns[0:2]].groupby(data.columns[0])

   temp1 = all.get_group('Ctrl')
   T1A = temp1[temp1.columns[1]].str.split("_", n = 2, expand = True)[1]
   T1B = temp1[temp1.columns[1]].str.split("_", n = 1, expand = True)[1]
   T1 = pd.concat([T1A, T1B], axis=1)
   T1.columns = ['SampleName','Name in print map']

   CTRLS = T1.groupby(T1.columns[0])
   CTRLS.count().index.tolist()
   CTRLS.get_group('Cellmix2')

   temp2 = all.get_group('Cal')
   T2A = temp2.iloc[:,1].str.split("_", n = 2, expand = True)[0] + '_' + temp2.iloc[:,1].str.split("_", n = 2, expand = True)[1]
   T2 = pd.concat([T2A, temp2.iloc[:,1]], axis=1)
   T2.columns = ['SampleName','Name in print map']

   CALS = T2.groupby(T2.columns[0])
   CALS.count().index.tolist()
   CALS.get_group('Cal_1')

   Plate[0,:] = [np.full((6,3), 'Empty', dtype='|S32'), np.full((6,3), 'Empty', dtype='|S32'),np.full((6,3), 'Empty', dtype='|S32'), np.full((6,3), 'Empty', dtype='|S32')]
   Plate[9,:] = [np.full((6,3), 'Empty', dtype='|S32'), np.full((6,3), 'Empty', dtype='|S32'), np.full((6,3), 'Empty', dtype='|S32'), np.full((6,3), 'Empty', dtype='|S32')]  

   IgGs = [x for x in pd.Series(CTRLS.count().index.tolist()) if 'IgGmix' in x]
   Plate[0,0][0,0] = 'Ctrl_' + IgGs[0] 
   Plate[0,3][0,2] = 'Ctrl_' + IgGs[1]  
   Plate[9,0][5,0] = 'Ctrl_' + IgGs[2] 
   Plate[9,3][5,2] = 'Ctrl_' + IgGs[3]


   std = CTRLS.get_group('Std')
   std.iloc[:,1] = 'Ctrl_' + CTRLS.get_group('Std').iloc[:,1].astype(str)

   for c in range(2):
      C1 = df_append(CTRLS.get_group('BCCmix{}'.format(c+1)), CTRLS.get_group('Buffer{}'.format(c+1)))
      C2 = df_append(CTRLS.get_group('NCImix{}'.format(c+1)), CTRLS.get_group('Cellmix{}'.format(c+1)))
      C3 = df_append(C1,C2)
      C4 = df_append(C3, CTRLS.get_group('mTissueMix{}'.format(c+1)))
      C4.iloc[:,1] = 'Ctrl_' + C4.iloc[:,1].astype(str)
      Ctrl = df_append(CALS.get_group('Cal_{}'.format(c+1)), C4)
		
      if c==0:
          assign2([0,0,0,1], Ctrl, Plate)
          assign2([9,2,0,0], std, Plate)
      else:
          assign2([8,1,0,1], Ctrl, Plate)

   all_spots = pd.DataFrame(all.size(), columns=['spots']).sort_values(by=['spots'], ascending=False)
   PI_spots = all_spots.drop(['Ctrl','Cal']).reset_index()
   PI_spots = PI_spots.rename(columns={PI_spots.columns[0]:'index'})

   PI_spots['blocks'] = (PI_spots['spots']/12).apply(np.ceil).astype(int)
   PI_spots['blocks'].sum() 

   if len(PI_spots[PI_spots['blocks']==1])>=2: 
       p1 = str(PI_spots[PI_spots['blocks']==1].iloc[0]['index'])
       p2 = str(PI_spots[PI_spots['blocks']==1].iloc[1]['index']) 
       n1 = PI_spots[PI_spots['index']==p1].index[0]
       n2 = PI_spots[PI_spots['index']==p2].index[0]

   A = pd.DataFrame()
   for I in range(len(PI_spots)):
       A = df_append(all.get_group(str(PI_spots.iloc[I]['index'])), A)

   assign2([1,0,0,0], A, Plate)

   DF = Combined_cols(Plate, 0)
   for R in range(10)[1:]:
       DF = pd.concat([DF, Combined_cols(Plate, R)], axis=0, ignore_index=True)

   DF.columns = pd.RangeIndex(start = 1, stop = len(DF.columns)+1, step = 1)
   DF.index = pd.RangeIndex(start = 1, stop = len(DF.index)+1, step = 1)

   return DF

if __name__ == '__main__':
    main(input_file)	
    