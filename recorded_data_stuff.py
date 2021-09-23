import pandas as pd
import numpy as np

#read_file = pd.read_csv (r'C:\Users\liorb\OneDrive - Technion\Documents\Project B - 044169\Recordings\accurate_position.txt')
#read_file.to_csv (r'C:\Users\liorb\OneDrive - Technion\Documents\Project B - 044169\Recordings\accurate_position.csv', index=None)

location = 'C:/Users/liorb/Documents/ProjectB/Recordings/second/'
df = pd.read_fwf(location + 'position_modified.txt')
#df =df.iloc[25:]
df.to_csv(location + 'position_modified.csv')

time_df = df['ms_gps'].values
diff = time_df[1:] - time_df[:-1]
indexes = np.where(diff != 25)
indexes = indexes[0]
for i in range(len(indexes)):
    if i == 0:
        partial = df.iloc[: indexes[i] + 1]
    else:
        partial = df.iloc[indexes[i-1] + 1: indexes[i] + 1]
    partial.to_csv(location + 'position_step' + str(i+1) + '.csv')
partial = df.iloc[indexes[8] + 1:]
partial.to_csv(location + 'position_step' + str(len(indexes) + 1) + '.csv')
