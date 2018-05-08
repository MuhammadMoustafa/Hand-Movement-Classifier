'''
Created on Dec 15, 2016

@author: Muhammad Moustafa

naive bayes classifier
'''
import scipy
import scipy.io as sio
from scipy.signal import iirfilter
import numpy as np
from scipy.stats import norm

# get the data from the matlab files 
# the function takes the name of the 2 files as paramters
# and return a np array contains the data of each file 

def read_data(file_name_1 , file_name_2) :
    return np.array(sio.loadmat(file_name_1 ,  appendmat=True)[file_name_1]) , np.array(sio.loadmat(file_name_2 ,  appendmat=True)[file_name_2])

# filterate the input data from the 50 hz noise    
# the function takes the np array and the sampling frequency as paramters
# and return a filtered np array 
def imp_filter(file,fs,freqLow=47,freqHigh=53,order = 3):

    nyq = fs/2.0
    freqLow /= nyq
    freqHigh /=nyq
    b, a = scipy.signal.iirfilter(order, [freqLow, freqHigh], btype='bandstop',analog=False)
    filteredRecord= scipy.signal.lfilter(b, a, file)
    return filteredRecord

# calculate the Energy feature for the file 
# the function takes a np array as a paramter
# and return an 1D np array contains the Energy of each record 
def Energy (file) :
    energy = (file**2).sum(axis=1) # sum the row after squaring each reading
    return energy

# calculate the Power feature for the file 
# the function takes a np array as a paramter
# and return an 1D np array contains the Power of each record 

def Power (file):
    power = (file**4).sum(axis=1) # sum the row after calculating the 4th power for each reading
    return power

# calculate the Non Linear Energy feature for the file 
# the function takes a np array as a paramter
# and return an 1D np array contains the Non Linear Energy of each record 

def NonLinearEnergy (file):
    nonlinearenergy = np.zeros((file.shape[0])) # intialize an 1D zero array
    for i in range (file.shape[0]): # for each row
        for j in range (file.shape[1]-2):# for each column except for the last 2 clolumns
            nonlinearenergy[i] += -file[i,j+2]*file[i,j] + file[i,j+1]**2 # calculate the summation 
    return nonlinearenergy

# calculate the Curve Length feature for the file 
# the function takes a np array as a paramter
# and return an 1D np array contains the Curve Length of each record

def CurveLength (file):
    curvelength = np.zeros((file.shape[0])) # intialize an 1D zero array
    for i in range (file.shape[0]): # for each row
        for j in range (0,file.shape[1]-2):# for each column except for the last clolumn 
            curvelength[i] += file[i,j+1] - file[i,j] # calculate the summation 
            
    return curvelength    

# calculate the normal distribution of each feature in each file 
# the function takes the number of rows to tarin the model with as a paramter
# and return a list contains the noraml distribution of each file  
def learn_phase(learn_size) :
    
    # intialize a 2D zero array to store the features of the learning set for each file
    file1_learn_Features = np.zeros((learn_size,4))    
    file2_learn_Features = np.zeros((learn_size,4))

    # filterate the data
    file1_learn_set = imp_filter(input1 , int(input1_fs))[0:learn_size] 
    file2_learn_set = imp_filter(input2 , int(input2_fs))[0:learn_size]

    # calculate the features for each file

    file1_learn_Features[:,0] = Energy(file1_learn_set)        
    file2_learn_Features[:,0] = Energy(file2_learn_set)
    
    file1_learn_Features[:,1] = Power(file1_learn_set)
    file2_learn_Features[:,1] = Power(file2_learn_set)
    
    file1_learn_Features[:,2] = NonLinearEnergy(file1_learn_set)
    file2_learn_Features[:,2] = NonLinearEnergy(file2_learn_set)
    
    file1_learn_Features[:,3] = CurveLength(file1_learn_set)
    file2_learn_Features[:,3] = CurveLength(file2_learn_set)
        
    file1_learn_features_mean = np.mean(file1_learn_Features,axis=0)
    file2_learn_features_mean = np.mean(file2_learn_Features,axis=0)
    file1_learn_features_std = np.std(file1_learn_Features,axis=0)
    file2_learn_features_std = np.std(file2_learn_Features,axis=0)

    # intialize an empty list to store the normal distributions
    file1_norm_dist = [] 
    file2_norm_dist = []

    # calulate the mean and the standard deviation of each feature and append the noraml distribution to the list
    for i in range (4):
        file1_norm_dist.append(norm(file1_learn_features_mean[i],file1_learn_features_std[i]))
        file2_norm_dist.append(norm(file2_learn_features_mean[i],file2_learn_features_std[i]))            
    
    return file1_norm_dist , file2_norm_dist

# calculate the normal distribution of each feature in each file 
# the function takes the number of rows to tarin the model with as a paramter .. so it can get the #rows to test
# and return a np array contains the features of each file 
def test_phase(learn_size):
    
    # calculate the number of test rows
    test_size = input1.shape[0] - learn_size
    
    # intialize a 2D zeor array to store the features of the test set
    file1_test_Features = np.zeros((test_size,4))    
    file2_test_Features = np.zeros((test_size,4))

    # filterate the data
    file1_test_set = imp_filter(input1 , int(input1_fs))[0+learn_size:test_size+learn_size] 
    file2_test_set = imp_filter(input2 , int(input2_fs))[0+learn_size:test_size+learn_size]

    # calculate the features for file#1 test set
    # calculate the features for file#2 test set  
    
    file1_test_Features[:,0] = Energy(file1_test_set)
    file1_test_Features[:,1] = Power(file1_test_set)
    file1_test_Features[:,2] = NonLinearEnergy(file1_test_set)
    file1_test_Features[:,3] = CurveLength(file1_test_set)    
            
    file2_test_Features[:,0] = Energy(file2_test_set)
    file2_test_Features[:,1] = Power(file2_test_set)
    file2_test_Features[:,2] = NonLinearEnergy(file2_test_set)
    file2_test_Features[:,3] = CurveLength(file2_test_set)          
            
    return file1_test_Features , file2_test_Features        

def calculate_performance():
    
    # get the noraml distribution of each feature in each file
    input1_norm , input2_norm = learn_phase(input_learn_size)
    # get the features of the test data set in each file
    input1_test , input2_test = test_phase(input_learn_size)
    
    # intailaize variables to calculate the accuracy of the model
    file1_acc = 0.0
    file2_acc = 0.0
    '''
    boolLateral = []
    boolPalmar = []
   '''
    
    # test the data in file#1
    #if the data abeied the normal distribution of file 1 then increase accuracy by 1  
    for i in range (input_test_size): #for each record
        
        # calculate the likelyhood of the record in file 1 to be in file 1
        ainput1= input1_norm[0].pdf(input1_test[i,0])*input1_norm[1].pdf(input1_test[i,1])*input1_norm[2].pdf(input1_test[i,2])*input1_norm[3].pdf(input1_test[i,3])
        
        # calculate the likelyhood of the record in file 1 to be in file 2
        ainput2=input2_norm[0].pdf(input1_test[i,0])*input2_norm[1].pdf(input1_test[i,1])*input2_norm[2].pdf(input1_test[i,2])*input2_norm[3].pdf(input1_test[i,3])
        
        # calculate the likelyhood of the record in file 2 to be in file 1
        binput1= input1_norm[0].pdf(input2_test[i,0])*input1_norm[1].pdf(input2_test[i,1])*input1_norm[2].pdf(input2_test[i,2])*input1_norm[3].pdf(input2_test[i,3])
        
        # calculate the likelyhood of the record in file 2 to be in file 2
        binput2=input2_norm[0].pdf(input2_test[i,0])*input2_norm[1].pdf(input2_test[i,1])*input2_norm[2].pdf(input2_test[i,2])*input2_norm[3].pdf(input2_test[i,3])
       
        if (ainput1 > ainput2): # if the propabilty of that record in file 1 to be in file 1 is more
            '''
            boolLateral.append(True)
            '''
            file1_acc +=1 # increase the accuracy by one
        
        '''
        else:
            boolLateral.append(False)
        '''
        
        if (binput1 < binput2): # if the propabilty of that record in file 2 to be in file 2 is more
            '''
            #boolPalmar.append(True)
            '''
            file2_acc +=1
        '''
        else:
            boolPalmar.append(False)
        ''' 
    
    print ("\nAccuracy is " , (file1_acc + file2_acc)*100.0/(2*input_test_size) , "%") # print the total Accuracy
    '''
    print (file1_acc,boolLateral"\n" , file2_acc , boolPalmar)
    '''

'''    
user interface    
'''
# intailaize a boolean variable to run the program multiple times 
exiit = True
print("This program is made to differentiate between two types on movement 'palmar' and 'lateral' based on EMG records based on naive bayes classifier")
print("please make sure that the 2 matlab files have the same shape")
print("please make sure that the 2 matlab file are in the same directory that the program on or write the full path instead of the file name")

while (exiit):
    try:
        select = input("\nwrite 1 for Biostatistics Project\nwrite anything else to write your files' name\nwrite exit to close the program\n\n")
        if select == '1' :
            while(True):
                try:
                    input1 , input2 = read_data('lateral' , 'palmar')
                    input1_fs = 1000
                    input2_fs = 1000
                    input_learn_size = 100
                    input_test_size = 50
                    select2 = input("Write 1 to shuffle the data\nany thing else to continue without shuffle\n") #use shuffle or not
                    if select2 == '1':
                        np.random.shuffle(input1)
                        np.random.shuffle(input2)
                    calculate_performance()
                except FileNotFoundError:
                    print("file not found\nplease make sure that the 2 matlab file are in the same directory that the program on or write the full path instead of the file name\n")                      
                finally:
                    break

    
        elif select == 'exit':
            exiit = False     
    
        else : 

            # get the data file name , sampling frequency from the user and pass them to their functions to store and filterate
            while(True):
                try:
                    input1 , input2 = read_data(input('insert the name of file#1 with out the extention .mat\n '), input('insert the name of file#2 with out the extention .mat\n '))    
                    break
                except FileNotFoundError:
                    print("file not found\nplease make sure that the 2 matlab file are in the same directory that the program on or write the full path instead of the file name\n")
                    
                
            while(True):
                input1_fs = input("write the sampling frequency of the first record\n")
                input2_fs = input("write the sampling frequency of the second record\n")
                if(input1_fs.isdigit() and input2_fs.isdigit() and int(input1_fs) > 105 and int(input2_fs) >105 ): 
                    break
                else :
                    print("\ncheck the written sampling frequency\n")
                
            # shuffle the files
            np.random.shuffle(input1)
            np.random.shuffle(input2)
        
            # get the learn size test
            while(True):
                temp = input("write the percentage you want to train the model with 'any number between 0 and 100' \n")
                try :  
                    if ( 0< float(temp) <100 ):
                        input_learn_size = int((np.ceil( input1.shape[0] * float(temp))) / 100.0)
                        break   
                except ValueError:
                    print("\ncheck the written percentage\n")
                    
            input_test_size = input1.shape[0] - input_learn_size
            calculate_performance()
    except:
        raise
print ("Thank You!!")       
