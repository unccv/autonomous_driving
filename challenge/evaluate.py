import numpy as np
import cv2
from tqdm import tqdm
import time
import signal

def timeout_handler(num, stack):
    raise Exception("TIMEOUT")

def evaluate(student_file = 'sample_student', 
             path_to_training_images = 'data/training/images',
             training_csv_file = 'data/training/steering_angles.csv', 
             path_to_testing_images = 'data/training/images',
             testing_csv_file = 'data/training/steering_angles.csv', 
             time_limit = 600):
    
    '''
    Evaluate Student Submission for autonomous driving challenge. 
    Train and test studen't neural network implementation. 
    Training time is limited to time_limit seconds, if your code takes 
    longer than this, it will be terminated and no score will be recorded.
    '''
    
    #Import student methods:
    train = getattr(__import__(student_file, 'train'), 'train')
    predict = getattr(__import__(student_file, 'predict'), 'predict')
    
    #Setup timout handler - I think this will only work on unix based systems:
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(time_limit)
    
    try:
        print("Training your network, start time: %s" % time.strftime("%H:%M:%S"))
        NN = train(path_to_images = path_to_training_images,
                   csv_file = training_csv_file)

    except ValueError as ex:
        pass
        
    finally:
        signal.alarm(0)
        print("Ending Time: %s" % time.strftime("%H:%M:%S"))
        
    print('Training Complete! \n')
    
    print('Measuring performance...')
    ## Measure Performance:
    data = np.genfromtxt(testing_csv_file, delimiter = ',')
    frame_nums = data[:,0]
    steering_angles = data[:,1]

    predicted_angles = []
    for frame_num in tqdm(frame_nums):
        im_path = path_to_testing_images + '/' + str(int(frame_num)).zfill(4) + '.jpg'
        predicted_angles.append(predict(NN, im_path))
        
    RMSE = np.sqrt(np.mean((np.array(predicted_angles)- steering_angles)**2))
    print('Test Set RMSE = ' + str(round(RMSE, 3)) + ' degrees.')

    return RMSE
