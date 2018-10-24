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
    RMSE = round(RMSE, 3)
    print('Test Set RMSE = ' + str(RMSE) + ' degrees.')

    return RMSE

def calculate_score(RMSE):
    score = 0
    if RMSE <= 12.5:
        score = 10
    elif RMSE <= 15:
        score = 9
    elif RMSE <= 20:
        score = 8
    elif RMSE <= 25:
        score = 7
    elif RMSE <= 30:
        score = 6
    elif RMSE <= 40:
        score = 5
    else:
        score = 4
    return score


if __name__ == '__main__':
    program_start = time.time()
    RMSE = evaluate(student_file='sample_student', 
                    path_to_training_images = 'data/training/images',
                    training_csv_file = 'data/training/steering_angles.csv', 
                    path_to_testing_images = 'data/training/images',
                    testing_csv_file = 'data/training/steering_angles.csv', 
                    time_limit = 600)
    
    score = calculate_score(RMSE)
    program_end = time.time()
    total_time = round(program_end - program_start,2)
    
    print("Execution time (seconds) = ", total_time)
    print("Score = ", score)
