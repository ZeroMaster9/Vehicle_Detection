
% Get the full path to the MATLAB script
matlab_script_path = fullfile(pwd, 'AHMED_217380_Vehicle_Detection.m');

% Get the path to the directory containing the script
script_directory = fileparts(matlab_script_path);

% Specify the names of the configuration and weights files
config_file_name = 'yolov3.cfg';
weights_file_name = 'yolov3.weights';

% Create the full file paths for the configuration and weights files
config_file_path = fullfile(script_directory, config_file_name);
weights_file_path = fullfile(script_directory, weights_file_name);

% Load the YOLOv3 object detection network
net = deepLearningDetector(config_file_path, weights_file_path);

% Rest of your code for vehicle detection and segmentation goes here...


% Load pre-trained YOLOv3 network
net = deepLearningDetector('yolov3.cfg', 'yolov3.weights');

% Read video file
video = VideoReader('Case1.wmv');

% Define ROI (region of interest)
roi = [500 50 800 450];

% Loop through video frames
while hasFrame(video)
    % Read frame
    frame = readFrame(video);
    
    % Crop ROI
    frame_roi = imcrop(frame, roi);
    
    % Perform object detection using YOLOv3
    [bboxes, scores, labels] = detect(net, frame_roi);
    
    % Select only vehicle objects
    vehicle_idx = strcmp(labels, 'car') | strcmp(labels, 'bus') | strcmp(labels, 'truck');
    vehicle_bboxes = bboxes(vehicle_idx, :);
    
    % Segment vehicle objects
    for i = 1:size(vehicle_bboxes, 1)
        bbox = vehicle_bboxes(i, :);
        vehicle_roi = imcrop(frame_roi, bbox);
        vehicle_gray = rgb2gray(vehicle_roi);
        vehicle_bw = imbinarize(vehicle_gray);
        vehicle_segmented = vehicle_roi .* uint8(vehicle_bw);
        frame_roi(bbox(2):bbox(2)+bbox(4), bbox(1):bbox(1)+bbox(3), :) = vehicle_segmented;
    end
    
    % Show video frame with vehicle segmentation
    imshow(frame_roi);
end
