addpath RectGrabber/

VT = 1/15;

movie = VideoWriter( sprintf( '%s\\result.avi', 'figures' ) );
open(movie)

A = [1 0 VT 0; 0 1 0 VT; 0 0 1 0; 0 0 0 1];
B = [];
C = [1 0 0 0; 0 1 0 0];
D = [0];
% Estimated speeds (px/s)
EVX = 40;
EVY = 5;
% Covariance of the process noise
Q = [(EVX*VT)^2 0 VT 0;
    0 (EVY*VT)^2 0 VT;
    VT 0 (EVX^2) 0;
    0 VT 0 (EVY^2)];
% Estimated HOG detector error (px)
EEX = 25;
EEY = 25;
% Covariance of the observation noise
R = [EEX^2 0;
    0 EEY^2];

peopleDetector = vision.PeopleDetector;
%% Waiting for a person
people = 0;
i = 70;
while people == 0
    image = imread(sprintf( '%s\\imgrect_%09d_c0.pgm', 'RectGrabber', i));
    [bboxes,scores] = step(peopleDetector,image);
    people = size(bboxes,1);
    i = i + 1;
    writeVideo(movie,image)
end

%% Kalman filter
% Initial mu and sigma
mu_t_t = [bboxes(1)+bboxes(3)/2; bboxes(2)+bboxes(4)/2; 0; 0];
%sigma_tm1 = [40^2 0 1/15 0; 0 40^2 0 1/15; 1/15 0 5^2 0; 0 1/15 0 5^2];
sigma_t_t = [0 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 0];


pre_draw = [];
mea_draw = [];
upd_draw = [];

var_X = 0;
var_Y = 0;

% Main bulce
for i = i:299
    %% Prediction
    mu_t_tm1 = A*mu_t_t;
    sigma_t_tm1 = A*sigma_t_t*A'+Q;
    
    %% Matching
    % Read the next frame
    image = imread(sprintf( '%s\\imgrect_%09d_c0.pgm', 'RectGrabber', i));
    [bboxes,scores] = step(peopleDetector,image);    
    if(size(scores,1) > 0)
        % There are people in the measure
        
        % Measure
        y_t = [bboxes(1) + bboxes(3)/2; bboxes(2) + bboxes(4)/2];
        lastMeasure = bboxes;
        
        % Residue
        r_t = y_t - (C*mu_t_tm1);
        
        % Residual covariance
        S_t = C*sigma_t_tm1*C' + R;
        
        % Kalman gain
        K_t = (sigma_t_tm1*C')/S_t;
        
        if (abs(r_t(1,1)) < sqrt(S_t(1,1)) && abs(r_t(2,1)) < sqrt(S_t(2,2)))
            % Valid measure --> Update
            mu_t_t = mu_t_tm1 + K_t*r_t;
            sigma_t_t = (eye(size(K_t,1)) - K_t*C)*sigma_t_tm1;
        else
            % No valid measure --> No update
            mu_t_t = mu_t_tm1;
            sigma_t_t = sigma_t_tm1;
        end        
    else
        % There are not people in the measure
        mu_t_t = mu_t_tm1;
        sigma_t_t = sigma_t_tm1;
    end
    
    pre_ellipse = [sigma_t_tm1(1,1) sigma_t_tm1(1,2); sigma_t_tm1(2,1) sigma_t_tm1(2,2)];
    mu_pre_ellipse = [mu_t_tm1(1,1) ; mu_t_tm1(2,1)];

    upd_ellipse = [sigma_t_t(1,1) sigma_t_t(1,2); sigma_t_t(2,1) sigma_t_t(2,2)];
    mu_upd_ellipse = [mu_t_t(1,1) ; mu_t_t(2,1)];

    width = 250;
    height = 300;

    % Building prediction bounding box
    boxPredict(1) = mu_t_tm1(1) - lastMeasure(3)/2;
    boxPredict(2) = mu_t_tm1(2) - lastMeasure(4)/2;
    boxPredict(3) = lastMeasure(3);
    boxPredict(4) = lastMeasure(4);

    % Building update bounding box
    boxUpdate(1) = mu_t_t(1) - lastMeasure(3)/2;
    boxUpdate(2) = mu_t_t(2) - lastMeasure(4)/2;
    boxUpdate(3) = lastMeasure(3);
    boxUpdate(4) = lastMeasure(4);


    % Drawing bounding boxes
    image = insertObjectAnnotation(image, 'rectangle', boxPredict , 'Prediction', 'Color', 'red');
    image = insertObjectAnnotation(image, 'rectangle', bboxes , 'Measure', 'Color', 'green');
    image = insertObjectAnnotation(image, 'rectangle', boxUpdate , 'Update', 'Color', 'blue');


    figure(1) , imshow(image);

    % Drawing ellipse
    plotUncertainEllip2D(pre_ellipse, mu_pre_ellipse, 'magenta');
    
    writeVideo(movie,image);
    
    pause(0.005);
    
end

close(movie);
