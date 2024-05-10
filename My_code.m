    %Candidate Number - 276236
    
    clc
    clear all
    
    % Prompt user to select an image file
    [filename, filepath] = uigetfile({'*.png;*.jpg;*.bmp;*.tif', 'Image Files (*.png, *.jpg, *.bmp, *.tif)'}, 'Select an image file');
    if isequal(filename, 0)
        disp('No file selected');
    else
        processImage(fullfile(filepath, filename));
    end
    
    function processImage(filename)
        % Load and preprocess the image
        inputImage = loadImage(filename);
        
        subplot(2, 2, 1);
        imshow(inputImage);
        title('Input image');
    
        % Detect circle locations
        circleLocations = detectCircles(inputImage);
        
        % Check if at least 4 circles are detected
        if size(circleLocations, 1) < 4
            disp('Less than 4 circles detected. Cannot perform geometric transformation.');
            return;
        end
        
        % Correct image distortion based on detected circles
        [correctedImage, ~] = correctImageDistortion(circleLocations, inputImage);
        
        % Display the corrected image
        subplot(2, 2, 4);
        imshow(correctedImage);
        title('Corrected Image');
    
        % Identify colors within specific regions
        colorGrid = identifyColorRegions(correctedImage);
        disp("Color Grid of the Image:");
        disp(colorGrid); % Display color grid
    end
    
    %% identifyColorRegions Function
    function colorGrid = identifyColorRegions(inputImage)
        % Converting RGB colorspace to LAB
        labImage = rgb2lab(inputImage); 
        
        colorPoints = [80 172 259 369];
        
      
        colorPointValues = zeros(16, 3);
        count = 0;
        
        for i = 1:4
            for j = 1:4
                count = count + 1;
                x = colorPoints(1, i);
                y = colorPoints(1, j);
                temp = labImage(x:x + 56, y:y + 56, :);
                colorPointValues(count, :) = mean(reshape(temp, [], 3), 1);
            end
        end
        
        % Defining the colors in RGB and LAB
        % Red, Green, Blue, Yellow, White, Purple
        rgbScale = [1 0 0; 0 1 0; 0 0 1; 1 1 0; 1 1 1];
        % r=red, g=green, b=blue, y=yellow, w=white
        colorNames = {'r', 'g', 'b', 'y', 'w'}; 
        labScale = rgb2lab(rgbScale);
        distances = pdist2(colorPointValues, labScale, 'euclidean');
        
       
        [~, colorIndices] = min(distances, [], 2);
        colorPatches = colorNames(colorIndices);
        
      
        colorGrid = reshape(colorPatches, 4, 4)';
    end
    
    % Load an image from the specified file and return it as a double precision matrix
    function image = loadImage(filename)
        img = imread(filename);
        image = im2double(img);
    end
    
    % Detect the coordinates of the black circles in the image
    function circleCoordinates = detectCircles(image)
        grayImage = rgb2gray(image);
        
        % Threshold the image to obtain a binary image
        threshold = graythresh(grayImage);
        binaryImage = imbinarize(grayImage, threshold);
        
        % Invert the binary image
        invertedBinaryImage = imcomplement(binaryImage);
        
        % Label connected components in the inverted binary image
        cc = bwconncomp(invertedBinaryImage);
        
        % Calculate the area of each connected component
        areas = cellfun(@numel, cc.PixelIdxList);
        
        % Sort areas in descending order
        [~, sortedIndices] = sort(areas, 'descend');
        
        % Get the coordinates of the first four largest black blobs
        numBlobs = 5;
        blobCoordinates = zeros(numBlobs, 2);
        for i = 2:numBlobs
            blobIndices = cc.PixelIdxList{sortedIndices(i)};
            [rows, cols] = ind2sub(size(invertedBinaryImage), blobIndices);
            blobCoordinates(i, :) = [mean(cols), mean(rows)];
        end
        
        % Remove the first coordinate from the blobCoordinates matrix
        blobCoordinates(1, :) = [];
        
        % Sort the coordinates in clockwise order starting from bottom-left
        sortedCoordinates = sortClockwise(blobCoordinates);
    
        subplot(2, 2, 2);
        imshow(grayImage);
        viscircles(sortedCoordinates, 20, 'EdgeColor', 'b'); % Display detected circles
        title('Circles in image');
        
        circleCoordinates = sortedCoordinates;
    end
    
    % Sort coordinates in clockwise order starting from bottom-left
    function sortedCoordinates = sortClockwise(coordinates)
        sortedCoordinates = sortrows(coordinates);
        if sortedCoordinates(2, 2) < sortedCoordinates(1, 2)
            sortedCoordinates([1 2], :) = sortedCoordinates([2 1], :);
        end
        if sortedCoordinates(4, 2) > sortedCoordinates(3, 2)
            sortedCoordinates([3 4], :) = sortedCoordinates([4 3], :);
        end
    end
    
    % Correct image distortion using geometric transformation
    function [outputImage, correctedCoordinates] = correctImageDistortion(coordinates, image)
        targetBox = [[0, 0]; [0, 480]; [480, 480]; [480, 0]];
        
        % Calculate the transformation matrix using projective transformation
        transformationMatrix = fitgeotrans(coordinates, targetBox, 'projective');
        
        % Create an image reference object with the size of the input image
        outputView = imref2d(size(image));
        
        % Apply the transformation matrix to the input image
        transformedImage = imwarp(image, transformationMatrix, 'fillvalues', 255, 'OutputView', outputView);
        
        % Crop the image to a size of 480x480
        croppedImage = imcrop(transformedImage, [0, 0, 480, 480]);
        
        % Suppress glare in the image using flat-field correction
        flatFieldCorrectedImage = imflatfield(croppedImage, 40);
        
        % Adjust the levels of the image to improve contrast
        contrastedImage = imadjust(flatFieldCorrectedImage, [0.4, 0.65]);
    
        % Denoise the image to improve image quality
        redChannel = medfilt2(contrastedImage(:,:,1), [5 5]);
        greenChannel = medfilt2(contrastedImage(:,:,2), [5 5]);
        blueChannel = medfilt2(contrastedImage(:,:,3), [5 5]);
        outputImage = cat(3, redChannel, greenChannel, blueChannel);
        
        correctedCoordinates = targetBox;
    end