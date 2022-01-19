function Perceptron()
    clear all, close all;
    DataPoints = [-10 -10;-8 -2; -6 -12; -4 -4; 10 10;8 2; 6 12; 4 4;]; 
    DataPoints = DataPoints';
    DataLabel = [-1 -1 -1 -1 1 1 1 1]; 
    class1Pts = DataPoints(:, DataLabel == -1); 
    class2Pts = DataPoints(:, DataLabel == 1); 
    plot (class1Pts(1, :), class2Pts (2, :), 'bo', class2Pts (1, :), class2Pts (2, :), 'rx'); 
    W = findSepLine(class1Pts, class2Pts);
end
function W = findSepLine (class1Pts, class2Pts)
    class1Pts = [1,1,1,1;class1Pts];
    class2Pts = [1,1,1,1;class2Pts];
    W = [0 1 -1];   %x = y line
    xMin = min(class1Pts(2,:)); %x range for plotting the line
    xMax = max(class2Pts(2,:));
    xrange = xMin:0.01:xMax; 
    alpha = 0.005;
    fprintf('Alpha : %f\n',alpha)
    for i = 1:30
        % ax + by + c = 0
        a = W(2);
        b = W(3);
        c = W(1); 
        y = (-a*xrange -c)/b; %Generate y points on the boundary
        clf; 
        plot (class1Pts(2,:), class1Pts(3,:), 'bo', class2Pts(2,:), class2Pts(3,:), 'rx'); 
        hold on;
        plot (xrange, y,'r'); pause(5); %Plot the boundary
        %Pick misclassified samples from negative class
        predClss1 = class1Pts'*W'; %Predict class for negative class samples
        pickdmcSac1 = find(predClss1>0); %pick those which are wrongly predicted as positive
        sumdxX1 = sum(pickdmcSac1, 2); %compute gradient contribution from negative sample
        %Pick misclassified samples from positive class
        predClss2 = class2Pts'*W'; %Predict class for positive class samples
        pickdmcSac2 = find(predClss2<=0); %pick those which are predicted as from negative class
        sumdxX2 = sum(-1*pickdmcSac2, 2); %compute gradient contribution from negative samples
        errCst = sumdxX1 + sumdxX2 %compute gradient for both negative and positive class
        W = W - alpha*errCst %apply gradient descent
        if (isempty(pickdmcSac1) && isempty(pickdmcSac2))
            %if no misclassified samples
            return;
        end
    end
end