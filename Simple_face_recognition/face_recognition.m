% We take images from the data base 
database = imageSet('orl_faces' , 'recursive');
% Display montage 
    %figure;
    %montage(database(2).ImageLocation);
%Split database into traning set and test set
[training,test] = partition(database,[0.8 0.2]);
%Extract and display HOG features for single image
person = 5;
[hogFeature, visualization] = ...
    extractHOGFeatures(read(training(person),1));
figure;
subplot(2,1,1);imshow(read(training(person),1));title('input face');
subplot(2,1,2);plot(visualization);
% Etract HOG features from training set
training_features = zeros(size(training,2)*training(1).Count,4680);
feature_count = 1;
for i = 1 : size(training,2)
    for j = 1 : training(i).Count
        training_features(feature_count , :) = extractHOGFeatures(read(training(i),1));
        training_label{feature_count} = training(i).Description;
        feature_count = feature_count+1;
        
    end
    person_index{i} = training(i).Description;
end

%Creating class classifier using fitcecoc
face_classifier = fitcecoc(training_features,training_label);

%test images from test set
person = 2;
query_image = read(test(person),1);
query_features = extractHOGFeatures(query_image);
person_label = predict(face_classifier,query_features);
%map to training set to find identity
boolean_index = strcmp(person_label,person_index);
integer_index = find(boolean_index);
subplot(1,2,1);imshow(query_image);title('Query image');
subplot(1,2,2);imshow(read(training(integer_index),1));title('matched class');
