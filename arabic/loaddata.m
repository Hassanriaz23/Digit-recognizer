X = load('csvTrainImages 60k x 784.csv');
y = load('csvTrainLabel 60k x 1.csv');
y = zeroto10(y);
X = double(X)/127.5 - 1;

Xtest = load('csvTestImages 10k x 784.csv');
ytest = load('csvTestLabel 10k x 1.csv');
ytest = zeroto10(ytest);
Xtest = double(Xtest)/127.5 - 1;

save('data.mat','X','y','Xtest','ytest');