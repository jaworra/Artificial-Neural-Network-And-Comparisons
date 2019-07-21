% Optimization of Artificial Neural Network Models and Evaluation against
%Random Forest and Multiple Linear Regression
% MAT8190 Assignment 2 - MATHEMATICS STATISTICS COMPLEMENTARY STUDIES
% Author: John Worrall
% Description: RF modeld  and MLR for prediction 
% Requirments: Excel file (RandomForestDataA1-SD-VERSION.xlsx)
%----------------------------

clear 
clc
close all

% Number of models (Default 5 times)
nModels = 5;

% Choose Options (CO)
% 1 - Artifical Neural Network
% 2 - Multiple Linear Regression Model
% 3 - Random Forest
CO = 3;

%randomise parameters, ensure the results are reproducible.
rand('twister', 123);
s = RandStream('mlfg6331_64');

%IO
trnData=xlsread('RandomForestDataA2-SD-VERSION.xlsx','Training');
chkData=xlsread('RandomForestDataA2-SD-VERSION.xlsx','CheckData_Testing');

%Prepare Train/Test
trnIn = trnData(:,1:6);
trnOut = trnData(:,7);
chkIn = chkData(:,1:6);
chkOut = chkData(:,7);

% Use GPU if available 
ngpus=gpuDeviceCount;
disp([num2str(ngpus) ' GPUs found'])
if ngpus>0
    lgpu=1;
    disp('GPU found')
    useGPU='yes';
else
    lgpu=0;
    disp('No GPU found')
    useGPU='no';
end
% Find number of cores
ncores=feature('numCores');
disp([num2str(ncores) ' cores found'])
% Find number of cpus
import java.lang.*;
r=Runtime.getRuntime;
ncpus=r.availableProcessors;
disp([num2str(ncpus) ' cpus found'])
if ncpus>1
    useParallel='yes';
else
    useParallel='no';
end
[archstr,maxsize,endian]=computer;
disp(['This is a ' archstr ' computer that can have up to ' num2str(maxsize) ' elements in a matlab array and uses ' endian ' byte ordering.'])

% Set up the size of the parallel pool if necessary
npool=ncores;

% Opening parallel pool
if ncpus>1
    tic
    disp('Opening parallel pool')
    % first check if there is a current pool
    poolobj=gcp('nocreate');
% If there is no pool create one
    if isempty(poolobj)
        command=['parpool(' num2str(npool) ');'];
        disp(command);
        eval(command);
    else
        poolsize=poolobj.NumWorkers;
        disp(['A pool of ' poolsize ' workers already exists.']);
    end
        % Set parallel options
        paroptions = statset('UseParallel',true);
        %Set parrellel streams to have same seed value for repeated results
        %paroptions = statset('UseParallel',true,'Streams', s, 'UseSubStreams',true);
        toc
end

if CO == 1
% %run pacf - on objective variable (for additional data)
% obVar = [trnOut; chkOut];
% figure
% subplot(2,1,1)
% autocorr(obVar)
% subplot(2,1,2)
% parcorr(obVar)
% %determine t-4 lag

%add lag variable (t-4)
%remove first 4 rows
%AORDOut(4,:)=[];

%Normalise
% x = [min(trnData,[],1);max(trnData,[],1)]
% b = bsxfun(@minus,trnData,x(1,:));
% b = bsxfun(@rdivide,b,diff(x,1,1))

%=============================================================
%devided into training/validation/testing
% 10% cut from training set - 1123(training) + 125(validation)
% Split training set into 40:60 (Training:Validation) (499:749) 
pn=trnIn(1:499,:)';
qn=trnOut(1:499,:)';

valP=trnIn(500:1248,:)';
valT=trnOut(500:1248,:)';

% %Create ANN
% net = newff (minmax(pn),[3 1],{'tansig' 'purelin'},'trainlm');
% %net = newff (minmax(pn),[250 1],{'tansig' 'purelin'},'trainlm');
% 
% %optimize your model, for HiddenTransfer functions 
% ParA = {'purelin','logsig','purelin','logsig','purelin','logsig','purelin','logsig','purelin','logsig','purelin','logsig'};
% ParB = {'logsig','purelin','logsig','purelin','logsig','purelin','logsig','purelin','logsig','purelin','logsig','purelin'};
% ParC = {'trainlm','trainlm','trainbfg','trainbfg','traingdx','traingdx','trainscg','trainscg','traincgf','traincgf','traincgp','traincgp'};
% %2*6 = 12 combinations
% for hn  = 1:1:12
%     net = newff (minmax(pn),[3 1],{ParA{hn} ParB{hn}},ParC{hn});
%     net=init(net);
%     net.trainParam.show=100;
%     net.trainParam.epochs=500;
%     net.trainParam.goal=0.0001;
%     net.performFcn='mse';
%     
%     w1 = net.IW{1,1};
%     w2 = net.LW{2,1};
%     b1 = net.b{1};
%     b2 = net.b{2}; %check  
%     
%     [net, tr] = train(net, pn, qn); % fix
%     a = sim(net, pn); 
%     av=a'; 
%     z = [a' qn'];
%     pval=chkIn;
%     qval=chkOut;
%     y = sim(net, chkIn')
%     dataSimTest = y';
%     zv = [y' chkOut];
%     
%     %Asses
%     [r,ENS,d,Pdv,RMSE,MAE,PI] =asseMetric(chkOut,dataSimTest);
%     HiddenTransferMetrics(:,hn) = PI';
%     HiddenTransferResults(:,hn)=dataSimTest;
% end 
% %From above, run 2 is clearly the best with 0.89 r, 2.345 RMSE, 1.786 MAE
% %That is - the following.. parameter A 'logsig', parameter B 'purelin'
% %prameter C 'trainlm'


% optimize your model, for Neurons. 
% 
% %Train
% net = newff(minmax(pn),[30 1],{'logsig' 'purelin'},'trainlm');
%     net=init(net);
%     net.trainParam.show=100;
%     net.trainParam.epochs=500;
%     net.trainParam.goal=0.0001;
%     net.performFcn='mse';
%     w1 = net.IW{1,1};
%     w2 = net.LW{2,1};
%     b1 = net.b{1};
%     b2 = net.b{2};
%     [net, tr] = train(net, pn, qn); % fix
%     a = sim(net, pn);
%     
% 
for hn = 1:1:250
    net = newff(minmax(valP),[hn 1],{'logsig' 'purelin'},'trainlm');
    net=init(net);
    net.trainParam.show=100;
    net.trainParam.epochs=50;
    net.trainParam.goal=0.0001;
    net.performFcn='mse';
    w1 = net.IW{1,1};
    w2 = net.LW{2,1};
    b1 = net.b{1};
    b2 = net.b{2};

    
    %Build on validation set
    [net, tr] = train(net, valP, valT); % fix
    a = sim(net, pn); 
    av=a'; 
    z = [a' qn'];
  
   
    pval=chkIn;  % test
    qval=chkOut;
    y = sim(net, chkIn');
    dataSimTest = y';
    zv = [y' chkOut];

    %Asses
    [r,ENS,d,Pdv,RMSE,MAE,PI] =asseMetric(chkOut,dataSimTest);
    TestErrorsMetrics(:,hn) = PI';
    SimulatedResults(:,hn)=dataSimTest;

end 

%RMSE as a function of hn
rmseModel=xlsread('Mat8180Assignment2-Results.xlsx','ANN_HN_RMSE');
rmseModel = rmseModel(:,1);
    figure
    hold on;
    plot(rmseModel)
    title('Peformance of Neurons on ANN model');
    xlabel('Errors (RRMSE)');
    ylabel('Number of Neurons');

    hold off;

    
% ===========================================================  
%    ANN results
bestModel=xlsread('Mat8180Assignment2-Results.xlsx','ANN_NuronsRuns');
bestModel = bestModel(:,10);
    subplot(2,2,1)
    %figure
    hold on;
    scatter(chkOut,bestModel)
    %title('Scatter plot Tested vs Simulated (monthly global solar radiation)');
    title({'Scatter plot Tested vs Simulated','(monthly global solar radiation)'})
    xlabel('Data Observation (MJ/m^2)');
    ylabel('Data Simulated (MJ/m^2)');
    %add line
    nnR = corrcoef(chkOut,bestModel);
    nnR = nnR(1,2);
    coeffs = polyfit(chkOut, bestModel, 1);
    % Get fitted values
    fittedX = linspace(min(chkOut), max(chkOut), 198);
    fittedY = polyval(coeffs, fittedX);
    C = round(polyval(coeffs, fittedX(1,1),0));
    plot(fittedX, fittedY, 'r-', 'LineWidth', 1);
    txt1 = ['y = ' num2str(round(nnR,3)) 'x + ' num2str(C)];
    ylim =get(gca,'ylim')-7
    text(min(chkOut),max(ylim), txt1);
    text(min(chkOut),max(ylim)-2, ['r^2 = ' num2str(round(nnR*nnR,3))]);
    hold off;
    
  %histogram
    subplot(2,2,2)
    dataErr = abs(chkOut - bestModel);
    hmin = min(dataErr);
    hmax = max(dataErr);
    hold on;
    hist(dataErr,[hmin:0.25:hmax]);
    %title('Histogram absolute error frequency(monthly global solar radiation)');
    title({'Histogram absolute error frequency','(monthly global solar radiation)'})
    ylabel('Frequency of Error');
    xlabel('Errors (brackets ± 0.25)');
    hold off;    
    

% Timeseries
    %figure
    subplot(2,1,2)
    plot(chkOut,'k');
    hold on;
    plot(bestModel);
    %title('Timeseries plot Tested vs Simulated (monthly global solar radiation)');
    title({'Timeseries plot Tested vs Simulated','(monthly global solar radiation)'})
    xlabel('Months')
    ylabel('\Delta [MJ/m^2]')
    var = {{'Observation','ANN'},'Location','northeast'}
    legend(var{:})
    %legend({'Test','arima1','arima2','t1mamdami','t1Sugerno','allSugerno','nt1Sugerno','en50ANFIS','en100ANFIS','SOMANFIS'},'Location','northwest','NumColumns',2)
    hold off
    
%------------------------------------------------------------    
% Multiple Linear Regression Model
elseif CO == 2
    pn=trnIn(1:499,:)';
    qn=trnOut(1:499,:)';
    valP=trnIn(500:1248,:)';
    valT=trnOut(500:1248,:)';
     
    %MLR with validation set
    In = valP';
    Out = valT';   
    
%     In = trnIn;
%     Out = trnOut;
    c = regress(Out,In);
    cIn = c';
    
    y2= cIn.*chkIn; %predict(c,chkIn);
    dataSimMLR = sum(y2,2);
    dataObs = chkOut;
    
    %Assement error metrics --------------------------------
    [nnR,nnENS,nnD,nnPDEV,nnRMSE,nnMAE,nnPI]=asseMetric(dataObs,dataSimMLR);
    ErrorsMLR = nnPI;
    %asseMetricVis(dataObs,dataSimMLR,nnR,1,'Multiple Linear Regression');    
    
%VISUALS  
%    MLR results
    subplot(2,2,1)
    %figure
    hold on;
    scatter(chkOut,dataSimMLR)
    %title('Scatter plot Tested vs Simulated (monthly global solar radiation)');
    title({'Scatter plot Tested vs Simulated','(monthly global solar radiation)'})
    xlabel('Data Observation (MJ/m^2)');
    ylabel('Data Simulated (MJ/m^2)');
    %add line
    nnR = corrcoef(chkOut,dataSimMLR);
    nnR = nnR(1,2);
    coeffs = polyfit(chkOut, dataSimMLR, 1);
    % Get fitted values
    fittedX = linspace(min(chkOut), max(chkOut), 198);
    fittedY = polyval(coeffs, fittedX);
    C = round(polyval(coeffs, fittedX(1,1),0));
    plot(fittedX, fittedY, 'r-', 'LineWidth', 1);
    txt1 = ['y = ' num2str(round(nnR,3)) 'x + ' num2str(C)];
    ylim =get(gca,'ylim')-1
    text(min(chkOut),max(ylim), txt1);
    text(min(chkOut),max(ylim)-2, ['r^2 = ' num2str(round(nnR*nnR,3))]);
    hold off;
  %histogram
    subplot(2,2,2)
    dataErr = abs(chkOut - dataSimMLR);
    hmin = min(dataErr);
    hmax = max(dataErr);
    hold on;
    hist(dataErr,[hmin:0.25:hmax]);
    %title('Histogram absolute error frequency(monthly global solar radiation)');
    title({'Histogram absolute error frequency','(monthly global solar radiation)'})
    ylabel('Frequency of Error');
    xlabel('Errors (brackets ± 0.25)');
    hold off;    
% Timeseries
    %figure
    subplot(2,1,2)
    plot(chkOut,'k');
    hold on;
    plot(dataSimMLR);
    %title('Timeseries plot Tested vs Simulated (monthly global solar radiation)');
    title({'Timeseries plot Tested vs Simulated','(monthly global solar radiation)'})
    xlabel('Months')
    ylabel('\Delta [MJ/m^2]')
    var = {{'Observation','MLR'},'Location','northeast'}
    legend(var{:})
    %legend({'Test','arima1','arima2','t1mamdami','t1Sugerno','allSugerno','nt1Sugerno','en50ANFIS','en100ANFIS','SOMANFIS'},'Location','northwest','NumColumns',2)
    hold off 

%------------------------------------------------------------
elseif CO == 3
    
    %Random Forest Model variables
    tic % starts the timer.
    leaf=5; % this number could be varied.
    ntrees=800; % this number could be varied.
    fboot=1; % this number could be varied.
    surrogate='on'; % this could be set ‘on’ or ‘off’
    
    pn=trnIn(1:499,:)';
    qn=trnOut(1:499,:)';

    valP=trnIn(500:1248,:);
    valT=trnOut(500:1248,:);
    
%      for x = 1:nModels  
%          b = TreeBagger(ntrees,valP,valT,'Method','regression',...
%         'oobvarimp','on','surrogate',surrogate,...
%         'minleaf',leaf,'FBoot',fboot,'Options',paroptions);
%     
%         %Predict with traing ------------------
%         y = predict(b, valP);
%         simulatedTrain = y;
%         dataObsTrain = trnOut;
%         mseTrain = oobError(b,'mode','ensemble'); %single MSE for RF
% 
%         %Predict with testing ------------------
%         dataSim = predict(b,chkIn);
%         dataObs = chkOut;
%         runs(:,x) = dataSim;
%         
%         %Assement check error metrics --------------------------------
%         [nnR,nnENS,nnD,nnPDEV,nnRMSE,nnMAE,nnPI]=asseMetric(dataObs,dataSim);
%         ErrorsTest = nnPI;
%         %asseMetricVis(dataObs,dataSim,nnR,1,'Random Trees - Test Errors');
%         runsErrorTestRF(:,x) = nnPI;
% 
%      end

 % Random Forest visuals       
RFModel=xlsread('Mat8180Assignment2-Results.xlsx','RF_Result');    
RFModel = RFModel(:,11);  

   subplot(2,2,1)
    %figure
    hold on;
    scatter(chkOut,RFModel)
    %title('Scatter plot Tested vs Simulated (monthly global solar radiation)');
    title({'Scatter plot Tested vs Simulated','(monthly global solar radiation)'})
    xlabel('Data Observation (MJ/m^2)');
    ylabel('Data Simulated (MJ/m^2)');
    %add line
    nnR = corrcoef(chkOut,RFModel);
    nnR = nnR(1,2);
    coeffs = polyfit(chkOut, RFModel, 1);
    % Get fitted values
    fittedX = linspace(min(chkOut), max(chkOut), 198);
    fittedY = polyval(coeffs, fittedX);
    C = round(polyval(coeffs, fittedX(1,1),0));
    plot(fittedX, fittedY, 'r-', 'LineWidth', 1);
    txt1 = ['y = ' num2str(round(nnR,3)) 'x + ' num2str(C)];
    ylim =get(gca,'ylim')-2
    text(min(chkOut),max(ylim), txt1);
    text(min(chkOut),max(ylim)-2, ['r^2 = ' num2str(round(nnR*nnR,3))]);
    hold off;
  %histogram
    subplot(2,2,2)
    dataErr = abs(chkOut - RFModel);
    hmin = min(dataErr);
    hmax = max(dataErr);
    hold on;
    hist(dataErr,[hmin:0.25:hmax]);
    %title('Histogram absolute error frequency(monthly global solar radiation)');
    title({'Histogram absolute error frequency','(monthly global solar radiation)'})
    ylabel('Frequency of Error');
    xlabel('Errors (brackets ± 0.25)');
    hold off;    
% Timeseries
    %figure
    subplot(2,1,2)
    plot(chkOut,'k');
    hold on;
    plot(RFModel);
    %title('Timeseries plot Tested vs Simulated (monthly global solar radiation)');
    title({'Timeseries plot Tested vs Simulated','(monthly global solar radiation)'})
    xlabel('Months')
    ylabel('\Delta [MJ/m^2]')
    var = {{'Observation','RF'},'Location','northeast'}
    legend(var{:})
    %legend({'Test','arima1','arima2','t1mamdami','t1Sugerno','allSugerno','nt1Sugerno','en50ANFIS','en100ANFIS','SOMANFIS'},'Location','northwest','NumColumns',2)
    hold off     
     
end



