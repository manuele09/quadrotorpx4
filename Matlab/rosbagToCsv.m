% rosbag record -a
% rosbag('info', "/Rosbag/data.bag");
clear
bag = rosbag("/home/emanuele/Desktop/data.bag");
bSel = select(bag,'Topic',{"/mavros/controlactiondrone"});
msgStructs = readMessages(bSel,'DataFormat','struct');
% msgStructs{1}

disp('Bag file loaded.');

numMessages = numel(msgStructs);
disp(['Total number of messages: ' num2str(numMessages)]);
% numMessages = min(numMessages, 10000);
disp(['Number of messages used: ' num2str(numMessages)]);

% Extract data from the msgStructs structure array
thrust = zeros(numMessages, 1);
V = zeros(numMessages, 1);
Uref = zeros(numMessages, 4);
position = zeros(numMessages, 9);
error = zeros(numMessages, 9);
KeYaw = zeros(numMessages, 1);
KeRoll = zeros(numMessages, 1);
KeThrust = zeros(numMessages, 1);
KePitch = zeros(numMessages, 1);


for i = 1:numMessages
    thrust(i) = msgStructs{i}.Thrust;
    V(i) = msgStructs{i}.V;
    Uref(i, :) = msgStructs{i}.Uref';
    position(i, :) = msgStructs{i}.Position';
    error(i, :) = msgStructs{i}.Error';
    KeRoll(i) = msgStructs{i}.KeRoll;
    KePitch(i) = msgStructs{i}.KePitch;
    KeYaw(i) = msgStructs{i}.KeYaw;
    KeThrust(i) = msgStructs{i}.KeThrust;
end



% dataMatrix = [error, KeYaw, KeRoll, KePitch, KeThrust];

input0 = Uref(:, 1) - KeYaw;
input1 = Uref(:, 2) - KeRoll;
input2 = Uref(:, 3) - KePitch;
input3 = Uref(:, 4) - KeThrust;
dataMatrix = [position, input0, input1, input2, input3];

filePath = 'controller.csv';
writematrix(dataMatrix, filePath, 'WriteMode', 'overwrite', 'Delimiter', ',');
disp(['Data written to ' filePath]);

% dataMatrix = [position, input0, input1, input2, input3];
% input = Uref - K*e;