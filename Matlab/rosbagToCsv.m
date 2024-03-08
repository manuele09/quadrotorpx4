% rosbag record -a
% rosbag('info', "/home/emanuele/Desktop/data.bag");
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
V = zeros(numMessages, 1);
state = zeros(numMessages, 9);
action = zeros(numMessages, 4);
action_lqr = zeros(numMessages, 4);

for i = 1:numMessages
    V(i) = msgStructs{i}.V;
    state(i, :) = msgStructs{i}.State';
    action(i, :) = msgStructs{i}.Action';
    action_lqr(i, :) = msgStructs{i}.ActionLqr';
end


dataMatrixController = [state, action];
filePath = 'controller.csv';
writematrix(dataMatrixController, filePath, 'WriteMode', 'overwrite', 'Delimiter', ',');
disp(['Controller dataset written to ' filePath]);

dataMatrixController = [state, action_lqr];
filePath = 'controller_lqr.csv';
writematrix(dataMatrixController, filePath, 'WriteMode', 'overwrite', 'Delimiter', ',');
disp(['Controller Lqr dataset written to ' filePath]);


delta_state = state(2:end, :) - state(1:end-1, :);
% Roll, Pitch, Yaw, Action, Delta Vel, Delta RPY
dataMatrixForward = [state(1:end-1, 4:6), action(1:end-1, :), delta_state(:, 7:9), delta_state(:, 4:6)];
filePath = 'forward.csv';
writematrix(dataMatrixForward, filePath, 'WriteMode', 'overwrite', 'Delimiter', ',');
disp(['Forward dataset written to ' filePath]);


dataMatrixLyapunov = [state, V];
filePath = 'lyapunov.csv';
writematrix(dataMatrixLyapunov, filePath, 'WriteMode', 'overwrite', 'Delimiter', ',');
disp(['Lyapunov dataset written to ' filePath]);