clc
clear all
close all

%% 
D = load("Dati_sistema_controllore.csv");

%% Fix colonne control action: 
% kethrust_new = - (-9.81 - kethrust_att)
% rate_new = -(rate_att)

D(:,10:12) = - D(:,10:12);
D(:,13) = -(-9.81 - D(:,13));

%% 
writematrix(D, 'Dati_sistema_controllore_fixed.csv')