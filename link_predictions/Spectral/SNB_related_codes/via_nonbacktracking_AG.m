%% via non-backtracking method
function [k,r] = via_nonbacktracking_AG(A,eig_num,opts)
% NB method
n = size(A,1); d = sum(A,1); tolerance = 10^(-5);

B = [zeros(n), diag(d-1); -eye(n), A]; % non-backtracking matrix
[~,D] = eigs(B,eig_num,'lr',opts); eigval = diag(D); 
eigval_R = real(eigval); eigval_E = imag(eigval); 

threshold = sqrt(eigval_R(1));%max(eigval_R((eigval_E ~= 0)));

%bulk_radius = sqrt(mean(d));

k = sum((eigval_R > threshold) & (abs(eigval_E) < tolerance )); r = threshold;

% subplot(2,1,1)
% scatter(eigval_R,eigval_E,'.')
end