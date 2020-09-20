%The delete_isnan function is a function that takes as input a matrix
%of any dimension and it returns the same matrix without the rows where all
%the elements were equal to NaN. If there are no NaN rows in the input
%matrix the algorithm returns the same matrix without any modification.
%
%-------------------------------------------------------------------------
%Input arguments:
%M            [nxm]     Generic nxm matrix                         [-]
%
%--------------------------------------------------------------------------
%Output arguments:
%R            [pxq]     Matrix with all NaN rows deleted           [-]

function [R] = delete_isnan(M)

[n,m] = size(M);

for i = 1:n
    if all(isnan(M(i,:)))
        M(i,:) = zeros(1,m);
    end
end

R = delete_null_rows(M);

end