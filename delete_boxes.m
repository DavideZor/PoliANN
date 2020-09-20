%The delete_boxes function is a function that takes as input a matrix
%of any dimension and it returns the same matrix without the rows where 
%the elements are included in a box x1 < x < x2 and y1 < y < y2, with 
%x1, x2, y1, y2 being the delimiters of the boxes (listed in another 
%matrix D).
%
%-------------------------------------------------------------------------
%Input arguments:
%M            [nx2]     Generic nx2 matrix                         [-]
%D            [nx4]     Generic nx4 matrix                         [-]
%
%--------------------------------------------------------------------------
%Output arguments:
%R            [pxq]     Matrix with all the elements in boxes 
%                       deleted                                    [-]


function [R] = delete_boxes(M, D)

[n,m] = size(D);

for i = 1:n
    logical = M(:,1) > D(i,1) & M(:,1) < D(i,2) & ...
        M(:,2) > D(i,3) & M(:,2) < D(i,4);
    flag = any(logical);
    while flag
        k = min(find(logical));
        M(k,:) = [0, 0];
        logical = (M(:,1) > D(i,1) & M(:,1) < D(i,2) & ...
            M(:,2) > D(i,3) & M(:,2) < D(i,4));
        flag = any(logical);
    end
end

R = delete_null_rows(M);

end