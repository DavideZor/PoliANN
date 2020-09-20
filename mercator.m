%The mercator function is a function that takes as input the longitude
%and latitude vectors and transforms them according to the Mercator
%projection. To get the values in units of distance you must multiply 
%this by the radius of the earth and then divide by the scaleFactor. 
%
%-------------------------------------------------------------------------
%Input arguments:
%lon         [nx1]     longitude of the n points in a nx1 vector      [-]
%lat         [nx1]     latitude of the n points in a nx1 vector       [-]
%
%-------------------------------------------------------------------------
%Output arguments:
%x           [nx1]     abscissa of the Mercator projection            [-]
%y           [nx1]     ordinate of the Mercator projection            [-]

function [x,y2,scaleFactor] = mercator(lon,lat,varargin)

if ischar(lon) || ischar(lat)
    warning('string input has been changed to numbers');
end

if ischar(lon)
    lon=str2double(lon);
end
if ischar(lat)
    lat=str2double(lat);
end



if isempty(varargin) % do the real projection
    x = deg2rad(lon);
    y = deg2rad(lat);
    % Projection:
    y2 = log(abs(tan(y)+sec(y)));
%   y2 = log(tan(pi/4+y/2));
    scaleFactor = sec(y);
    
else % do the inverse projection
    x = rad2deg(lon);
    y2 = atan(sinh(lat));
    y2 = rad2deg(y2);
    
    scaleFactor=sec(lat);
end

function deg = rad2deg(rad)
deg = rad*180/pi;

function rad = deg2rad(deg)
rad = deg*pi/180;
