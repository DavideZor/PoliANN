clc, clear all, close all;

%% Reading data

% High resolution data
Italy = shaperead('ITA_adm0.shp');

% % Low resolution data
% countries = shaperead('ne_10m_admin_0_countries.shp');
% Italy = countries(92);
% % mapshow(Italy) %MATLAB function to plot 2D polygons

lon = Italy.X; %Longitude
lat = Italy.Y; %Latitude

figure();
plot(lon, lat), grid on, axis equal;
title('Italy');
xlabel('Longitude [°]'), ylabel('Latitude [°]');

%% Map cleaning

%Here the regions that needs to be deleted/selected are listed as
%list = [x1, x2, y1, y2]
%where the square x1 < x < x2, y1 < y < y2 is going to be deleted/selected

map = [lon', lat'];

smallislands = [12.38, 12.5, 43.88, 43.99;
    9.7, 10.468, 42.2, 43.1;
    10.85, 10.95, 42.31, 42.4;
    13.3, 14, 40.55, 40.85;
    12.315, 12.365, 45.4273, 45.46;
    12.36, 12.44, 45.4, 45.435;
    12.31, 12.38, 45.34, 45.42;
    11.6, 13, 35, 37;
    14.2, 15.4, 38.3, 38.9;
    12.9, 13.5, 38.55, 38.8;
    8.18, 8.38, 40.98, 41.15;
    9.37, 9.44, 41.2, 41.27;
    9.4365, 9.485, 41.17, 41.25;
    8.22, 8.32, 39.08, 39.2;
    8.34, 8.5, 38.96, 39.08;
    8.34, 8.46, 39.08, 39.115;
    8.356, 8.37, 39.115, 39.1185;
    12, 12.4, 37.7, 38.2;
    12.42, 12.49, 41.85, 41.95];

cleanItaly = delete_boxes(map, smallislands);

lon_clean = cleanItaly(:,1)';
lat_clean = cleanItaly(:,2)';

% figure();
% plot(lon_clean, lat_clean), grid on, axis equal;
% title('Italy without small islands');
% xlabel('Longitude [°]'), ylabel('Latitude [°]');

sardinia = [8, 10, 38, 42];

sicily = [12, 15.5, 36.6, 38.4;
    15.5, 15.62, 38, 38.4; 
    15.62, 15.66, 38.25, 38.29];
    
peninsula_map = delete_boxes(cleanItaly, [sicily; sardinia]);

peninsula_map = delete_isnan(peninsula_map);

lon_peninsula = peninsula_map(:,1)';
lat_peninsula = peninsula_map(:,2)';

% figure();
% plot(lon_peninsula, lat_peninsula), grid on, axis equal;
% title('Italian Peninsula');
% xlabel('Longitude [°]'), ylabel('Latitude [°]');

sardinia_map = select_boxes(cleanItaly, sardinia);

lon_sardinia = sardinia_map(:,1)';
lat_sardinia = sardinia_map(:,2)';

% figure();
% plot(lon_sardinia, lat_sardinia), grid on, axis equal;
% title('Sardinia');
% xlabel('Longitude [°]'), ylabel('Latitude [°]');

sicily_map = select_boxes(cleanItaly, sicily);

lon_sicily = sicily_map(:,1)';
lat_sicily = sicily_map(:,2)';

% figure();
% plot(lon_sicily, lat_sicily), grid on, axis equal;
% title('Sicily');
% xlabel('Longitude [°]'), ylabel('Latitude [°]');

%% Mercator projection (dimensionless)

%Uncomment the following section if the dimensionless plots are needed

%The Mercator projection is calculated
[x_peninsula, y_peninsula, ScaleFactorPe] = mercator(lon_peninsula, lat_peninsula);
xy_peninsula = [x_peninsula', y_peninsula'];

figure();
plot(x_peninsula, y_peninsula), grid on, axis equal;
title('Italian Peninsula (Mercator projection)');
xlabel('X [-]'), ylabel('Y [-]');

%The Mercator projection is calculated
[x_sardinia, y_sardinia, ScaleFactorSa] = mercator(lon_sardinia, lat_sardinia);
xy_sardinia = [x_sardinia', y_sardinia'];

figure();
plot(x_sardinia, y_sardinia), grid on, axis equal;
title('Sardinia (Mercator projection)');
xlabel('X [-]'), ylabel('Y [-]');

%The Mercator projection is calculated
[x_sicily, y_sicily, ScaleFactorSi] = mercator(lon_sicily, lat_sicily);
xy_sicily = [x_sicily', y_sicily'];

figure();
plot(x_sicily, y_sicily), grid on, axis equal;
title('Sicily (Mercator projection)');
xlabel('X [-]'), ylabel('Y [-]');

%% Dimensional Mercator projection

%Uncomment the following section if the dimensional plots are needed

% %In order to obtain the dimensional quantities (in km)
% earth_radius = 6378.1;
% meanScaleFactorPe = mean(ScaleFactorPe);
% x_peninsula_km = x_peninsula*earth_radius/meanScaleFactorPe;
% y_peninsula_km = y_peninsula*earth_radius/meanScaleFactorPe;
% xy_peninsula_km = [x_peninsula_km, y_peninsula_km];
% 
% figure();
% plot(x_peninsula_km, y_peninsula_km), grid on, axis equal;
% title('Italian Peninsula (Dimensional Mercator projection)');
% xlabel('X [km]'), ylabel('Y [km]');
% 
% %In order to obtain the dimensional quantities (in km)
% earth_radius = 6378.1;
% meanScaleFactorSa = mean(ScaleFactorSa);
% x_sardinia_km = x_sardinia*earth_radius/meanScaleFactorSa;
% y_sardinia_km = y_sardinia*earth_radius/meanScaleFactorSa;
% xy_sardinia_km = [x_sardinia_km, y_sardinia_km];
% 
% figure();
% plot(x_sardinia_km, y_sardinia_km), grid on, axis equal;
% title('Sardinia (Dimensional Mercator projection)');
% xlabel('X [km]'), ylabel('Y [km]');
% 
% %In order to obtain the dimensional quantities (in km)
% earth_radius = 6378.1;
% meanScaleFactorSi = mean(ScaleFactorSi);
% x_sicily_km = x_sicily*earth_radius/meanScaleFactorSi;
% y_sicily_km = y_sicily*earth_radius/meanScaleFactorSi;
% xy_sicily_km = [x_sicily_km, y_sicily_km];
% 
% figure();
% plot(x_sicily_km, y_sicily_km), grid on, axis equal;
% title('Sicily (Dimensional Mercator projection)');
% xlabel('X [km]'), ylabel('Y [km]');


%% Generation of .txt files

%Uncomment the following section if the files needs to be updated

% dlmwrite('Italy.txt',map);
% dlmwrite('Italy_Cleaned.txt',map);
% dlmwrite('Italy_Peninsula_LatLon.txt', peninsula_map);
% dlmwrite('Sardinia_LatLon.txt', sardinia_map);
% dlmwrite('Sicily_LatLon.txt', sicily_map);
% dlmwrite('Italy_Peninsula_XY.txt', xy_peninsula);
% dlmwrite('Sardinia_XY.txt', xy_sardinia);
% dlmwrite('Sicily_XY.txt', xy_sicily);
% dlmwrite('Italy_Peninsula_XY_km.txt', xy_peninsula_km);
% dlmwrite('Sardinia_XY_km.txt', xy_sardinia_km);
% dlmwrite('Sicily_XY_km.txt', xy_sicily_km);
