
% Clear workspace
clear all, close all, clc
format compact

% Open serial port
com = serial('COM5','InputBufferSize',100000);
fopen(com);

% Open binary file for saving inertial data
filename = 'mimu_data10.bin';
file = fopen(filename, 'w');

% Flush serial ports
while com.BytesAvailable
    fread(com,com.BytesAvailable,'uint8');
end
% Make sure data is read from the IMUs
command = [48 19 0 0 67];
fwrite(com,command,'uint8');
fread(com,4,'uint8');

% Request raw inertial data
 nr_imus=4;
%nr_imus=32;
header = 40;
rate_divider = 4;
imu_mask = [0 0 0 15];
%imu_mask = [255 255 255 255];
command = [header imu_mask rate_divider+64];
command = [command (sum(command)-mod(sum(command),256))/256 mod(sum(command),256)];
%command = [40 0 0 0 15 65 0 120]
fwrite(com,command,'uint8');
fread(com,4,'uint8');

% Open dummy figure with pushbutton such that logging can be aborted
abort = 0;
figure(10);
uicontrol('style','push','string','Abort data logging','callback','abort=1;');
drawnow

% Logg data until pushbutton pressed
while abort==0
    if com.BytesAvailable>0
        fwrite(file,fread(com,com.BytesAvailable,'uint8'),'uint8');
    end
    drawnow
end

% Stop output
fwrite(com,[34 0 34],'uint8');

% Close serial port and file
fclose(com);
fclose(file);
close(gcf)

% Parse data and delete logging file
 [inertial_data,time_stamps,raw_data]=mimu_parse_bin(filename,uint8(nr_imus));
%delete(filename);

% Plot data in SI units
 inertial_data_double = double(inertial_data);

scale_acc  = 1/2048*9.80665;
scale_gyro = 1/16.4;
figure(1),clf, hold on
  for i=0:nr_imus-1
  %  for i=2
    plot(inertial_data_double(i*6+1,:)'*scale_acc,'b-')
    plot(inertial_data_double(i*6+2,:)'*scale_acc,'g-')
    plot(inertial_data_double(i*6+3,:)'*scale_acc,'r-')
end
grid on
title('Accelerometer readings');
xlabel('sample number')
ylabel('a [m/s^2]');
figure(2),clf, hold on
   for i=0:nr_imus-1
  %for i=2
    plot(inertial_data_double(i*6+4,:)'*scale_gyro,'b-')
    plot(inertial_data_double(i*6+5,:)'*scale_gyro,'g-')
    plot(inertial_data_double(i*6+6,:)'*scale_gyro,'r-')
end
grid on
title('Gyroscope readings');
xlabel('sample number')
ylabel('\omega [deg/s]');
figure(3),clf, hold on
subplot(2,1,1);
plot(double(time_stamps)'/64e6,'b-');
grid on
title('Time stamps');
xlabel('sample number')
ylabel('[s]');
subplot(2,1,2);
dt = diff(double(time_stamps)');
%plot(dt,'b-');
for i=1:numel(dt)
    if dt(i)<0
        dt(i) = dt(i)+2^32;
    end
end
plot(dt/64e6,'b-');
grid on
title('Time differentials');
xlabel('sample number')
ylabel('[s]');
