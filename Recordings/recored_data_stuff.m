clear all, close all, clc
format compact
nr_imus=4;
filename = 'second/mimu_data1.bin';
[inertial_data,time_stamps,raw_data]=mimu_parse_bin(filename,uint8(nr_imus));
inertial_data_double = double(inertial_data);

scale_acc  = (1/2048)*9.80665;
%scale_acc = 1/2048;
scale_gyro = 1/16.4;

  for i=0:nr_imus-1
    x_acc(i+1,:) = inertial_data_double(i*6+1,:)*scale_acc;
    y_acc(i+1,:) = inertial_data_double(i*6+2,:)*scale_acc;
    z_acc(i+1,:) = inertial_data_double(i*6+3,:)*scale_acc;
  end
%m = [inertial_data_double(1,:); inertial_data_double(2,:); inertial_data_double(3,:)]'*scale_acc;
m = [x_acc; y_acc; z_acc]';
%m(1,:) = inertial_data_double(1,:)*scale_acc;
%m(2,:) = inertial_data_double(2,:)*scale_acc;
%m(3,:) = inertial_data_double(3,:)*scale_acc;
csvwrite('second/acce_data1.csv', m) 