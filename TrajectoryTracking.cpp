#if defined(__linux__) || defined(__APPLE__)
#include <fcntl.h>
#include <termios.h>
#define STDIN_FILENO 0
#elif defined(_WIN32) || defined(_WIN64)
#include <conio.h>
#endif

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include <signal.h>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include "dynamixel_sdk.h"

#define ADDR_TORQUE_ENABLE             64
#define ADDR_GOAL_CURRENT              102
#define ADDR_PRESENT_POSITION          132
#define ADDR_PRESENT_CURRENT           126
#define LEN_GOAL_CURRENT                2
#define LEN_PRESENT_CURRENT             2
#define LEN_PRESENT_POSITION            4
#define PROTOCOL_VERSION                2.0
#define DXL1_ID                         1
#define DXL2_ID                         2
#define BAUDRATE                        2000000
#define DEVICENAME                      "/dev/ttyUSB0"
#define TORQUE_ENABLE                   1
#define TORQUE_DISABLE                  0
#define ESC_ASCII_VALUE                 0x1b
#define PI                              3.14159265
#define ENCODER_RESOLUTION              0.087912
#define SAFETY_THRESHOLD                1.1    // 0.515542 is threshold for motor side frame which is almost 30 degrees
#define CURRENT_RESOLUTION              0.00269
#define FRAME_WIDTH                     320
#define FRAME_HEIGHT                    240
#define CAMERA_FPS                      120
#define X_PIXEL                         119.0
#define Y_PIXEL                         119.0
#define X_SIZE                          0.1524  // cm
#define Y_SIZE                          0.1524 // cm
#define MAX_AREA                        150
#define WEIGHT_CURRENT                  0.6
#define WHILE_LOOP_TIME                 40.00
#define STABILIZATION_TIME              10.00
#define CIRCLE_RADIUS                   0.08
#define CIRCLE_FREQUENCY                2.0
#define DIVISION_CORRECTION             10.0

float r = 2.54 / 200;
float mb = 66.88 / 1000;
float g = 9.81;
float Ib = (2.0/5.0)*mb*powf(r,2);
float mp = 0.5266;
float a = 0.3048;
float Ip = (1/12.0)*mp*powf(a,2);
float L = 0.05608;
float d = 0.1397;
float K1 = 2401;
float K2 = 1372;
float K3 = 294;
float K4 = 28;
float vx = 0;
float vy = 0;
int M1_n = 0;
int M2_n = 0;
float M1_I = 0;
float M2_I = 0;
float M1_tau = 0;
float M2_tau = 0;
float tau1 = 0;
float tau2 = 0;
float M1_theta = 0;
float M2_theta = 0;
float theta1 = 0;
float theta2 = 0;
float theta1_prev = 0;
float theta2_prev = 0;
float dtheta1 = 0;
float dtheta2 =0;
float dtheta1_prev = 0;
float dtheta2_prev = 0;
float x = 0;
float y = 0;
float x_prev = 0;
float y_prev = 0;
float dx = 0;
float dy = 0;
float dx_prev = 0;
float dy_prev = 0;
float x_desired = 0;
float dx_desired = 0;
float ddx_desired = 0;
float d3x_desired = 0;
float d4x_desired = 0;
float y_desired = 0;
float dy_desired = 0;
float ddy_desired = 0;
float d3y_desired = 0;
float d4y_desired = 0;
float velocity_filter = 0.6;
float Ax = 0;
float Bx = 0;
float zeta1x = 0;
float zeta2x = 0;
float zeta3x = 0;
float zeta4x = 0;
float zeta1x_desired = 0;
float zeta2x_desired = 0;
float zeta3x_desired = 0;
float zeta4x_desired = 0;
float alphax = 0;
float betax = 0;
float Ay = 0;
float By = 0;
float zeta1y = 0;
float zeta2y = 0;
float zeta3y = 0;
float zeta4y = 0;
float zeta1y_desired = 0;
float zeta2y_desired = 0;
float zeta3y_desired = 0;
float zeta4y_desired = 0;
float alphay = 0;
float betay = 0;
int iteration = 1;
uint8_t param_M1_n[2];
uint8_t param_M2_n[2];

int int_x = 0;
int int_y = 0;
cv::VideoCapture cap(1);

dynamixel::PortHandler *portHandler = dynamixel::PortHandler::getPortHandler(DEVICENAME);
dynamixel::PacketHandler *packetHandler = dynamixel::PacketHandler::getPacketHandler(PROTOCOL_VERSION);
dynamixel::GroupSyncWrite groupSyncWrite(portHandler, packetHandler, ADDR_GOAL_CURRENT, LEN_GOAL_CURRENT);
dynamixel::GroupSyncRead groupSyncReadPosition(portHandler, packetHandler, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION);
int dxl_comm_result = COMM_TX_FAIL;
bool dxl_addparam_result = false;
bool dxl_getdata_result = false;
uint8_t dxl_error = 0;
int32_t dxl1_present_position = 0, dxl2_present_position = 0;

class Camera{
  int minh = 50;
  int maxh = 100;
  int mins = 130;
  int maxs = 255;
  int minv = 0;
  int maxv = 100;
  int max_area = 0;
  bool flag = false;
  int index_ = 0;
  uint i = 0;
  cv::Mat src, hsv, mask;
  cv::Moments mu;
  cv::Point2f mc;
  std::vector<std::vector<cv::Point> > Contours;
  std::vector<cv::Vec4i> hierarchy;
public:
  Camera();
  void Ball_Detecting();
};

void Signal_Handler(int);
void Safety();
void Motor_Initialization();
void Ball_Position();
void Motor_Angle();
void Taking_Derivative(float);
void Desired_States(float, float);
void Controller(float);
void Command_To_Motors();
void Current_To_Previous(float, float);
void Data_Print(float, float);
void Disable_Motors();

int main(){
  signal( SIGINT, Signal_Handler);
  Camera camera;
  Motor_Initialization();
  auto start0 = std::chrono::steady_clock::now();
  auto start = std::chrono::steady_clock::now();
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_time = end-start0;
  std::chrono::duration<double> iteration_time = end-start;
  while(elapsed_time.count() < WHILE_LOOP_TIME){
    camera.Ball_Detecting();
    Ball_Position();
    Motor_Angle();
    end = std::chrono::steady_clock::now();
    elapsed_time = end - start0;
    iteration_time = end - start;
    start = end;
    Taking_Derivative(iteration_time.count());
    Desired_States(elapsed_time.count(), iteration_time.count());
    Controller(elapsed_time.count());
    Command_To_Motors();
    Current_To_Previous(elapsed_time.count(), iteration_time.count());
    Data_Print(elapsed_time.count(), iteration_time.count());
  }
  Disable_Motors();
  return 0;
}

//********************************* Functions **********************************
void Signal_Handler(int signal){
  printf("\nYou pressed Ctrl+C\n");
  dxl_comm_result = packetHandler->write1ByteTxRx(portHandler, DXL1_ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE, &dxl_error);
  if (dxl_comm_result != COMM_SUCCESS){
    printf("%s\n", packetHandler->getTxRxResult(dxl_comm_result));
  }
  else if (dxl_error != 0){
    printf("%s\n", packetHandler->getRxPacketError(dxl_error));
  }
  dxl_comm_result = packetHandler->write1ByteTxRx(portHandler, DXL2_ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE, &dxl_error);
  if (dxl_comm_result != COMM_SUCCESS){
    printf("%s\n", packetHandler->getTxRxResult(dxl_comm_result));
  }
  else if (dxl_error != 0){
    printf("%s\n", packetHandler->getRxPacketError(dxl_error));
  }
  portHandler->closePort();
  exit(1);
}

void Safety(){
  printf("\nMotor passed safety threshold\n");
  dxl_comm_result = packetHandler->write1ByteTxRx(portHandler, DXL1_ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE, &dxl_error);
  if (dxl_comm_result != COMM_SUCCESS){
    printf("%s\n", packetHandler->getTxRxResult(dxl_comm_result));
  }
  else if (dxl_error != 0){
    printf("%s\n", packetHandler->getRxPacketError(dxl_error));
  }
  dxl_comm_result = packetHandler->write1ByteTxRx(portHandler, DXL2_ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE, &dxl_error);
  if (dxl_comm_result != COMM_SUCCESS){
    printf("%s\n", packetHandler->getTxRxResult(dxl_comm_result));
  }
  else if (dxl_error != 0){
    printf("%s\n", packetHandler->getRxPacketError(dxl_error));
  }
  portHandler->closePort();
  exit(1);
}

void Motor_Initialization(){
  if (portHandler->openPort()){
    printf("Succeeded to open the port!\n");
  }
  else{
    printf("Failed to open the port!\n");
    printf("Press any key to terminate...\n");
    exit(1);
  }
  if (portHandler->setBaudRate(BAUDRATE)){
    printf("Succeeded to change the baudrate!\n");
  }
  else{
    printf("Failed to change the baudrate!\n");
    printf("Press any key to terminate...\n");
    exit(1);
  }
  dxl_comm_result = packetHandler->write1ByteTxRx(portHandler, DXL1_ID, ADDR_TORQUE_ENABLE, TORQUE_ENABLE, &dxl_error);
  if (dxl_comm_result != COMM_SUCCESS){
    printf("%s\n", packetHandler->getTxRxResult(dxl_comm_result));
  }
  else if (dxl_error != 0){
    printf("%s\n", packetHandler->getRxPacketError(dxl_error));
  }
  else{
    printf("Dynamixel#%d has been successfully connected \n", DXL1_ID);
  }
  dxl_comm_result = packetHandler->write1ByteTxRx(portHandler, DXL2_ID, ADDR_TORQUE_ENABLE, TORQUE_ENABLE, &dxl_error);
  if (dxl_comm_result != COMM_SUCCESS){
    printf("%s\n", packetHandler->getTxRxResult(dxl_comm_result));
  }
  else if (dxl_error != 0){
    printf("%s\n", packetHandler->getRxPacketError(dxl_error));
  }
  else{
    printf("Dynamixel#%d has been successfully connected \n", DXL2_ID);
  }
  dxl_addparam_result = groupSyncReadPosition.addParam(DXL1_ID);
  if (dxl_addparam_result != true){
    fprintf(stderr, "[ID:%03d] groupSyncReadPosition addparam failed\n", DXL1_ID);
    exit(1);
  }
  dxl_addparam_result = groupSyncReadPosition.addParam(DXL2_ID);
  if (dxl_addparam_result != true){
    fprintf(stderr, "[ID:%03d] groupSyncReadPosition addparam failed\n", DXL2_ID);
    exit(1);
  }
}

Camera::Camera(){
  cap.set(CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
  cap.set(CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);
  cap.set(CV_CAP_PROP_FPS, CAMERA_FPS);
  if (cap.isOpened() == false) {
    printf("Error: Camera doesn't work or can't be accessed\n\n");
    exit(1);
  }
}

void Camera::Ball_Detecting(){
  if (cap.read(src) == 0) {
    printf("Error: frame not read from Camera\n\n");
    exit(1);
  }
  cv::cvtColor(src,hsv,CV_BGR2HSV);
  cv::inRange(hsv, cv::Scalar(minh, mins, minv), cv::Scalar(maxh, maxs, maxv),mask);
  cv::findContours(mask,Contours,hierarchy,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE,cv::Point(0,0));
  max_area = 0;
  index_ = 0;
  flag = false;
  for(i = 0; i < Contours.size(); i++){
    if(cv::contourArea(Contours[i]) > max_area){
      max_area = cv::contourArea(Contours[i]);
      if(max_area > MAX_AREA){
        index_ = i;
        flag = true;
      }
    }
  }
  if(flag == true){
    mu = cv::moments(Contours[index_], false);
    mc = cv::Point2f(mu.m10/mu.m00, mu.m01/mu.m00);
  }
  else{
    mc = cv::Point2f(FRAME_WIDTH/2, FRAME_HEIGHT/2);
  }
  int_x = mc.x - FRAME_WIDTH/2;
  int_y = -(mc.y - FRAME_HEIGHT/2);
}

void Ball_Position(){
  x = (int_x/X_PIXEL)*X_SIZE;
  y = (int_y/Y_PIXEL)*Y_SIZE;
}

void Motor_Angle(){
  dxl_comm_result = groupSyncReadPosition.txRxPacket();
  if (dxl_comm_result != COMM_SUCCESS){
    printf("%s\n", packetHandler->getTxRxResult(dxl_comm_result));
  }
  dxl_getdata_result = groupSyncReadPosition.isAvailable(DXL1_ID, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION);
  if (dxl_getdata_result != true){
    fprintf(stderr, "[ID:%03d] groupSyncReadPosition getdata failed\n", DXL1_ID);
    exit(1);
  }
  dxl_getdata_result = groupSyncReadPosition.isAvailable(DXL2_ID, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION);
  if (dxl_getdata_result != true){
    fprintf(stderr, "[ID:%03d] groupSyncReadPosition getdata failed\n", DXL2_ID);
    exit(1);
  }
  dxl1_present_position = groupSyncReadPosition.getData(DXL1_ID, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION);
  dxl2_present_position = groupSyncReadPosition.getData(DXL2_ID, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION);
  M1_theta = (dxl1_present_position-4095) * ENCODER_RESOLUTION * (PI/180.0);
  M2_theta = (dxl2_present_position-4095) * ENCODER_RESOLUTION * (PI/180.0);
  if(M1_theta > SAFETY_THRESHOLD || M2_theta > SAFETY_THRESHOLD){
    Safety();
  }
  theta1 = (L/d)*sin(M1_theta);
  theta2 = (L/d)*sin(M2_theta);
  if(iteration == 1){
    theta1_prev = theta1;
    theta2_prev = theta2;
  }
  if(iteration == 1){
    x_prev = x;
    y_prev = y;
    iteration = 2;
  }
}

void Taking_Derivative(float iteration_time){
  dx = (x - x_prev) / iteration_time;
  dx = velocity_filter*dx + (1-velocity_filter)*dx_prev;
  dy = (y - y_prev) / iteration_time;
  dy = velocity_filter*dy + (1-velocity_filter)*dy_prev;
  dtheta1 = (theta1 - theta1_prev) / iteration_time;
  dtheta1 = velocity_filter*dtheta1 + (1-velocity_filter)*dtheta1_prev;
  dtheta2 = (theta2 - theta2_prev) / iteration_time;
  dtheta2 = velocity_filter*dtheta2 + (1-velocity_filter)*dtheta2_prev;
}

void Desired_States(float elapsed_time, float iteration_time){
  x_desired = CIRCLE_RADIUS*sin(CIRCLE_FREQUENCY*elapsed_time);
  dx_desired = CIRCLE_RADIUS*CIRCLE_FREQUENCY*cos(CIRCLE_FREQUENCY*elapsed_time);
  ddx_desired = -CIRCLE_RADIUS*powf(CIRCLE_FREQUENCY,2)*sin(CIRCLE_FREQUENCY*elapsed_time);
  d3x_desired = -CIRCLE_RADIUS*powf(CIRCLE_FREQUENCY,3)*cos(CIRCLE_FREQUENCY*elapsed_time);
  d4x_desired = CIRCLE_RADIUS*powf(CIRCLE_FREQUENCY,4)*sin(CIRCLE_FREQUENCY*elapsed_time);
  zeta1x_desired = x_desired;
  zeta2x_desired = dx_desired;
  zeta3x_desired = ddx_desired;
  zeta4x_desired = d3x_desired;
  y_desired = CIRCLE_RADIUS*cos(CIRCLE_FREQUENCY*elapsed_time);
  dy_desired = -CIRCLE_RADIUS*CIRCLE_FREQUENCY*sin(CIRCLE_FREQUENCY*elapsed_time);
  ddy_desired = -CIRCLE_RADIUS*powf(CIRCLE_FREQUENCY,2)*cos(CIRCLE_FREQUENCY*elapsed_time);
  d3y_desired = CIRCLE_RADIUS*powf(CIRCLE_FREQUENCY,3)*sin(CIRCLE_FREQUENCY*elapsed_time);
  d4y_desired = CIRCLE_RADIUS*powf(CIRCLE_FREQUENCY,4)*cos(CIRCLE_FREQUENCY*elapsed_time);
  zeta1y_desired = y_desired;
  zeta2y_desired = dy_desired;
  zeta3y_desired = ddy_desired;
  zeta4y_desired = d3y_desired;
}

void Controller(float elapsed_time){
  Ax=1/(Ib/powf(r,2)+mb);
  Bx=1/(mb*powf(x,2)+Ip+Ib);
  zeta1x=x;
  zeta2x=dx;
  zeta3x=Ax*(mb*x*powf(dtheta2,2)+mb*g*sin(theta2));
  zeta4x=Ax*(mb*g*dtheta2*cos(theta2)+mb*dx*powf(dtheta2,2));
  if(elapsed_time > STABILIZATION_TIME){
    vx = d4x_desired + K1*(zeta1x_desired - zeta1x) + K2*(zeta2x_desired - zeta2x) + K3*(zeta3x_desired - zeta3x) + K4*(zeta4x_desired - zeta4x);
  }
  else{
    vx = K1*(-zeta1x) + K2*(-zeta2x) + K3*(-zeta3x) + K4*(-zeta4x);
  }
  alphax=Ax*(Ax*mb*powf(dtheta2,2)*(mb*x*powf(dtheta2,2)+mb*g*sin(theta2))-mb*g*powf(dtheta2,2)*sin(theta2)+Bx*(2*mb*dx*dtheta2+mb*g*cos(theta2))*(-2*mb*x*dx*dtheta2+mb*g*x*cos(theta2)+mb*g*r*sin(theta2)));
  betax=Ax*Bx*(2*mb*dx*dtheta2+mb*g*cos(theta2));
  tau2=(vx-alphax)/betax;
  M2_tau = (d/L)*tau2/(cos(theta2)*cos(M2_theta));
  if(M2_tau > 0){
    M2_I = (M2_tau + 0.32)/1.3 + WEIGHT_CURRENT;
    M2_I = M2_I / DIVISION_CORRECTION;
  }
  else if(M2_tau < 0){
    M2_I = (M2_tau - 0.32)/1.3;
    M2_I = M2_I / DIVISION_CORRECTION;
  }
  else{
    M2_I = 0;
  }
  Ay=1/(mb+Ib/powf(r,2));
  By=1/(Ib+Ip+mb*powf(y,2));
  zeta1y=y;
  zeta2y=dy;
  zeta3y=Ay*(mb*y*powf(dtheta1,2)-mb*g*sin(theta1));
  zeta4y=Ay*(mb*dy*powf(dtheta1,2)-mb*g*dtheta1*cos(theta1));
  if(elapsed_time > STABILIZATION_TIME){
    vy = d4y_desired + K1*(zeta1y_desired - zeta1y) + K2*(zeta2y_desired - zeta2y) + K3*(zeta3y_desired - zeta3y) + K4*(zeta4y_desired - zeta4y);
  }
  else{
    vy = K1*(-zeta1y) + K2*(-zeta2y) + K3*(-zeta3y) + K4*(-zeta4y);
  }
  alphay=Ay*(Ay*mb*powf(dtheta1,2)*(mb*y*powf(dtheta1,2)-mb*g*sin(theta1))+mb*g*powf(dtheta1,2)*sin(theta1)+By*(2*mb*dy*dtheta1-mb*g*cos(theta1))*(-2*mb*y*dy*dtheta1-mb*g*y*cos(theta1)+mb*g*r*sin(theta1)));
  betay=Ay*By*(2*mb*dy*dtheta1-mb*g*cos(theta1));
  tau1=(vy-alphay)/betay;
  M1_tau = (d/L)*tau1/(cos(theta1)*cos(M1_theta));
  if(M1_tau > 0){
    M1_I = (M1_tau + 0.32)/1.3 + WEIGHT_CURRENT;
    M1_I = M1_I / DIVISION_CORRECTION;
  }
  else if(M1_tau < 0){
    M1_I = (M1_tau - 0.32)/1.3;
    M1_I = M1_I / DIVISION_CORRECTION;
  }
  else{
    M1_I = 0;
  }
}

void Command_To_Motors(){
  M1_n = M1_I / CURRENT_RESOLUTION;
  M2_n = M2_I / CURRENT_RESOLUTION;
  param_M1_n[0] = DXL_LOWORD(M1_n);
  param_M1_n[1] = DXL_HIWORD(M1_n);
  param_M2_n[0] = DXL_LOWORD(M2_n);
  param_M2_n[1] = DXL_HIWORD(M2_n);
  dxl_addparam_result = groupSyncWrite.addParam(DXL1_ID, param_M1_n);
  if (dxl_addparam_result != true){
    fprintf(stderr, "[ID:%03d] groupSyncWrite addparam failed", DXL1_ID);
    exit(1);
  }
  dxl_addparam_result = groupSyncWrite.addParam(DXL2_ID, param_M2_n);
  if (dxl_addparam_result != true){
    fprintf(stderr, "[ID:%03d] groupSyncWrite addparam failed", DXL2_ID);
    exit(1);
  }
  dxl_comm_result = groupSyncWrite.txPacket();
  if (dxl_comm_result != COMM_SUCCESS){
    printf("%s\n", packetHandler->getTxRxResult(dxl_comm_result));
  }
  groupSyncWrite.clearParam();
}

void Current_To_Previous(float elapsed_time, float iteration_time){
  dx_prev = dx;
  x_prev = x;
  dy_prev = dy;
  y_prev = y;
  dtheta1_prev = dtheta1;
  theta1_prev = theta1;
  dtheta2_prev = dtheta2;
  theta2_prev = theta2;
}

void Data_Print(float elapsed_time, float iteration_time){
  printf("%f6 , %f6 , %f6 , %f6 , %f6 , %f6 , %f6 , %f6 , %f6 , %f6 , %f6\n", theta1, dtheta1, theta2, dtheta2, x, dx, y, dy, tau1, tau2, elapsed_time);
}

void Disable_Motors(){
  dxl_comm_result = packetHandler->write1ByteTxRx(portHandler, DXL1_ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE, &dxl_error);
  if (dxl_comm_result != COMM_SUCCESS){
    printf("%s\n", packetHandler->getTxRxResult(dxl_comm_result));
  }
  else if (dxl_error != 0){
    printf("%s\n", packetHandler->getRxPacketError(dxl_error));
  }
  dxl_comm_result = packetHandler->write1ByteTxRx(portHandler, DXL2_ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE, &dxl_error);
  if (dxl_comm_result != COMM_SUCCESS){
    printf("%s\n", packetHandler->getTxRxResult(dxl_comm_result));
  }
  else if (dxl_error != 0){
    printf("%s\n", packetHandler->getRxPacketError(dxl_error));
  }
  portHandler->closePort();
}
