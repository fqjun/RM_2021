#include "Modules/RM_Buff/RM_Buff.hpp"

namespace buff {
RM_Buff::RM_Buff(const std::string& _buff_config_address)

{
  // 读取buff配置文件
  cv::FileStorage buff_config_fs(_buff_config_address, cv::FileStorage::READ);
  readBuffConfig(buff_config_fs);
  buff_config_fs.release();

  target_2d_point_.reserve(4);
  target_rect_ = RotatedRect();
  my_color_    = 0;

  /* 预处理 */
  split_img_.reserve(3);
  average_th_ = 0;

  /* 查找目标 */
  inaction_cnt_      = 0;
  small_rect_area_   = 0.f;
  small_rect_length_ = 0.f;
  big_rect_area_     = 0.f;
  big_rect_length_   = 0.f;

#ifndef RELEASE
  small_target_aspect_ratio_max_int_ = buff_config_.param.SMALL_TARGET_ASPECT_RATIO_MAX * 10;
  small_target_aspect_ratio_min_int_ = buff_config_.param.SMALL_TARGET_ASPECT_RATIO_MIN * 10;
  area_ratio_max_int_                = buff_config_.param.AREA_RATIO_MAX * 100;
  area_ratio_min_int_                = buff_config_.param.AREA_RATIO_MIN * 100;
#endif  // !RELEASE

  /* 判断目标是否为空 */
  is_find_target_ = false;

  /* 查找圆心 */

  /* 计算运转状态值：速度、方向、角度 */
  current_angle_ = 0.f;
  last_angle_    = 0.f;
  diff_angle_    = 0.f;

  current_direction_ = 0.f;
  find_cnt_          = 0;
  d_angle_           = 1.f;
  confirm_cnt_       = 0;
  is_confirm_        = false;

  current_speed_   = 0.f;
  last_time_       = 0.0;
  last_diff_angle_ = 0.f;

  /* 计算预测量 */
  barrel_buff_botton_h_ = (buff_config_.param.BUFF_H - buff_config_.param.BUFF_RADIUS) -
                          (buff_config_.param.PLATFORM_H + buff_config_.param.BARREL_ROBOT_H);
  current_radian_          = 0.f;
  target_buff_h_           = 0.f;
  target_y_                = 0.f;
  target_x_                = 0.f;
  target_z_                = 0.f;
  offset_angle_int_        = 0;
  offset_angle_float_      = 0.f;
  bullet_tof_              = 0.f;
  fixed_forecast_quantity_ = 0.f;
  final_forecast_quantity_ = 0.f;

  /* 计算获取最终目标（矩形、顶点） */
  theta_       = 0.0;
  final_angle_ = 0.f;
  sin_calcu_   = 0.f;
  cos_calcu_   = 0.f;
  pre_center_  = Point2f(0.f, 0.f);
  radio_       = 0.f;

  /* 计算云台角度 */
  /* 输入串口数据 */
}

RM_Buff::~RM_Buff() {}

void RM_Buff::runTask(Mat& _input_img, Receive_Info& _receive_info, Send_Info& _send_info)
{
  /* 获取基本信息 */
  Input(_input_img, _receive_info.my_color);

  /* 预处理 */
  imageProcessing(this->src_img_, this->bin_img_, this->my_color_, BGR_MODE);

  /* 查找目标 */
  findTarget(this->dst_img_, this->bin_img_, this->target_box_);

  /* 判断目标是否为空 */
  this->is_find_target_ = isFindTarget(this->dst_img_, this->target_box_);

  /* 查找圆心 */
  this->final_center_r_ =
      findCircleR(this->src_img_, this->bin_img_, this->dst_img_, this->is_find_target_);

  /* 计算运转状态值：速度、方向、角度 */
  this->judgeCondition(is_find_target_);

  /* 计算预测量 单位为弧度 */
  this->final_forecast_quantity_ =
      Predict(static_cast<float>(_receive_info.bullet_velocity), this->is_find_target_);

  cout << "测试 提前了" << final_forecast_quantity_ * 180 / CV_PI << "度" << endl;

  /* 计算获取最终目标（矩形、顶点） */
  this->calculateTargetPointSet(this->final_forecast_quantity_, this->final_center_r_,
                                this->target_2d_point_, this->dst_img_, this->is_find_target_);

  /* 计算云台角度 */
  if (is_find_target_) {
    /* 计算云台角度 */
    this->buff_pnp_.run_Solvepnp(_receive_info.bullet_velocity, 2, this->dst_img_,
                                 this->target_2d_point_, target_z_);
    _send_info.angle_yaw   = this->buff_pnp_.returnYawAngle();
    _send_info.angle_pitch = buff_pnp_.returnPitchAngle();
    _send_info.depth       = buff_pnp_.returnDepth();
    std::cout << " yaw:" << _send_info.angle_yaw << " pitch:" << _send_info.angle_pitch
              << " depth:" << _send_info.depth << std::endl;
  }
  else {
    _send_info = Send_Info();
  }

  /* TODO:自动控制 */

  this->displayDst();

  /* 更新上一帧数据 */
  this->updateLastData(is_find_target_);

  /* 输入串口数据 */
}

void RM_Buff::runTask(Mat& _input_img, RM_Messenger* _messenger)
{
  /* 获取基本信息 */
  Input(_input_img, _messenger->getReceiveInfo().my_color);

  /* 预处理 */
  imageProcessing(this->src_img_, this->bin_img_, this->my_color_, BGR_MODE);

  /* 查找目标 */
  findTarget(this->dst_img_, this->bin_img_, this->target_box_);

  /* 判断目标是否为空 */
  this->is_find_target_ = isFindTarget(this->dst_img_, this->target_box_);
  _messenger->getSendInfo().is_found_target =
      this->is_find_target_;  // TODO:待改进为内嵌赋值，而不是外漏

  /* 查找圆心 */
  this->final_center_r_ =
      findCircleR(this->src_img_, this->bin_img_, this->dst_img_, this->is_find_target_);

  /* 计算运转状态值：速度、方向、角度 */
  this->judgeCondition(is_find_target_);

  /* 计算预测量 单位为弧度 */
  this->final_forecast_quantity_ = Predict(
      static_cast<float>(_messenger->getReceiveInfo().bullet_velocity), this->is_find_target_);

  cout << "测试 提前了" << final_forecast_quantity_ * 180 / CV_PI << "度" << endl;

  /* 计算获取最终目标（矩形、顶点） */
  this->calculateTargetPointSet(this->final_forecast_quantity_, this->final_center_r_,
                                this->target_2d_point_, this->dst_img_, this->is_find_target_);

  if (is_find_target_) {
    /* 计算云台角度 */

    this->buff_pnp_.run_Solvepnp(_messenger->getReceiveInfo().bullet_velocity, 2, this->dst_img_,
                                 this->target_2d_point_, target_z_);
    _messenger->getSendInfo().angle_yaw   = this->buff_pnp_.returnYawAngle();
    _messenger->getSendInfo().angle_pitch = buff_pnp_.returnPitchAngle();
    _messenger->getSendInfo().depth       = buff_pnp_.returnDepth();
    std::cout << " yaw:" << _messenger->getSendInfo().angle_yaw
              << " pitch:" << _messenger->getSendInfo().angle_pitch
              << " depth:" << _messenger->getSendInfo().depth << std::endl;
  }
  else {
    _messenger->setSendLostInfo();
  }
  /* TODO:自动控制 */

  this->displayDst();

  /* 更新上一帧数据 */
  this->updateLastData(is_find_target_);
}

Send_Info RM_Buff::runTask(Mat& _input_img, Receive_Info& _receive_info) {
  Send_Info send_info;

  /* 获取基本信息 */
  Input(_input_img, _receive_info.my_color);

  /* 预处理 */
  imageProcessing(this->src_img_, this->bin_img_, this->my_color_, BGR_MODE);

  /* 查找目标 */
  findTarget(this->dst_img_, this->bin_img_, this->target_box_);

  /* 判断目标是否为空 */
  this->is_find_target_ = isFindTarget(this->dst_img_, this->target_box_);

  /* 查找圆心 */
  this->final_center_r_ =
      findCircleR(this->src_img_, this->bin_img_, this->dst_img_, this->is_find_target_);

  /* 计算运转状态值：速度、方向、角度 */
  this->judgeCondition(is_find_target_);

  /* 计算预测量 单位为弧度 */
  this->final_forecast_quantity_ =
      Predict(static_cast<float>(_receive_info.bullet_velocity), this->is_find_target_);

  cout << "测试 提前了" << final_forecast_quantity_ * 180 / CV_PI << "度" << endl;

  /* 计算获取最终目标（矩形、顶点） */
  this->calculateTargetPointSet(this->final_forecast_quantity_, this->final_center_r_,
                                this->target_2d_point_, this->dst_img_, this->is_find_target_);

  /* 计算云台角度 */
  if (is_find_target_) {
    /* 计算云台角度 */
    this->buff_pnp_.run_Solvepnp(_receive_info.bullet_velocity, 2, this->dst_img_,
                                 this->target_2d_point_, target_z_);
    send_info.angle_yaw   = this->buff_pnp_.returnYawAngle();
    send_info.angle_pitch = buff_pnp_.returnPitchAngle();
    send_info.depth       = buff_pnp_.returnDepth();
    std::cout << " yaw:" << send_info.angle_yaw << " pitch:" << send_info.angle_pitch
              << " depth:" << send_info.depth << std::endl;
  }
  else {
    send_info = Send_Info();
  }

  /* TODO:自动控制 */

  this->displayDst();

  /* 更新上一帧数据 */
  this->updateLastData(is_find_target_);

  return send_info;
}

void RM_Buff::readBuffConfig(const cv::FileStorage& _fs)
{
  // ctrl
  _fs["IS_SHOW_BIN_IMG"] >> this->buff_config_.ctrl.IS_SHOW_BIN_IMG;
  _fs["PROCESSING_MODE"] >> this->buff_config_.ctrl.PROCESSING_MODE;
  _fs["IS_PARAM_ADJUSTMENT"] >> this->buff_config_.ctrl.IS_PARAM_ADJUSTMENT;

  // BGR
  _fs["RED_BUFF_GRAY_TH"] >> this->buff_config_.param.RED_BUFF_GRAY_TH;
  _fs["RED_BUFF_COLOR_TH"] >> this->buff_config_.param.RED_BUFF_COLOR_TH;
  _fs["BLUE_BUFF_GRAY_TH"] >> this->buff_config_.param.BLUE_BUFF_GRAY_TH;
  _fs["BLUE_BUFF_COLOR_TH"] >> this->buff_config_.param.BLUE_BUFF_COLOR_TH;

  // HSV-red
  _fs["H_RED_MAX"] >> this->buff_config_.param.H_RED_MAX;
  _fs["H_RED_MIN"] >> this->buff_config_.param.H_RED_MIN;
  _fs["S_RED_MAX"] >> this->buff_config_.param.S_RED_MAX;
  _fs["S_RED_MIN"] >> this->buff_config_.param.S_RED_MIN;
  _fs["V_RED_MAX"] >> this->buff_config_.param.V_RED_MAX;
  _fs["V_RED_MIN"] >> this->buff_config_.param.V_RED_MIN;

  // HSV-blue
  _fs["H_BLUE_MAX"] >> this->buff_config_.param.H_BLUE_MAX;
  _fs["H_BLUE_MIN"] >> this->buff_config_.param.H_BLUE_MIN;
  _fs["S_BLUE_MAX"] >> this->buff_config_.param.S_BLUE_MAX;
  _fs["S_BLUE_MIN"] >> this->buff_config_.param.S_BLUE_MIN;
  _fs["V_BLUE_MAX"] >> this->buff_config_.param.V_BLUE_MAX;
  _fs["V_BLUE_MIN"] >> this->buff_config_.param.V_BLUE_MIN;

  // area
  _fs["SMALL_TARGET_AREA_MAX"] >> this->buff_config_.param.SMALL_TARGET_AREA_MAX;
  _fs["SMALL_TARGET_AREA_MIN"] >> this->buff_config_.param.SMALL_TARGET_AREA_MIN;
  _fs["BIG_TARGET_AREA_MAX"] >> this->buff_config_.param.BIG_TARGET_AREA_MAX;
  _fs["BIG_TARGET_AREA_MIN"] >> this->buff_config_.param.BIG_TARGET_AREA_MIN;

  // length
  _fs["SMALL_TARGET_Length_MAX"] >> this->buff_config_.param.SMALL_TARGET_Length_MAX;
  _fs["SMALL_TARGET_Length_MIN"] >> this->buff_config_.param.SMALL_TARGET_Length_MIN;
  _fs["BIG_TARGET_Length_MAX"] >> this->buff_config_.param.BIG_TARGET_Length_MAX;
  _fs["BIG_TARGET_Length_MIN"] >> this->buff_config_.param.BIG_TARGET_Length_MIN;

  // diff_angle
  _fs["DIFF_ANGLE_MAX"] >> this->buff_config_.param.DIFF_ANGLE_MAX;
  _fs["DIFF_ANGLE_MIN"] >> this->buff_config_.param.DIFF_ANGLE_MIN;

  // aspect_ratio
  _fs["SMALL_TARGET_ASPECT_RATIO_MAX"] >> this->buff_config_.param.SMALL_TARGET_ASPECT_RATIO_MAX;
  _fs["SMALL_TARGET_ASPECT_RATIO_MIN"] >> this->buff_config_.param.SMALL_TARGET_ASPECT_RATIO_MIN;

  // area_ratio
  _fs["AREA_RATIO_MAX"] >> this->buff_config_.param.AREA_RATIO_MAX;
  _fs["AREA_RATIO_MIN"] >> this->buff_config_.param.AREA_RATIO_MIN;

  _fs["BIG_LENTH_R"] >> this->buff_config_.param.BIG_LENTH_R;

  _fs["CENTER_R_ROI_SIZE"] >> this->buff_config_.param.CENTER_R_ROI_SIZE;
  //
  // 能量机关打击模型参数
  _fs["BUFF_H"] >> this->buff_config_.param.BUFF_H;
  _fs["BUFF_RADIUS"] >> this->buff_config_.param.BUFF_RADIUS;
  _fs["PLATFORM_H"] >> this->buff_config_.param.PLATFORM_H;
  _fs["BARREL_ROBOT_H"] >> this->buff_config_.param.BARREL_ROBOT_H;
  _fs["TARGET_X"] >> this->buff_config_.param.TARGET_X;

  // 输出提示
  std::cout
      << "✔️ ✔️ ✔️ 🌈 能量机关初始化参数 读取成功 🌈 ✔️ ✔️ "
         "✔️"
      << std::endl;
}

void RM_Buff::imageProcessing(Mat&                     _input_img,
                              Mat&                     _output_img,
                              const int&               _my_color,
                              const Processing_Moudle& _process_moudle)
{
  //  更新灰度图
  cvtColor(_input_img, this->gray_img_, COLOR_BGR2GRAY);

  // 选择预处理的模式：BGR、HSV
  // switch (this->buff_config_.ctrl.PROCESSING_MODE) {
  switch (_process_moudle) {
    case BGR_MODE: {
      cout << "+++ BGR MODOL +++" << endl;

      this->bgrProcessing(_my_color);

      break;
    }
    case HSV_MODE: {
      cout << "--- HSV MODOL ---" << endl;

      this->hsvProcessing(_my_color);

      break;
    }
    default: {
      cout << "=== DEFAULT MODOL ===" << endl;

      this->bgrProcessing(_my_color);

      break;
    }
  }

// 显示各部分的二值图
#ifndef RELEASE
  if (buff_config_.ctrl.IS_SHOW_BIN_IMG == 1 && buff_config_.ctrl.IS_PARAM_ADJUSTMENT == 1) {
    imshow("bin_img_color", bin_img_color_);
    imshow("bin_img_gray", bin_img_gray_);
  }
#endif  // !RELEASE

  // 求交集
  bitwise_and(bin_img_color_, bin_img_gray_, bin_img_);

  // 膨胀处理
  morphologyEx(bin_img_, bin_img_, MORPH_DILATE, this->ele_);

// 显示最终合并的二值图
#ifndef RELEASE
  if (buff_config_.ctrl.IS_SHOW_BIN_IMG == 1 && buff_config_.ctrl.IS_PARAM_ADJUSTMENT == 1) {
    imshow("bin_img_final", bin_img_);
  }
#endif  // !RELEASE
}

void RM_Buff::bgrProcessing(const int& _my_color)
{
  // 分离通道
  split(this->src_img_, this->split_img_);

  // 选择颜色
  switch (_my_color) {
    case RED: {
      cout << "My color is red!" << endl;

      /* my_color为红色，则处理红色的情况 */
      /* 灰度图与RGB同样做红色处理 */
      subtract(split_img_[2], split_img_[0], bin_img_color_);  // r-b

#ifndef RELEASE
      if (this->buff_config_.ctrl.IS_PARAM_ADJUSTMENT == 1) {
        namedWindow("trackbar");
        createTrackbar("GRAY_TH_RED:", "trackbar", &this->buff_config_.param.RED_BUFF_GRAY_TH, 255,
                       nullptr);
        createTrackbar("COLOR_TH_RED:", "trackbar", &this->buff_config_.param.RED_BUFF_COLOR_TH,
                       255, nullptr);
        imshow("trackbar", trackbar_img_);
        cout << "🧐 BGR红色预处理调参面板已打开 🧐" << endl;
      }

#endif  // !RELEASE

      // 亮度部分
      threshold(this->gray_img_, this->bin_img_gray_, this->buff_config_.param.RED_BUFF_GRAY_TH,
                255, THRESH_BINARY);

      // 颜色部分
      threshold(this->bin_img_color_, this->bin_img_color_,
                this->buff_config_.param.RED_BUFF_COLOR_TH, 255, THRESH_BINARY);

      break;
    }
    case BLUE: {
      cout << "My color is blue!" << endl;

      /* my_color为蓝色，则处理蓝色的情况 */
      /* 灰度图与RGB同样做蓝色处理 */
      subtract(split_img_[0], split_img_[2], bin_img_color_);  // b-r

#ifndef RELEASE
      if (this->buff_config_.ctrl.IS_PARAM_ADJUSTMENT == 1) {
        namedWindow("trackbar");
        createTrackbar("GRAY_TH_BLUE:", "trackbar", &this->buff_config_.param.BLUE_BUFF_GRAY_TH,
                       255, nullptr);
        createTrackbar("COLOR_TH_BLUE:", "trackbar", &this->buff_config_.param.BLUE_BUFF_COLOR_TH,
                       255, nullptr);
        imshow("trackbar", trackbar_img_);
        cout << "🧐 BGR蓝色预处理调参面板已打开 🧐" << endl;
      }
#endif  // !RELEASE

      // 亮度部分
      threshold(this->gray_img_, this->bin_img_gray_, this->buff_config_.param.BLUE_BUFF_GRAY_TH,
                255, THRESH_BINARY);

      // 颜色部分
      threshold(this->bin_img_color_, this->bin_img_color_,
                this->buff_config_.param.BLUE_BUFF_COLOR_TH, 255, THRESH_BINARY);

      break;
    }
    default: {
      cout << "My color is default!" << endl;

      subtract(this->split_img_[0], this->split_img_[2], bin_img_color1_);  // b-r
      subtract(this->split_img_[2], this->split_img_[0], bin_img_color2_);  // r-b

#ifndef RELEASE
      if (this->buff_config_.ctrl.IS_PARAM_ADJUSTMENT == 1) {
        namedWindow("trackbar");
        createTrackbar("GRAY_TH_RED:", "trackbar", &this->buff_config_.param.RED_BUFF_GRAY_TH, 255,
                       nullptr);
        createTrackbar("COLOR_TH_RED:", "trackbar", &this->buff_config_.param.RED_BUFF_COLOR_TH,
                       255, nullptr);
        createTrackbar("GRAY_TH_BLUE:", "trackbar", &this->buff_config_.param.BLUE_BUFF_GRAY_TH,
                       255, nullptr);
        createTrackbar("COLOR_TH_BLUE:", "trackbar", &this->buff_config_.param.BLUE_BUFF_COLOR_TH,
                       255, nullptr);
        imshow("trackbar", trackbar_img_);
        cout << "🧐 BGR通用预处理调参面板已打开 🧐" << endl;
      }
#endif  // !RELEASE

      // 亮度部分
      this->average_th_ = static_cast<int>(
          (this->buff_config_.param.RED_BUFF_GRAY_TH + this->buff_config_.param.BLUE_BUFF_GRAY_TH) *
          0.5);
      threshold(this->gray_img_, this->bin_img_gray_, average_th_, 255, THRESH_BINARY);

      // 颜色部分
      threshold(this->bin_img_color1_, this->bin_img_color1_,
                this->buff_config_.param.BLUE_BUFF_COLOR_TH, 255, THRESH_BINARY);
      threshold(this->bin_img_color2_, this->bin_img_color2_,
                this->buff_config_.param.RED_BUFF_COLOR_TH, 255, THRESH_BINARY);

      // 求并集
      bitwise_or(bin_img_color1_, bin_img_color2_, bin_img_color_);

      break;
    }
  }

  split_img_.clear();
  vector<Mat>(split_img_).swap(split_img_);  // TODO:查看容量有多大
}

void RM_Buff::hsvProcessing(const int& _my_color)
{
  cvtColor(this->src_img_, this->hsv_img_, COLOR_BGR2HSV_FULL);

  switch (_my_color) {
    case RED:

      cout << "My color is red!" << endl;
#ifndef RELEASE
      if (this->buff_config_.ctrl.IS_PARAM_ADJUSTMENT == 1) {
        namedWindow("trackbar");
        createTrackbar("GRAY_TH_RED:", "trackbar", &this->buff_config_.param.RED_BUFF_GRAY_TH, 255,
                       nullptr);
        createTrackbar("H_RED_MAX:", "trackbar", &this->buff_config_.param.H_RED_MAX, 360, nullptr);
        createTrackbar("H_RED_MIN:", "trackbar", &this->buff_config_.param.H_RED_MIN, 360, nullptr);
        createTrackbar("S_RED_MAX:", "trackbar", &this->buff_config_.param.S_RED_MAX, 255, nullptr);
        createTrackbar("S_RED_MIN:", "trackbar", &this->buff_config_.param.S_RED_MIN, 255, nullptr);
        createTrackbar("V_RED_MAX:", "trackbar", &this->buff_config_.param.V_RED_MAX, 255, nullptr);
        createTrackbar("V_RED_MIN:", "trackbar", &this->buff_config_.param.V_RED_MIN, 255, nullptr);
        imshow("trackbar", trackbar_img_);
        cout << "🧐 HSV红色预处理调参面板已打开 🧐" << endl;
      }
#endif  // !RELEASE

      // 颜色部分

      inRange(this->hsv_img_,
              Scalar(this->buff_config_.param.H_RED_MIN, this->buff_config_.param.S_RED_MIN,
                     this->buff_config_.param.V_RED_MIN),
              Scalar(this->buff_config_.param.H_RED_MAX, this->buff_config_.param.S_RED_MAX,
                     this->buff_config_.param.V_RED_MAX),
              bin_img_color_);

      // 亮度部分
      threshold(this->gray_img_, this->bin_img_gray_, this->buff_config_.param.RED_BUFF_GRAY_TH,
                255, THRESH_BINARY);

      break;
    case BLUE:

      cout << "My color is blue!" << endl;

#ifndef RELEASE
      if (this->buff_config_.ctrl.IS_PARAM_ADJUSTMENT == 1) {
        namedWindow("trackbar");
        createTrackbar("GRAY_TH_BLUE:", "trackbar", &this->buff_config_.param.BLUE_BUFF_GRAY_TH,
                       255, nullptr);
        createTrackbar("H_BLUE_MAX:", "trackbar", &this->buff_config_.param.H_BLUE_MAX, 255,
                       nullptr);
        createTrackbar("H_BLUE_MIN:", "trackbar", &this->buff_config_.param.H_BLUE_MIN, 255,
                       nullptr);
        createTrackbar("S_BLUE_MAX:", "trackbar", &this->buff_config_.param.S_BLUE_MAX, 255,
                       nullptr);
        createTrackbar("S_BLUE_MIN:", "trackbar", &this->buff_config_.param.S_BLUE_MIN, 255,
                       nullptr);
        createTrackbar("V_BLUE_MAX:", "trackbar", &this->buff_config_.param.V_BLUE_MAX, 255,
                       nullptr);
        createTrackbar("V_BLUE_MIN:", "trackbar", &this->buff_config_.param.V_BLUE_MIN, 255,
                       nullptr);
        imshow("trackbar", trackbar_img_);
        cout << "🧐 HSV蓝色预处理调参面板已打开 🧐" << endl;
      }
#endif  // !RELEASE

      // 颜色部分
      inRange(this->hsv_img_,
              Scalar(this->buff_config_.param.H_BLUE_MIN, this->buff_config_.param.S_BLUE_MIN,
                     this->buff_config_.param.V_BLUE_MIN),
              Scalar(this->buff_config_.param.H_BLUE_MAX, this->buff_config_.param.S_BLUE_MAX,
                     this->buff_config_.param.V_BLUE_MAX),
              bin_img_color_);

      // 亮度部分
      threshold(this->gray_img_, this->bin_img_gray_, this->buff_config_.param.BLUE_BUFF_GRAY_TH,
                255, THRESH_BINARY);

      break;
    default:

      cout << "My color is default!" << endl;

#ifndef RELEASE
      if (this->buff_config_.ctrl.IS_PARAM_ADJUSTMENT == 1) {
        namedWindow("trackbar");

        createTrackbar("GRAY_TH_RED:", "trackbar", &this->buff_config_.param.RED_BUFF_GRAY_TH, 255,
                       nullptr);
        createTrackbar("H_RED_MAX:", "trackbar", &this->buff_config_.param.H_RED_MAX, 360, nullptr);
        createTrackbar("H_RED_MIN:", "trackbar", &this->buff_config_.param.H_RED_MIN, 360, nullptr);
        createTrackbar("S_RED_MAX:", "trackbar", &this->buff_config_.param.S_RED_MAX, 255, nullptr);
        createTrackbar("S_RED_MIN:", "trackbar", &this->buff_config_.param.S_RED_MIN, 255, nullptr);
        createTrackbar("V_RED_MAX:", "trackbar", &this->buff_config_.param.V_RED_MAX, 255, nullptr);
        createTrackbar("V_RED_MIN:", "trackbar", &this->buff_config_.param.V_RED_MIN, 255, nullptr);

        createTrackbar("GRAY_TH_BLUE:", "trackbar", &this->buff_config_.param.BLUE_BUFF_GRAY_TH,
                       255, nullptr);
        createTrackbar("H_BLUE_MAX:", "trackbar", &this->buff_config_.param.H_BLUE_MAX, 255,
                       nullptr);
        createTrackbar("H_BLUE_MIN:", "trackbar", &this->buff_config_.param.H_BLUE_MIN, 255,
                       nullptr);
        createTrackbar("S_BLUE_MAX:", "trackbar", &this->buff_config_.param.S_BLUE_MAX, 255,
                       nullptr);
        createTrackbar("S_BLUE_MIN:", "trackbar", &this->buff_config_.param.S_BLUE_MIN, 255,
                       nullptr);
        createTrackbar("V_BLUE_MAX:", "trackbar", &this->buff_config_.param.V_BLUE_MAX, 255,
                       nullptr);
        createTrackbar("V_BLUE_MIN:", "trackbar", &this->buff_config_.param.V_BLUE_MIN, 255,
                       nullptr);
        imshow("trackbar", trackbar_img_);
        cout << "🧐 HSV通用预处理调参面板已打开 🧐" << endl;
      }
#endif  // !RELEASE

      // 亮度部分
      this->average_th_ = static_cast<int>(
          (this->buff_config_.param.RED_BUFF_GRAY_TH + this->buff_config_.param.BLUE_BUFF_GRAY_TH) *
          0.5);
      threshold(this->gray_img_, this->bin_img_gray_, average_th_, 255, THRESH_BINARY);

      // 红色
      inRange(this->hsv_img_,
              Scalar(this->buff_config_.param.H_RED_MIN, this->buff_config_.param.S_RED_MIN,
                     this->buff_config_.param.V_RED_MIN),
              Scalar(this->buff_config_.param.H_RED_MAX, this->buff_config_.param.S_RED_MAX,
                     this->buff_config_.param.V_RED_MAX),
              bin_img_color2_);
      // 蓝色
      inRange(this->hsv_img_,
              Scalar(this->buff_config_.param.H_BLUE_MIN, this->buff_config_.param.S_BLUE_MIN,
                     this->buff_config_.param.V_BLUE_MIN),
              Scalar(this->buff_config_.param.H_BLUE_MAX, this->buff_config_.param.S_BLUE_MAX,
                     this->buff_config_.param.V_BLUE_MAX),
              bin_img_color1_);

      // 求并集
      bitwise_or(bin_img_color1_, bin_img_color2_, bin_img_color_);
      break;
  }
}

void RM_Buff::findTarget(Mat& _input_dst_img, Mat& _input_bin_img, vector<Target>& _target_box)
{
  findContours(_input_bin_img, contours_, hierarchy_, 2, CHAIN_APPROX_NONE);

  for (size_t i = 0; i < contours_.size(); ++i) {
    // 用于寻找小轮廓，没有父轮廓的跳过，以及不满足6点拟合椭圆
    if (hierarchy_[i][3] < 0 || contours_[i].size() < 6 ||
        contours_[static_cast<uint>(hierarchy_[i][3])].size() < 6) {
      continue;
    }

    // 小轮廓周长条件
    this->small_rect_length_ = arcLength(contours_[i], true);
    if (small_rect_length_ < this->buff_config_.param.SMALL_TARGET_Length_MIN) {
      continue;
    }

    // 小轮廓面积条件
    this->small_rect_area_ = contourArea(contours_[i]);
    if (this->small_rect_area_ < this->buff_config_.param.SMALL_TARGET_AREA_MIN ||
        this->small_rect_area_ > this->buff_config_.param.SMALL_TARGET_AREA_MAX) {
      continue;
    }

    // 大轮廓周长条件
    this->big_rect_length_ = arcLength(contours_[static_cast<uint>(hierarchy_[i][3])], true);
    if (this->big_rect_length_ < this->buff_config_.param.BIG_TARGET_Length_MIN) {
      continue;
    }

    // 大轮廓面积条件
    this->big_rect_area_ = contourArea(contours_[static_cast<uint>(hierarchy_[i][3])]);
    if (this->big_rect_area_ < this->buff_config_.param.BIG_TARGET_AREA_MIN ||
        this->big_rect_area_ > this->buff_config_.param.BIG_TARGET_AREA_MAX) {
      continue;
    }

    // 保存扇叶和装甲板的数据
    small_target_.inputParams(contours_[i]);
    big_target_.inputParams(contours_[static_cast<uint>(hierarchy_[i][3])]);
    candidated_target_.inputParams(big_target_, small_target_);

    // 组合判断角度差
    if (this->candidated_target_.diffAngle() >= this->buff_config_.param.DIFF_ANGLE_MAX ||
        this->candidated_target_.diffAngle() <= this->buff_config_.param.DIFF_ANGLE_MIN) {
      continue;
    }

    // 判断内轮廓的长宽比是否正常
    if (this->candidated_target_.Armor().aspectRatio() >=
            this->buff_config_.param.SMALL_TARGET_ASPECT_RATIO_MAX ||
        this->candidated_target_.Armor().aspectRatio() <=
            this->buff_config_.param.SMALL_TARGET_ASPECT_RATIO_MIN) {
      continue;
    }

    // 判断内外轮廓的面积比是否正常
    if (this->candidated_target_.areaRatio() <= this->buff_config_.param.AREA_RATIO_MIN ||
        this->candidated_target_.areaRatio() >= this->buff_config_.param.AREA_RATIO_MAX) {
      continue;
    }

    // #ifndef RELEASE
    this->small_target_.displayFanArmor(_input_dst_img);
    this->big_target_.displayFanBlade(_input_dst_img);
    // #endif  // !RELEASE

    // 设置默认扇叶状态
    this->candidated_target_.setType(ACTION);
    // 更新装甲板的四个顶点编号
    this->candidated_target_.updateVertex(_input_dst_img);
    // 更新扇叶状态
    this->candidated_target_.setType(_input_bin_img, _input_dst_img);

    _target_box.emplace_back(candidated_target_);
  }
  cout << "扇叶数量：" << _target_box.size() << endl;
}

bool RM_Buff::isFindTarget(Mat& _input_img, vector<Target>& _target_box)
{
  if (_target_box.size() < 1) {
    cout << "XXX 没有目标 XXX" << endl;

    current_target_ = Target();
    this->contours_.clear();
    this->hierarchy_.clear();
    _target_box.clear();

    vector<vector<Point>>(contours_).swap(contours_);
    vector<Vec4i>(hierarchy_).swap(hierarchy_);
    vector<Target>(_target_box).swap(_target_box);

    return false;
  }

  this->inaction_cnt_ = 0;

  // 遍历容器获取未激活目标
  for (auto iter = _target_box.begin(); iter != _target_box.end(); ++iter) {
    if ((*iter).Type() != INACTION) {
      // TODO 测试是否会多或者少了几个对象，如果没有顺序的话，有可能第一个就退出了
      ++this->inaction_cnt_;
      continue;
    }

    // 获取到未激活对象后退出遍历 TODO 查看是否筛选出想要的内容
    this->current_target_ = *iter;
    this->current_target_.displayInactionTarget(_input_img);
  }

  // 清除容器
  this->contours_.clear();
  this->hierarchy_.clear();
  _target_box.clear();

  vector<vector<Point>>(contours_).swap(contours_);
  vector<Vec4i>(hierarchy_).swap(hierarchy_);
  vector<Target>(_target_box).swap(_target_box);

  return true;
}

Point2f RM_Buff::findCircleR(Mat&        _input_src_img,
                             Mat&        _input_bin_img,
                             Mat&        _dst_img,
                             const bool& _is_find_target)
{
  // 更新图像
  _input_src_img.copyTo(this->roi_img_);
  _input_bin_img.copyTo(this->result_img_);

  Point2f center_r_point2f;
  center_r_point2f = Point2f(0.f, 0.f);

  // 若没有扇叶目标则提前退出
  if (!_is_find_target) {
    this->is_circle_  = false;
    roi_local_center_ = Point2f(0.f, 0.f);

    // 清理容器
    this->center_r_box_.clear();
    vector<Center_R>(center_r_box_).swap(center_r_box_);
    return center_r_point2f;
  }

  // 计算圆心大概位置
  this->delta_height_point_ = this->current_target_.deltaPoint();
  this->roi_global_center_  = this->current_target_.Armor().Rect().center -
                             this->buff_config_.param.BIG_LENTH_R * this->delta_height_point_;

  // roi中心安全条件
  if (this->roi_global_center_.x < 0 || this->roi_global_center_.y < 0 ||
      this->roi_global_center_.x > _input_src_img.cols ||
      this->roi_global_center_.y > _input_src_img.rows) {
    if (this->roi_global_center_.x < 0) {
      roi_global_center_.x = 1;
    }
    if (this->roi_global_center_.y < 0) {
      roi_global_center_.y = 1;
    }
    if (this->roi_global_center_.x > _input_src_img.cols) {
      roi_global_center_.x = _input_src_img.cols - 1;
    }
    if (this->roi_global_center_.y > _input_src_img.rows - 1) {
      roi_global_center_.y = _input_src_img.rows - 1;
    }
  }

  // 画出假定圆心的roi矩形
  RotatedRect roi_R(
      roi_global_center_,
      Size(this->buff_config_.param.CENTER_R_ROI_SIZE, this->buff_config_.param.CENTER_R_ROI_SIZE),
      0);
  cv::Rect roi = roi_R.boundingRect();

  // roi安全条件
  roi = roi_tool_.makeRectSafe_Tailor(_input_src_img, roi);

  // 截取roi大小的图像，并绘制截取区域
  this->result_img_ = roi_tool_.cutRoi_Rect(_input_bin_img, roi);
  this->roi_img_    = roi_tool_.cutRoi_Rect(_input_src_img, roi);
  rectangle(_dst_img, roi, Scalar(0, 255, 200), 2, 8, 0);

  this->is_circle_ = false;

  // 更新roi的中心点
  this->roi_local_center_ = Point2f(roi_img_.cols * 0.5, roi_img_.rows * 0.5);

  // 查找轮廓
  findContours(result_img_, contours_r_, RETR_EXTERNAL, CHAIN_APPROX_NONE);

  cout << "遍历轮廓数量：" << this->contours_r_.size() << endl;

  // 选择并记录合适的圆心目标
  for (size_t j = 0; j < this->contours_r_.size(); ++j) {
    if (this->contours_r_[j].size() < 6) {
      continue;
    }

    this->center_r_.inputParams(this->contours_r_[j], this->roi_img_);

    cout << "矩形比例：" << this->center_r_.aspectRatio() << endl;
    if (this->center_r_.aspectRatio() < 0.9f || this->center_r_.aspectRatio() > 1.25f) {
      continue;
    }

    cout << "矩形面积：" << this->center_r_.Rect().boundingRect().area() << endl;
    if (this->center_r_.Rect().boundingRect().area() < 1000 ||
        this->center_r_.Rect().boundingRect().area() > 3500) {
      continue;
    }

    this->center_r_box_.emplace_back(center_r_);

    for (int k = 0; k < 4; ++k) {
      line(this->roi_img_, center_r_.Vertex(k), center_r_.Vertex((k + 1) % 4), Scalar(0, 130, 255),
           3);
    }

    cout << "正确的矩形比例：" << this->center_r_.aspectRatio() << endl;
  }

  cout << "符合比例条件的：" << this->center_r_box_.size() << endl;

  // 如果没有圆心目标，则退出
  if (this->center_r_box_.size() < 1) {
    cout << "圆心为：假定圆心" << endl;
    this->is_circle_ = false;
    center_r_point2f = this->roi_global_center_;

#ifndef RELEASE
    // 画出小轮廓到假定圆心的距离线
    line(_dst_img, this->current_target_.Armor().Rect().center, center_r_point2f, Scalar(0, 0, 255),
         2);
    // 画出假定圆心
    circle(_dst_img, center_r_point2f, 2, Scalar(0, 0, 255), 2, 8, 0);
#endif  // !RELEASE
  }
  else {
    sort(this->center_r_box_.begin(), this->center_r_box_.end(),
         [](Center_R& c1, Center_R& c2) { return c1.centerDist() < c2.centerDist(); });

    cout << "圆心为：真实圆心" << endl;
    this->is_circle_ = true;
    center_r_point2f = this->center_r_box_[0].Rect().center + roi_R.boundingRect2f().tl();

#ifndef RELEASE
    // 画出小轮廓到假定圆心的距离线
    line(_dst_img, this->current_target_.Armor().Rect().center, center_r_point2f, Scalar(0, 255, 0),
         2);
    // 画出假定圆心
    circle(_dst_img, center_r_point2f, 2, Scalar(0, 0, 255), 2, 8, 0);
#endif  // !RELEASE
  }

  // 清理容器
  this->center_r_box_.clear();
  this->contours_r_.clear();

  vector<Center_R>(center_r_box_).swap(center_r_box_);
  vector<vector<Point>>(contours_r_).swap(contours_r_);

  return center_r_point2f;
}

void RM_Buff::judgeCondition(const bool& _is_find_target)
{
  if (!_is_find_target) {
    // 没有目标，角度为上一帧的角度，方向重置为零，速度为0
    current_angle_ = last_target_.Angle();
    diff_angle_    = 0.f;
    // current_direction_ = 0.f;
    current_speed_ = 0.f;

    return;
  }

  // 计算角度
  Angle();

  // 计算方向
  Direction();

  // 计算速度
  Velocity();

  return;
}

void RM_Buff::Angle()
{
  // 装甲板到圆心的连线所代表的角度
  current_angle_ = atan2((current_target_.Armor().Rect().center.y - final_center_r_.y),
                         (current_target_.Armor().Rect().center.x - final_center_r_.x)) *
                   180 / static_cast<float>(CV_PI);

  // 过零处理
  if (current_angle_ < 0.f) {
    current_angle_ += 360.f;
  }

  // 角度差
  diff_angle_ = current_angle_ - last_angle_;
  cout << "测试 当前角度差为：" << diff_angle_ << endl;

  // 过零处理
  if (diff_angle_ > 180) {
    diff_angle_ -= 360;
  }
  else if (diff_angle_ < -180) {
    diff_angle_ += 360;
  }

  // TODO:当变化量大于30°时，则是切换装甲板，则重置diff为0，last为当前。
}

void RM_Buff::Direction()
{
  ++find_cnt_;
  if (find_cnt_ % 2 == 0) {
    // 隔帧读取数据
    if (current_direction_ != 0 && confirm_cnt_ < 10) {
      confirm_cnt_ += 1;
    }

    if (confirm_cnt_ > 5) {
      is_confirm_ = true;
    }

    // 超过一定标志位循环出现后，强行控制运转方向
    // TODO：待优化为慢速的时候都能检测转动，而不是不动和强行给命令
    if (!is_confirm_) {
      current_direction_ = getState();
    }

    if (find_cnt_ == 10) {
      find_cnt_ = 0;
    }
  }

  // 显示当前转动信息
  if (current_direction_ == 0) {
    cout << "转动方向：不转动" << endl;
  }
  else if (current_direction_ == 1) {
    cout << "转动方向：顺时针转动" << endl;
  }
  else if (current_direction_ == -1) {
    cout << "转动方向：逆时针转动" << endl;
  }
}

int RM_Buff::getState()
{
  if (fabs(diff_angle_) < 10 && fabs(diff_angle_) > 1e-6) {
    d_angle_ = (1 - 0.1) * d_angle_ + 0.1 * this->diff_angle_;
  }

  if (d_angle_ > 1.5) {
    return 1;
  }
  else if (d_angle_ < -1.5) {
    return -1;
  }
  else {
    return 0;
  }
}

void RM_Buff::Velocity()
{
  double current_time       = buff_fps_.lastTime();
  float  current_diff_angle = diff_angle_;

  if (find_cnt_ % 2 == 0) {
    // 隔帧读取数据
    current_time += last_time_;
    current_diff_angle += last_diff_angle_;
    // 默认单位为角度/s
    if (current_time == 0) {
      this->current_speed_ = 0.f;
    }
    else {
      // 将单位转为rad/s
      this->current_speed_ = current_diff_angle / current_time * CV_PI / 180;
    }
  }
  else {
    last_time_       = current_time;
    last_diff_angle_ = diff_angle_;
  }

  cout << "测试 当前风车转速为：" << current_speed_ << endl;
}

float RM_Buff::Predict(const float& _bullet_velocity, const bool& _is_find_target)
{
  // 判断是否发现目标，没有返回0，有则进行计算预测
  if (!_is_find_target) {
    target_z_ = 0.f;
    // 重置参数
    return 0.f;
  }

  float predict_quantity;

  // 计算固定预测量 原来是给0.35弧度 TODO:测一下最快和最慢速度时的提前量，以确定范围
  // predict_quantity = fixedPredict(_bullet_velocity*1000);
  predict_quantity = fixedPredict(29 * 1000);  // 默认先给30m/s

  // 计算移动预测量 TODO

  return predict_quantity;
}

float RM_Buff::fixedPredict(const float& _bullet_velocity)
{
  // 转换为弧度
  current_radian_ = current_angle_ * CV_PI / 180;

  // 计算能量机关的高度
  target_buff_h_ = 800 + sin(current_radian_ - CV_PI) * 800;

  target_y_ = target_buff_h_ + barrel_buff_botton_h_;

  // 计算能量机关的水平直线距离
  target_x_ = buff_config_.param.TARGET_X;

  // 计算能量机关目标的直线距离
  target_z_ = sqrt((target_y_ * target_y_) + (target_x_ * target_x_));

  // 计算子弹飞行时间
  bullet_tof_ = (target_z_ + offset_angle_float_) / _bullet_velocity;

  // 计算固定提前量（也可以直接给定）
  fixed_forecast_quantity_ = current_speed_ * bullet_tof_;

  return fixed_forecast_quantity_;
}

void RM_Buff::mutativePredict(const float& _input_predict_quantity, float& _output_predict_quantity)
{
}

void RM_Buff::calculateTargetPointSet(const float&     _predict_quantity,
                                      const Point2f&   _final_center_r,
                                      vector<Point2f>& _target_2d_point,
                                      Mat&             _input_dst_img,
                                      const bool&      _is_find_target)
{
  // 判断有无目标，若无则重置参数并提前退出
  if (!_is_find_target) {
    _target_2d_point.clear();
    _target_2d_point = vector<Point2f>(4, Point2f(0.f, 0.f));
    // 重置参数
    return;
  }

  // 计算theta
  theta_ =
      atan2(static_cast<double>(current_target_.Armor().Rect().center.y - roi_local_center_.y),
            static_cast<double>(current_target_.Armor().Rect().center.x - roi_local_center_.x));
  if (theta_ < 0) {
    theta_ += (2 * CV_PI);
  }

  // 计算最终角度和弧度
  final_radian_ = theta_ + current_direction_ * _predict_quantity;
  final_angle_  = final_radian_ * 180 / CV_PI;

  // 计算sin和cos
  sin_calcu_ = sin(final_radian_);
  cos_calcu_ = cos(final_radian_);

  // 计算最终坐标点
  pre_center_.x = (current_target_.Armor().Rect().center.x - _final_center_r.x) * cos_calcu_ -
                  (current_target_.Armor().Rect().center.y - _final_center_r.y) * sin_calcu_ +
                  _final_center_r.x;
  pre_center_.y = (current_target_.Armor().Rect().center.x - _final_center_r.x) * sin_calcu_ +
                  (current_target_.Armor().Rect().center.y - _final_center_r.y) * cos_calcu_ +
                  _final_center_r.y;

  // 计算最终目标的旋转矩形
  target_rect_ = RotatedRect(pre_center_, current_target_.Armor().Rect().size, 90);

  // 保存最终目标的顶点，暂时用的是排序点的返序存入才比较稳定，正确使用应为0123
  _target_2d_point.clear();
#ifdef DEBUG
  Point2f target_vertex[4];
  target_rect_.points(target_vertex);
  _target_2d_point.emplace_back(target_vertex[0]);
  _target_2d_point.emplace_back(target_vertex[1]);
  _target_2d_point.emplace_back(target_vertex[2]);
  _target_2d_point.emplace_back(target_vertex[3]);
  // _target_2d_point.emplace_back(current_target_.Armor().Vertex(0));
  // _target_2d_point.emplace_back(current_target_.Armor().Vertex(1));
  // _target_2d_point.emplace_back(current_target_.Armor().Vertex(2));
  // _target_2d_point.emplace_back(current_target_.Armor().Vertex(3));

#else

  _target_2d_point.emplace_back(current_target_.vector2DPoint(0));
  _target_2d_point.emplace_back(current_target_.vector2DPoint(1));
  _target_2d_point.emplace_back(current_target_.vector2DPoint(2));
  _target_2d_point.emplace_back(current_target_.vector2DPoint(3));
#endif  // DEBUG

  // 绘制图像
  radio_ = centerDistance(_final_center_r, pre_center_);

  for (int k = 0; k < 4; ++k) {
    line(_input_dst_img, _target_2d_point[k], _target_2d_point[(k + 1) % 4], Scalar(0, 130, 255),
         8);  // orange
  }

  circle(_input_dst_img, _final_center_r, radio_, Scalar(0, 255, 125), 2, 8, 0);
  circle(_input_dst_img, pre_center_, 3, Scalar(255, 0, 0), 3, 8, 0);
  line(_input_dst_img, pre_center_, _final_center_r, Scalar(0, 255, 255), 2);
  line(_input_dst_img, current_target_.Armor().Rect().center, _final_center_r, Scalar(0, 255, 0),
       2);

  circle(_input_dst_img, _target_2d_point[0], 10, Scalar(0, 0, 255), -1, 8, 0);
  circle(_input_dst_img, _target_2d_point[1], 10, Scalar(0, 255, 255), -1, 8, 0);
  circle(_input_dst_img, _target_2d_point[2], 10, Scalar(255, 0, 0), -1, 8, 0);
  circle(_input_dst_img, _target_2d_point[3], 10, Scalar(0, 255, 0), -1, 8, 0);
}

void RM_Buff::updateLastData(const bool& _is_find_target)

{
  if (!_is_find_target) {
    cout << "没有目标，不需要更新上一帧数据" << endl;
    is_find_last_target_ = _is_find_target;
    target_2d_point_.clear();
    vector<Point2f>(target_2d_point_).swap(target_2d_point_);
    target_rect_ = RotatedRect();
    return;
  }

  this->last_target_   = this->current_target_;
  this->last_angle_    = this->current_angle_;
  is_find_last_target_ = _is_find_target;
  target_2d_point_.clear();
  vector<Point2f>(target_2d_point_).swap(target_2d_point_);
  target_rect_ = RotatedRect();

  cout << "发现目标，已更新上一帧数据" << endl;
}

}  // namespace buff