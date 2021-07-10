#ifndef RM_BUFFDETECTION_H_
#define RM_BUFFDETECTION_H_

#include <algorithm>
#include <string>
#include <vector>

#include "devices/serial/rm_serial_port.hpp"
#include "module/RM_Buff/Center_R/Center_R.hpp"
#include "module/RM_Buff/RM_FPS/RM_FPS.h"
#include "module/RM_Buff/Target/Target.hpp"
#include "module/angle_solve/rm_solve_pnp.hpp"
#include "module/roi/rm_roi.h"

using namespace cv;

// 宏定义开关
#define FPS

#define DEBUG

namespace buff {

// 扇叶状态

struct Buff_Param {
  // BGR
  int RED_BUFF_GRAY_TH;
  int RED_BUFF_COLOR_TH;
  int BLUE_BUFF_GRAY_TH;
  int BLUE_BUFF_COLOR_TH;

  // HSV-red
  int H_RED_MAX;
  int H_RED_MIN;
  int S_RED_MAX;
  int S_RED_MIN;
  int V_RED_MAX;
  int V_RED_MIN;

  // HSV-blue
  int H_BLUE_MAX;
  int H_BLUE_MIN;
  int S_BLUE_MAX;
  int S_BLUE_MIN;
  int V_BLUE_MAX;
  int V_BLUE_MIN;

  // 筛选条件
  // 面积 MAX MIN 大 小
  int SMALL_TARGET_AREA_MAX;
  int SMALL_TARGET_AREA_MIN;
  int BIG_TARGET_AREA_MAX;
  int BIG_TARGET_AREA_MIN;
  // 周长 MAX MIN
  int SMALL_TARGET_Length_MIN;
  int SMALL_TARGET_Length_MAX;
  int BIG_TARGET_Length_MIN;
  int BIG_TARGET_Length_MAX;
  // 角度差 MAX MIN
  int DIFF_ANGLE_MAX;
  int DIFF_ANGLE_MIN;
  // 长宽比 MAX MIN
  float SMALL_TARGET_ASPECT_RATIO_MAX;
  float SMALL_TARGET_ASPECT_RATIO_MIN;
  // 面积比 MAX MIN
  float AREA_RATIO_MAX;
  float AREA_RATIO_MIN;

  // 圆心R距离小轮廓中心的距离系数
  float BIG_LENTH_R;

  /* 圆心限制条件 */
  // 圆心roi矩形大小
  int CENTER_R_ROI_SIZE;
  // 面积

  // 轮廓面积

  // 滤波器系数
  float FILTER_COEFFICIENT;

  // 能量机关打击模型参数，详情见 buff_config.xml
  float BUFF_H;
  float BUFF_RADIUS;
  float PLATFORM_H;
  float BARREL_ROBOT_H;
  float TARGET_X;

  Buff_Param()  // TODO :添加默认参数
  {}
};

struct Buff_Ctrl {
  int IS_PARAM_ADJUSTMENT;
  int IS_SHOW_BIN_IMG;
  int PROCESSING_MODE;
};

struct Buff_Cfg {
  Buff_Param param;
  Buff_Ctrl ctrl;

  Buff_Cfg() {
    // Buff
  }
};

//目标类（基类）

//内轮廓类 继承目标类

// 外轮廓类 继承目标类

// 打击目标类 继承目标类

// 圆心类 继承目标类

// TODO：能量机关类 继承抽象类
class RM_Buff {
 public:
  // 初始化参数结构体
  // RM_Buff() = default;
  RM_Buff(const std::string& _buff_config_address);
  ~RM_Buff();

  // 总执行函数（接口）TODO
  /**
   * @brief 总执行函数（接口）
   * @param  _input_img       输入图像
   * @param  _receive_info    串口接收结构体
   * @param  _send_info       串口发送结构体
   */
  void runTask(Mat& _input_img, serial_port::Receive_Data& _receive_info,
               serial_port::Write_Data& _send_info);

  /**
   * @brief 总执行函数（接口）
   * @param  _input_img       输入图像
   * @param  _receive_info    串口接收结构体
   * @return Send_Info 串口发送结构体
   */
  serial_port::Write_Data runTask(Mat& _input_img,
                                  serial_port::Receive_Data& _receive_info);

 private:
  /**
   * @brief 获取参数更新结构体
   * @param[in]  _fs               文件对象
   */
  void readBuffConfig(const cv::FileStorage& _fs);

  /**
   * @brief 获取基本信息
   * @details 图像和颜色
   * @param[in]  _input_img       输入图像
   * @param[in]  _my_color        己方颜色
   */
  void Input(Mat& _input_img, const int& _my_color);

  /**
   * @brief 显示最终图像
   */
  void displayDst();

  // 类中全局变量
  Mat src_img_;                           // 输入原图
  Mat dst_img_;                           // 图像效果展示图
  std::vector<Point2f> target_2d_point_;  // 目标二维点集
  RotatedRect target_rect_;               // 目标矩形
  int my_color_;                          // 当前自己的颜色
  Buff_Cfg buff_config_;                  // 参数结构体

  bool is_find_last_target_;  // 上一帧是否发现目标 true：发现 false：未发现
  bool is_find_target_;  // 是否发现目标 true：发现 false：未发现

  Target last_target_;  //  上一个打击目标

 private:
  /* 预处理 */
  /**
   * @brief 预处理的模式
   */
  enum Processing_Moudle {
    BGR_MODE,
    HSV_MODE,
  };

  /**
   * @brief 预处理执行函数
   * @param[in]  _input_img       输入图像（src）
   * @param[out] _output_img      输出图像（bin）
   * @param[in]  _my_color        颜色参数
   * @param[in]  _process_moudle  预处理模式
   */
  void imageProcessing(Mat& _input_img, Mat& _output_img, const int& _my_color,
                       const Processing_Moudle& _process_moudle);

  /**
   * @brief BGR颜色空间预处理
   * @param  _my_color        颜色参数
   */
  void bgrProcessing(const int& _my_color);

  /**
   * @brief HSV颜色空间预处理
   * @param  _my_color        颜色参数
   */
  void hsvProcessing(const int& _my_color);

  Mat gray_img_;        // 灰度图
  Mat bin_img_;         // 最终二值图
  Mat bin_img_color_;   // 颜色二值图
  Mat bin_img_color1_;  // 颜色二值图一
  Mat bin_img_color2_;  // 颜色二值图二
  Mat bin_img_gray_;    // 灰度二值图

  Mat ele_ = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));  // 椭圆内核

  // BGR
  std::vector<Mat> split_img_;  //分通道图
  int average_th_;              //平均阈值

  // HSV
  Mat hsv_img_;  // hsv预处理输入图

#ifndef RELEASE
  // 滑动条窗口
  Mat trackbar_img_ = Mat::zeros(1, 300, CV_8UC1);  // 预处理
#endif                                              // !RELEASE

 private:
  /* 查找目标 */
  /**
   * @brief 查找目标
   * @param  _input_dst_img   输入图像
   * @param  _input_bin_img   输入二值图
   * @param  _target_box      输出目标容器
   */
  void findTarget(Mat& _input_dst_img, Mat& _input_bin_img,
                  std::vector<Target>& _target_box);

  // 内轮廓
  FanArmor small_target_;
  // 外轮廓
  FanBlade big_target_;

  // 当前检测得到的打击目标（遍历使用）
  Target candidated_target_;
  // 当前检测打击目标（现在）
  Target current_target_;
  // 打击目标队列
  std::vector<Target> target_box_;
  // 已打击扇叶数
  int action_cnt_;
  // 未击打扇叶数
  int inaction_cnt_;

  // 轮廓点集 轮廓关系
  std::vector<std::vector<Point>> contours_;
  std::vector<Vec4i> hierarchy_;

  // 小轮廓条件(area and length)
  float small_rect_area_;
  float small_rect_length_;

  // 大轮廓条件(area and length)
  float big_rect_area_;
  float big_rect_length_;

#ifndef RELEASE
  // 滑动条窗口
  Mat target_trackbar_img_ = Mat::zeros(1, 300, CV_8UC1);

  int small_target_aspect_ratio_max_int_;
  int small_target_aspect_ratio_min_int_;
  int area_ratio_max_int_;
  int area_ratio_min_int_;
#endif  // !RELEASE

 private:
  /**
   * @brief 判断是否有目标
   * @param[in]  _input_img       绘制未激活的目标
   * @param[in]  _target_box      扇叶目标容器
   * @return true                 发现目标
   * @return false                丢失目标
   */
  bool isFindTarget(Mat& _input_img, std::vector<Target>& _target_box);

 private:
  /**
   * @brief  查找圆心
   * @param  _input_src_img   输入src原图
   * @param  _input_bin_img   输入bin二值图
   * @param  _dst_img         输入dst画板图
   * @param  _is_find_target  是否发现扇叶目标
   * @return Point2f          返回圆心R中点坐标
   */
  Point2f findCircleR(Mat& _input_src_img, Mat& _input_bin_img, Mat& _dst_img,
                      const bool& _is_find_target);

  bool is_circle_;              // 是否找到圆心
  Point2f delta_height_point_;  // 获取装甲板的高度点差
  Point2f roi_global_center_;   // roi 圆心中点位置(在原图中)
  Mat result_img_;              // 二值图
  Mat roi_img_;                 // 截取原图
  roi::ImageRoi roi_tool_;      // roi截取工具

  Center_R center_r_;                           // 候选圆心R
  std::vector<std::vector<Point>> contours_r_;  // 中心R的遍历点集
  Point2f roi_local_center_;  // 截取roi的图像中心点(在roi_img中)
  std::vector<Center_R> center_r_box_;  // 第一次筛选之后得到的待选中心R
  Point2f final_center_r_;              // 最终圆心（假定/真实）

 private:
  /* 计算运转状态值：速度、方向、角度 */
  /**
   * @brief 计算运转状态值：速度、方向、角度
   * @param  _is_find_target  是否发现目标
   */
  void judgeCondition(const bool& _is_find_target);

  /**
   * @brief 计算角度和角度差
   */
  void Angle();

  /**
   * @brief 计算转动方向
   * @details 1：顺时针 -1：逆时针 0：不转动
   */
  void Direction();

  /**
   * @brief 获取风车转向
   * @return int
   */
  int getState();

  /**
   * @brief 计算当前扇叶转动速度
   */
  void Velocity();

  // 角度
  // 当前目标角度
  float current_angle_;
  // 上一次目标角度
  float last_angle_;
  // 两帧之间角度差
  float diff_angle_;
  // 上一帧扇叶移动的角度
  float last_diff_angle_;

  // 方向
  float filter_direction_;    // 第二次滤波方向
  int final_direction_;       // 最终的方向
  int last_final_direction_;  // 上一次最终的方向
  float current_direction_;   // 当前方向
  float last_direction_;      // 上一次方向
  int find_cnt_;              // 发现目标次数
  float d_angle_;             // 滤波器系数
  int confirm_cnt_;           // 记录达到条件次数
  bool is_confirm_;           // 判断是否达到条件

  // 速度
  float current_speed_;  //当前转速
  double last_time_;     // 上一帧的时间

 private:
  /* 计算预测量 */

  /**
   * @brief 计算预测量
   * @param[in]  _bullet_velocity 子弹速度
   * @param[in]  _is_find_target  是否发现目标
   * @return float 预测量
   */
  float Predict(const float& _bullet_velocity, const bool& _is_find_target);

  /**
   * @brief 计算固定预测量
   * @param[in]  _bullet_velocity 子弹速度
   * @return float  预测量
   */
  float fixedPredict(const float& _bullet_velocity);

  // 变化预测量
  void mutativePredict(const float& _input_predict_quantity,
                       float& _output_predict_quantity);

  // 当前弧度
  float current_radian_;
  // 枪口到扇叶底部高度
  float barrel_buff_botton_h_;
  // 目标在扇叶上的高度
  float target_buff_h_;
  // 真实目标高度
  float target_y_;
  // 目标到高台的直线距离
  float target_x_;
  // 目标直线距离
  float target_z_;
  // 手动补偿值
  int offset_angle_int_;
  float offset_angle_float_;
  // 子弹飞行时间
  float bullet_tof_;
  // 固定预测量（扇叶的单帧移动量）
  float fixed_forecast_quantity_;
  // 最终合成的预测量
  float final_forecast_quantity_;

 private:
  /* 计算获取最终目标（矩形、顶点） */

  /**
   * @brief 计算最终目标矩形顶点点集
   * @param  _predict_quantity 预测量
   * @param  _final_center_r   圆心坐标（src）
   * @param  _target_2d_point  目标矩形顶点容器
   * @param  _input_dst_img    输入画板
   * @param  _is_find_target   是否有目标
   */
  void calculateTargetPointSet(const float& _predict_quantity,
                               const Point2f& _final_center_r,
                               std::vector<Point2f>& _target_2d_point,
                               Mat& _input_dst_img,
                               const bool& _is_find_target);

  // 特殊的弧度
  double theta_;
  // 最终角度
  float final_angle_;
  // 最终弧度
  float final_radian_;
  // 计算的sin值
  float sin_calcu_;
  // 计算的cos值
  float cos_calcu_;
  // 最终预测点
  Point2f pre_center_;
  // 轨迹圆半径
  float radio_;

 private:
  /* 计算云台角度 */

  angle_solve::RM_Solvepnp buff_pnp_ = angle_solve::RM_Solvepnp(
      "devices/camera/cameraParams/cameraParams_407.xml",
      "module/angle_solve/pnp_config.xml");

 private:
  /* 自动控制 */

 private:
  /* 更新上一帧数据 */

  void updateLastData(const bool& _is_find_target);

 private:
  // 帧率测试
  RM_FPS buff_fps_1_{"Part 1"};
  RM_FPS buff_fps_2_{"Part 2"};
  RM_FPS buff_fps_;  // 计算时间
};

inline void RM_Buff::Input(Mat& _input_img, const int& _my_color) {
  this->src_img_ = _input_img;
  this->my_color_ = _my_color;
  this->src_img_.copyTo(this->dst_img_);
  this->is_find_target_ = false;
}

inline void RM_Buff::displayDst() { imshow("dst_img", this->dst_img_); }
}  // namespace buff

#endif  // !RM_BUFFDETECTION_H