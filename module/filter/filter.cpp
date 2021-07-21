#include "filter.hpp"
namespace kalman {
RM_kalmanfilter::RM_kalmanfilter() : KF_(4, 2) {
  measurement_matrix = cv::Mat::zeros(2, 1, CV_32F);
  KF_.transitionMatrix =
      (cv::Mat_<float>(4, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);

  setIdentity(KF_.measurementMatrix, cv::Scalar::all(1));
  setIdentity(KF_.processNoiseCov, cv::Scalar::all(1e-3));  //
  setIdentity(KF_.measurementNoiseCov,
              cv::Scalar::all(1e-2));  //测量协方差矩阵R，数值越大回归越慢
  setIdentity(KF_.errorCovPost, cv::Scalar::all(1));

  KF_.statePost = (cv::Mat_<float>(4, 1) << x, y, 0, 0);  //初始值
}

RM_kalmanfilter::~RM_kalmanfilter() {}

cv::Point2f RM_kalmanfilter::predict_point(cv::Point2f _p) {
  // p = _p;
  cv::Mat prediction = KF_.predict();
  cv::Point2f predict_pt = cv::Point2f(prediction.at<float>(0), 0);

  measurement_matrix.at<float>(0, 0) = _p.x;

  KF_.correct(measurement_matrix);
  return predict_pt;
}

void RM_kalmanfilter::reset() {
  measurement_matrix = cv::Mat::zeros(2, 1, CV_32F);
}

//
float RM_kalmanfilter::use_RM_KF(float _top,
                                 cv::Point _armor_center) {
  if (_armor_center == cv::Point()) {
    top_angle_differ.get_top_times = 0;
    return 0;
  }
  double armor_vary = std::sqrt((lost_armor_center.x - _armor_center.x) *
                                    (lost_armor_center.x - _armor_center.x) +
                                (lost_armor_center.y - _armor_center.y) *
                                    (lost_armor_center.y - _armor_center.y));
  lost_armor_center = _armor_center;

  cv::namedWindow("filter_trackbar");
  cv::createTrackbar("multiple_", "filter_trackbar", &multiple_, 1000, NULL);
  cv::createTrackbar("first_ignore_time_", "filter_trackbar",
                     &first_ignore_time_, 100, NULL);
  cv::createTrackbar("armor_threshold_max_", "filter_trackbar", &armor_threshold_max_, 100,
                     NULL);
  cv::imshow("filter_trackbar", filter_trackbar_);

  if(armor_vary > armor_threshold_max_){
    std::cout<<"---------------------------------------------------------tab armor-------------------------------------------------------" << std::endl;
    top_angle_differ.get_top_times = 0;
    return 0;
  }

  top_angle_differ.top_angle_ = _top;

  // 第一次获取的陀螺仪数据时, 对上一时刻(不存在)的陀螺仪数据的假设==> 0 / 此时刻与上一时刻的陀螺仪数值相同
  if (top_angle_differ.get_top_times == 0) {
    top_angle_differ.top_angle = _top;
  }

  top_angle_differ.get_top_times++;  // 自动计数 -> 获得陀螺仪数据的次数

  top_angle_differ.differ =
      top_angle_differ.top_angle_ - top_angle_differ.top_angle;

  if (top_angle_differ.last_differ != 0)
  {
    if(top_angle_differ.differ > 0.5)
    {
      top_angle_differ.differ = top_angle_differ.last_differ;
    }
  }
  top_angle_differ.last_differ = top_angle_differ.differ;

  // 这一时刻的陀螺仪数据 更新为 上一时刻的陀螺仪数据
  top_angle_differ.top_angle = top_angle_differ.top_angle_;

  cv::Point2f top_differ = cv::Point2f(top_angle_differ.differ * multiple_, 0);

  return predict_point(top_differ).x;

}
}  // namespace kalman
