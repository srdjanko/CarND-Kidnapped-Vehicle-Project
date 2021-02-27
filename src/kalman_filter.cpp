#include "kalman_filter.h"

/* 
 * Please note that the Eigen library does not initialize 
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd x_in, MatrixXd P_in, MatrixXd F_in,
                        MatrixXd H_in, MatrixXd R_in, MatrixXd Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;

  I = MatrixXd::Identity(x_in.size(), x_in.size());
}

void KalmanFilter::Predict() {
  /**
   * TODO: predict the state
   */
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::PredictEKF(const MatrixXd &f_dt, const MatrixXd &Fj_dt) {
  /**
   * TODO: predict the state
   */
  x_ = x_ + f_dt;
  P_ = Fj_dt * P_ * Fj_dt.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
   * TODO: update the state by using Kalman Filter equations
   */
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  // new estimate
  x_ = x_ + K * y;
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z, const VectorXd &h, const MatrixXd &Hj) {
  /**
   * TODO: update the state by using Extended Kalman Filter equations
   */
  VectorXd res = z - h;
  MatrixXd Hjt = Hj.transpose();
  MatrixXd S = Hj * P_ * Hjt + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Hjt;
  MatrixXd K = PHt * Si;

  // new estimate
  x_ = x_ + K * res;
  P_ = (I - K * Hj) * P_;
}
