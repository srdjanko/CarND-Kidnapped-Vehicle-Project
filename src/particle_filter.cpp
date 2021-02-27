/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <set>
#include "kalman_filter.h"

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;
using std::uniform_real_distribution;

using MatrixXd = Eigen::MatrixXd;
using VectorXd = Eigen::VectorXd;

namespace
{
  std::default_random_engine gen;

//  vector<LandmarkObs> tracked_obs;
  KalmanFilter kf;
  VectorXd f_dt(3);
  MatrixXd Fj_dt(3, 3);

  // Previous best
  Particle best_p;

  int initialize_kl()
  {
    // state vector
    VectorXd x(3);
    x << 0, 0, 0;

    // state covariance matrix P
    MatrixXd P(3, 3);
    P << 10, 0, 0,
        0, 10, 0,
        0, 0, 100;

    MatrixXd H(2, 3);
    H << 1, 0, 0,
        0, 1, 0;

    // measurement covariance matrix - laser
    MatrixXd R_laser(2, 2);
    R_laser << 0.3, 0,
        0, 0.3;

    // process covariance matrix
    MatrixXd Q(3, 3);
    Q << 0.1, 0, 0,
        0, 0.1, 0,
        0, 0, 0.01;

    kf.Init(x, P, MatrixXd(0, 0), H, R_laser, Q);
  }

  double multiv_prob(double sig_x, double sig_y, double x_obs, double y_obs,
                     double mu_x, double mu_y)
  {
    // calculate normalization term
    double gauss_norm;
    gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

    // calculate exponent
    double exponent;
    exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2)))
               + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));

    // calculate weight using normalization terms and exponent
    double weight;
    weight = gauss_norm * exp(-exponent);

    return weight;
  }
}

void ParticleFilter::init(double x, double y, double theta, double std[])
{
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 450;  // TODO: Set the number of particles

  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (size_t i = 0; i < num_particles; ++i)
  {
    // Add exact position particles
    particles.push_back(Particle{(int) i + 1, dist_x(gen), dist_y(gen), dist_theta(gen), 1});
  }

  // Initialize Kalman filter states
  kf.x_ << dist_x(gen), dist_y(gen), dist_theta(gen);

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate)
{
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  for (auto &p : particles)
  {
    // Movement prediction model
    if (abs(yaw_rate) < 0.00001)
    {
      p.x += velocity * delta_t * cos(p.theta);
      p.y += velocity * delta_t * sin(p.theta);
//      p.theta += yaw_rate * delta_t;
    } else
    {
      p.x += velocity * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta)) / yaw_rate;
      p.y += velocity * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t)) / yaw_rate;
      p.theta += yaw_rate * delta_t;
    }

    // Update KF
    const auto theta = kf.x_[2];
    f_dt << velocity * cos(theta),
        velocity * sin(theta),
        yaw_rate;

    f_dt *= delta_t;

    Fj_dt << 0, 0, -velocity * sin(theta),
        0, 0, velocity * cos(theta),
        0, 0, 0;

    Fj_dt *= delta_t;
    kf.PredictEKF(f_dt, Fj_dt);

    // SrKo: Why do we add (x,y,z) noise when it is (velocity, yaw_rate) that is measured?
    p.x += dist_x(gen);
    p.y += dist_y(gen);
    p.theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> &predicted,
                                     vector<LandmarkObs> &observations)
{
  /**
   * TODO: Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will
   *   probably find it useful to implement this method and use it as a helper
   *   during the updateWeights phase.
   */
  // Given predicted measurements in the map coordinates, find associated landmarks
  for (LandmarkObs &o: observations)
  {
    double min_dist = std::numeric_limits<double>::max();
    for (const LandmarkObs &p: predicted)
    {
      const double current_dist = dist(p.x, p.y, o.x, o.y);
      if (current_dist < min_dist)
      {
        min_dist = current_dist;
        o.id = p.id;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks)
{
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian
   *   distribution. You can read more about this distribution here:
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system.
   *   Your particles are located according to the MAP'S coordinate system.
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  vector<LandmarkObs> predicted_observations;
  vector<LandmarkObs> transformed_observations;

  // Use Kalman filter to filter out the noise from the landmark measurements
  // Kalman filter will be used for the previous best values particle
  const auto &best = std::max(particles.begin(), particles.end(), [](Particle &p)
  { return p.weight; });

  // Associate observations with previous best prediction
  for (const auto &o: observations)
  {
    const auto sense_x = best->x + cos(best->theta) * o.x - sin(best->theta) * o.y;
    const auto sense_y = best->y + sin(best->theta) * o.x + cos(best->theta) * o.y;

    transformed_observations.push_back(LandmarkObs{0, sense_x, sense_y});
  }

  // We want to make sure that previous observations, linked to the same landmark
  // are not too distance from the current observations
  vector<LandmarkObs> previous_observations;

  for (size_t i; i < best->associations.size(); ++i)
  {
    previous_observations.push_back(LandmarkObs{best->associations[i], best->sense_x[i], best->sense_y[i]});
  }

  dataAssociation(previous_observations, transformed_observations);

  // If previous and current observations are close, update Kalman Filter
  for (auto &to: transformed_observations)
  {
    // If we already track a landmark from previous step and it is in the vicinity of
    // new position, then we have a candidate to update KF

    // Find previous landmark
    const auto &it_tracked_to = std::find_if(previous_observations.begin(), previous_observations.end(),
                                             [&to](LandmarkObs &o)
                                             { return to.id == o.id; });

    if (dist(it_tracked_to->x, it_tracked_to->y, to.x, to.y) > 2.0)
    {
      continue;
    }

    // Update KF with this measurement
    VectorXd z(2);
    z << to.x, to.y;
    kf.Update(z);
  }

  for (Particle &p: particles)
  {
    // Clear previous associations
    p.associations.clear();
    p.sense_x.clear();
    p.sense_y.clear();
    p.weight = 1;

    transformed_observations.clear();
    predicted_observations.clear();

    // Select observations that are within sensor range of the particle
    for (const auto &l: map_landmarks.landmark_list)
    {
      if (dist(p.x, p.y, l.x_f, l.y_f) <= sensor_range)
      {
        predicted_observations.push_back(LandmarkObs{l.id_i, l.x_f, l.y_f});
      }
    }

    // Transform each landmark observation coordinates from the viewpoint of the particle, to the
    // map coordinate system.
    for (const auto &o: observations)
    {
      const auto sense_x = p.x + cos(p.theta) * o.x - sin(p.theta) * o.y;
      const auto sense_y = p.y + sin(p.theta) * o.x + cos(p.theta) * o.y;

      transformed_observations.push_back(LandmarkObs{0, sense_x, sense_y});
    }

    dataAssociation(predicted_observations, transformed_observations);

    // Update landmark associations for the given particle
    for (auto &t: transformed_observations)
    {
      p.associations.push_back(t.id);
      p.sense_x.push_back(t.x);
      p.sense_y.push_back(t.y);

      // Find associated predicted landmark
      const auto &it = std::find_if(predicted_observations.begin(), predicted_observations.end(), [&t](LandmarkObs o)
      { return t.id == o.id; });

      p.weight *= multiv_prob(std_landmark[0], std_landmark[1], t.x, t.y, it->x, it->y);
    }
  }
}

void ParticleFilter::resample()
{
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  vector<double> weights(particles.size());
  std::transform(particles.begin(), particles.end(), weights.begin(), [](Particle &p){ return p.weight;});
  std::discrete_distribution<>  w_dist(weights.begin(), weights.end());

  // Problem: Only a few or just one particle survives the resample process
  std::vector<Particle> new_particles;

  for(size_t i = 0; i < num_particles; ++i)
  {
    new_particles.push_back(particles[w_dist(gen)]);
  }

  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle &particle,
                                     const vector<int> &associations,
                                     const vector<double> &sense_x,
                                     const vector<double> &sense_y)
{
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord)
{
  vector<double> v;

  if (coord == "X")
  {
    v = best.sense_x;
  } else
  {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}