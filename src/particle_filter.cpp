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

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;
using std::uniform_real_distribution;

namespace
{
  std::default_random_engine gen;

  std::set<int> particle_ids;

  double multiv_prob(double sig_x, double sig_y, double x_obs, double y_obs,
                     double mu_x, double mu_y) {
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

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 50;  // TODO: Set the number of particles

  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (size_t i = 0; i < num_particles; ++i)
  {
    // Add exact position particles
//    particles.push_back(Particle { (int) i + 1, x, y, theta, 1 });
    particles.push_back(Particle { (int) i + 1, dist_x(gen) , dist_y(gen), dist_theta(gen), 1 });
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  // SrKo: I have assumed here std's follow order of the velocity and yaw_rate parameters.
  normal_distribution<double> dist_v(velocity, std_pos[0]);
  normal_distribution<double> dist_yaw_rate(yaw_rate, std_pos[1]);

  for(auto & p : particles)
  {
    const auto v_measured = dist_v(gen);
    const auto delta_theta_measured = dist_yaw_rate(gen); // yaw_rate; //

    // Movement prediction model
    if(abs(delta_t) < 0.0001)
    {
      p.x = p.x + v_measured * delta_t * cos(p.theta);
      p.y = p.y + v_measured * delta_t * sin(p.theta);
      // p.theta is unchanged
    }
    else
    {
      p.x = p.x + v_measured * (sin(p.theta + delta_theta_measured * delta_t) - sin(p.theta)) / delta_theta_measured;
      p.y = p.y + v_measured * (cos(p.theta) - cos(p.theta + delta_theta_measured * delta_t)) / delta_theta_measured;
      p.theta = p.theta + delta_theta_measured * delta_t;
    }
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
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
  if(map_landmarks.landmark_list.size() == 0)
  {
    throw "Landmark map can not be empty!";
  }

  // Filter measurements to those within sensor range
  vector<LandmarkObs> observations_filtered;
  for(const LandmarkObs &o: observations)
  {
      if(dist(0, 0, o.x, o.y) > sensor_range)
      {
        continue;
      }
      observations_filtered.push_back(o);
  }

  double weight_sum = 0;

  for(Particle &p: particles)
  {
    // Clear previous associations
    p.associations.clear();
    p.sense_x.clear();
    p.sense_y.clear();
    p.weight = 1;

    // Mozemo prebaciti u DataAssociation funkciju
    for(const LandmarkObs &o: observations_filtered)
    {
      // Transform each landmark observation coordinates from the viewpoint of the particle, to the
      // map coordinate system. This would be predicted landmark coordinates.
      const auto sense_x = p.x + cos(p.theta) * o.x - sin(p.theta) * o.y;
      const auto sense_y = p.y + sin(p.theta) * o.x + cos(p.theta) * o.y;

      double l_x = 0;
      double l_y = 0;
      int landmark_id = -1;
      double min_dist = std::numeric_limits<double>::max();

      // Given predicted measurements in the map coordinates, find associated landmarks
      for(const Map::single_landmark_s &l: map_landmarks.landmark_list)
      {
        const double current_dist = dist(sense_x, sense_y, l.x_f, l.y_f);
        if(current_dist < min_dist)
        {
          min_dist = current_dist;
          landmark_id = l.id_i;
          l_x = l.x_f;
          l_y = l.y_f;
        }
      }

      // Update landmark associations for the given particle
      p.associations.push_back(landmark_id);
      p.sense_x.push_back(sense_x);
      p.sense_y.push_back(sense_y);

      // Update weightsweight = {double} 1.0302681898920017e-08
      p.weight *= multiv_prob(std_landmark[0], std_landmark[1], sense_x, sense_y, l_x, l_y);
    }

    weight_sum += p.weight;
  }

  // Normalize the weights across particles
  for(Particle &p: particles)
  {
    p.weight /= weight_sum;
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  // Problem: Only a few or just one particle survives the resample process

  uniform_real_distribution<double> u_dist;
  std::vector<Particle> new_particles;

  for(size_t i = 0; i < particles.size(); ++i)
  {
    auto beta = u_dist(gen);

    for(Particle &p: particles)
    {
      if(beta > p.weight)
      {
        beta -= p.weight;
      }
      else
      {
        new_particles.push_back(p);
        break;
      }
    }

    // In unlikely case that beta is larger than the sum of all weights, which may occur
    // due to rounding errors.
    if(new_particles.size() != i + 1)
    {
      new_particles.push_back(particles.back());
    }
  }

  // Debug: show number of different particles
  particle_ids.clear();
  for(const Particle &p: new_particles)
  {
    if(particle_ids.find(p.id) == particle_ids.end())
    {
      particle_ids.insert(p.id);
    }
  }

  std::cout << "Remaining unique particles: " << particle_ids.size() << std::endl;

  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}