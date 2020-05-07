/*
 * Kalman.hpp
 */

#ifndef SRC_KALMAN_HPP_
#define SRC_KALMAN_HPP_

#include "opencv2/opencv.hpp"
using namespace cv;
using namespace std;

class Kalman{

// state transition matrix, observation matrix, state COV error, process COV noise
	 Mat A, H, P, Q, R;
	 Mat STATE , MEASUREMENT;


// 	 sizes
	 int ST_size, MEAS_size;

	 KalmanFilter KF;
	 int Kalman_type;
	 int init = 0 ;

 	 Point center_update;

// type of route for coloring
 	std::vector<std::string> route;

// to store predicted points for trajectory
	std::vector<cv::Point> points_pred;


 public:

	Kalman(int type);

    // set parameters
    void set_kalman_param();

    // predict (depends on the type)
	void predict(std::vector<Point> centers, bool blob_extracted);

	// Velocity
	void constantVelocity(std::vector<Point> centers, bool blob_extracted);

	// Acceleration
	void constantAcceleration(std::vector<Point> centers, bool blob_extracted);

	// draw results
	void draw(std::vector<Point> centers, Mat &frame_common, Mat &frame_measurment, Mat &trajectory);


	// GETTERS AND SETTERS

	const Mat& getSTATE() const {return STATE;}
	int getSize() const {return ST_size;}
	const Mat& getMEASUREMENT() const {return MEASUREMENT;}

	const std::vector<std::string> getRoute() const {return route;}
	void setRoute(string input) {route.push_back(input);}

	const std::vector<cv::Point> getPointsPred() const {return points_pred;}
	int getInit() const {return init;}
};

#endif
