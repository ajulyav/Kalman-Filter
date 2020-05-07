/*
 * Kalman.cpp
 */

#include "Kalman.hpp"

Kalman::Kalman(int type)

{
	center_update = Point(0 ,0);
	Kalman_type = type;

	if(type == 1)
	{
		ST_size = 4;
	}
	else
	{
		ST_size = 6;
	}

	MEAS_size = 2;

	// initialize
	STATE = Mat::zeros(ST_size,1,CV_32F);
    MEASUREMENT = Mat::zeros(MEAS_size,1,CV_32F);
 	KF.init(ST_size,MEAS_size,0,CV_32F);
    set_kalman_param();

}


void Kalman::set_kalman_param()

{
	   //Matrixes for Kalman algorithm
		A = Mat::zeros(Size(ST_size, ST_size), CV_32F);
		Q = Mat::eye(Size(ST_size, ST_size), CV_32F);
		P = Mat::eye(Size(ST_size, ST_size), CV_32F);
		H = Mat::zeros(Size(ST_size, MEAS_size), CV_32F);
		R = Mat::eye(Size(2, 2), CV_32F);

		switch(Kalman_type){

		case 1:

	        setIdentity(A);
	        A.at<float>(0,1) = A.at<float>(2,3) = 1.0f;

	        Q.at<float>(0,0) = Q.at<float>(2,2) = 25.0f;
	        Q.at<float>(1,1) = Q.at<float>(3,3) = 10.0f;

	        H.at<float>(0,0) = H.at<float>(1,2) = 1.0f;

			break;

		case 2:
			 setIdentity(A);
 			 A.at<float>(0,2) = A.at<float>(3,5) = 0.5f;
		     A.at<float>(0,1) = A.at<float>(1,2)=A.at<float>(3,4) =A.at<float>(4,5)= 1.0f;


			 /* For The rest
			  *
			 	 Q.at<float>(0,0) = Q.at<float>(3,3) = 25.0f;
			 	 Q.at<float>(1,1) = Q.at<float>(4,4) = 10.0f;
			 	 Q.at<float>(2,2) = Q.at<float>(5,5) = 1.0f;
			  *
			  */


		 	 Q.at<float>(0,0) = Q.at<float>(3,3) = 25.0f;
		 	 Q.at<float>(1,1) = Q.at<float>(4,4) = 10.0f; //For
		 	 Q.at<float>(2,2) = Q.at<float>(5,5) = 1.0f;

			 H.at<float>(0,0) = H.at<float>(1,3) = 1.0f;

		  break;

		default :
			 break;

		}
		A.copyTo(KF.transitionMatrix);
		Q.copyTo(KF.processNoiseCov);
		H.copyTo(KF.measurementMatrix);

		//Prediction error COV
		setIdentity(P,Scalar(10e4f));
		P.copyTo(KF.errorCovPre);

		//Measurement Noise COV

		/* For all sequences except boat
		 * setIdentity(R,Scalar(25.0f));
		 */

		/* Try out for Boat
		 * If we decrease the observation matrix values it means that we trust my observation more. However for boat, since we find
		 * cars it means we probably don't want to trust it as much, specially on the y axis. R(1,1)
		 * The value refers to the pixels of noise tolerance 50 and 100 respectively
		 * This tunning is important on the boats sequence due to the cars object detection

		 R.at<float>(0,0) = 100.0f;  //Having problems with the x trustworthy of our observations.
		 R.at<float>(1,1) = 100.0f;

		*/
		R.at<float>(0,0) = 100.0f;  //Having problems with the x trustworthy of our observations.
		R.at<float>(1,1) = 100.0f;
		R.copyTo(KF.measurementNoiseCov);

}

// predict
void Kalman::predict(std::vector<Point> centers, bool blob_extracted){

  if(Kalman_type == 1){constantVelocity(centers,blob_extracted);}
  else{constantAcceleration(centers,blob_extracted);}

}

void Kalman::constantVelocity(std::vector<Point> centers, bool blob_extracted){

    if(blob_extracted){

    	 if(!init){

    		 // Init Measurement
			 MEASUREMENT.at<float>(0) = centers[centers.size()-1].x;
			 MEASUREMENT.at<float>(1) = centers[centers.size()-1].y;

		    //Init State

			STATE.at<float>(0) = MEASUREMENT.at<float>(0);
			STATE.at<float>(1) = 0;
			STATE.at<float>(2) = MEASUREMENT.at<float>(1);
			STATE.at<float>(3)=0;

           // update Kalman with the State
			KF.statePost = STATE;

    	   setRoute("Initialized");

    	 }
    	 else
    	 {
			//a new observation
			MEASUREMENT.at<float>(0) = centers[centers.size()-1].x;
			MEASUREMENT.at<float>(1) = centers[centers.size()-1].y;

			KF.correct(MEASUREMENT);

			setRoute("Corrected");

    		//predict --> We make sure no prediction is performed without an initialization of ta measurement
			STATE = KF.predict();

    	 }

    	 init++;

    }
    else
    {
    	 // prediction
    	 if(init > 0){

		  MEASUREMENT.at<float>(0) = STATE.at<float>(0);
		  MEASUREMENT.at<float>(1) = STATE.at<float>(2);

		  setRoute("Predicted");

     	  //predict --> We make sure no prediction is performed without an initialization of ta measurement
    	  STATE = KF.predict();

    	 }

    }

    // save
    if(init>0)
    {
		center_update.x = STATE.at<float>(0);
		center_update.y = STATE.at<float>(2);
		points_pred.push_back(center_update);
    }

}

void Kalman::constantAcceleration(std::vector<Point> centers, bool blob_extracted){


    if(blob_extracted){

    	 if(!init){

    		 // Init Measurement
			 MEASUREMENT.at<float>(0) = centers[centers.size()-1].x;
			 MEASUREMENT.at<float>(1) = centers[centers.size()-1].y;

		    //Init State
			STATE.at<float>(0) = MEASUREMENT.at<float>(0);
			STATE.at<float>(1) = 0;
			STATE.at<float>(2) = 0;
			STATE.at<float>(3) = MEASUREMENT.at<float>(1);
			STATE.at<float>(4)= 0;
			STATE.at<float>(5) = 0;

		   //update Kalman with the State
		   KF.statePost = STATE;

    	   setRoute("Initialized");

    	 }
    	 // not first time
    	 else
    	 {

			//new observation
			MEASUREMENT.at<float>(0) = centers[centers.size()-1].x;
			MEASUREMENT.at<float>(1) = centers[centers.size()-1].y;
			KF.correct(MEASUREMENT);

			setRoute("Corrected");

			//predict
			STATE = KF.predict();
    	 }

    	 init++;

    }
    else{

    	 //prediction
    	 if(init > 0){

          STATE = KF.predict();

		  MEASUREMENT.at<float>(0) = STATE.at<float>(0);
		  MEASUREMENT.at<float>(1) = STATE.at<float>(3);

		  setRoute("Predicted");

    	 }

    }

    // save
    if(init>0)
    {
		center_update.x = STATE.at<float>(0);
		center_update.y = STATE.at<float>(3);
		points_pred.push_back(center_update);
    }
}


// draw trajectory
void Kalman::draw(std::vector<Point> centers, Mat &frame_common, Mat &frame_measurment, Mat &frame_trajectory)

{

	// Color Selection
	 Scalar meas_color = Scalar(255,0,0);
	 Scalar est_color = Scalar (0,0,255);
	 Scalar pred_color = Scalar (0,255,0);
     Scalar color;

     // Set the colors
	 for(int i = 0 ;i<getPointsPred().size();i++)
	 {
		 if(Kalman::getRoute()[i] == "Initialized")
		 {
			 color = meas_color;
		 }
		 if(Kalman::getRoute()[i] == "Corrected")
		 {
			 color = est_color;
		 }
		 if(Kalman::getRoute()[i] == "Predicted")
		 {
			 color = pred_color;
		 }

		 circle(frame_common,getPointsPred()[i], 5, color);


		 // show measurements route
	  for(int i = 0; i<centers.size(); i++)
		 {
			circle(frame_measurment,centers[i], 5, meas_color);
			circle(frame_common,centers[i], 5, meas_color);
		 }

	  // show predicted trajectory
 	   if(i!=0) {line(frame_trajectory ,getPointsPred()[i-1] , getPointsPred()[i],est_color,1);}

	 }

     // put text
	 putText(frame_common,"Measurement", Point(20,20), FONT_HERSHEY_SIMPLEX, 0.7, meas_color, 1);
     putText(frame_common,"Estimated", Point(20,40), FONT_HERSHEY_SIMPLEX, 0.7, est_color, 1);
	 putText(frame_common,"Predicted", Point(20,60), FONT_HERSHEY_SIMPLEX, 0.7, pred_color, 1);

}
