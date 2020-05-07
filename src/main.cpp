/* Applied Video Analysis of Sequences (AVSA)
 *	LAB3: Object Tracking
 */

//system libraries
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>

//openCV
#include <opencv2/opencv.hpp>
#include <opencv2/video/background_segm.hpp>

#include "FgSegment.hpp"
#include "Kalman.hpp"
#include "ShowManyImages.hpp"

//namespaces
using namespace cv;
using namespace std;


//main function
int main(int argc, char ** argv) 
{
	    //std::string inputvideo = "/home/avsa/AVSA2020datasets/dataset_lab3/lab3.3/abandonedBox_600_1000_clip.mp4"; //Video Seq works well with the given params
	    //std::string inputvideo = "/home/avsa/AVSA2020datasets/dataset_lab3/lab3.3/boats_6950_7900_clip.mp4";
	    //std::string inputvideo = "/home/avsa/AVSA2020datasets/dataset_lab3/lab3.3/pedestrians_800_1025_clip.mp4";
    	std::string inputvideo = "/home/avsa/AVSA2020datasets/dataset_lab3/lab3.3/streetCornerAtNight_0_100_clip.mp4";


	    //INIT NEW VARIABLES
	    int int_counter = 0;
	    Mat frame, fgmask;
	    Mat frame_blob, frame_measurment, frame_common, frame_trajectory; //to show later results


	    Kalman kalman(2); // 1 - velocity, 2 - acceleration
	    FgSegment fgseg; // measurement extraction

	    VideoCapture cap;

	    //Read video

	    cap.open(inputvideo);
	    	if (!cap.isOpened())
	    	{
	    		std::cout << "Could not open video file " << inputvideo << std::endl;
	    		return -1;
	    	}
	    	cap >> frame; //get first video frame


	     //Go through all video

             while(true)
             {
				  cap.read(frame);

				  if(!frame.data){cout<<"Finished";break;};

				  // copy exact image for
				  frame.copyTo(frame_trajectory);
				  frame.copyTo(frame_blob);
				  frame.copyTo(frame_measurment);
				  frame.copyTo(frame_common);

				 //fg_segmentation
				  fgmask = fgseg.bkg_subtraction(frame);

                 //morphological_opening
				  fgmask = fgseg.MorphologicalOpen(fgmask);

				 //extract blobs
				  fgseg.extractBlobs(fgmask);

				 //Kalman filter
				  kalman.predict(fgseg.getBlobCenters(),fgseg.BlobExists());


				 //draw results
              	  kalman.draw(fgseg.getBlobCenters(), frame_common, frame_measurment, frame_trajectory);


              	 //Show videos
              	 ShowManyImages("frame | fgmask | blob || measurement | common | trajectory", 6,

           						                            frame,
															fgmask,

															fgseg.paintBlobImage(frame_blob, fgseg.getBloblist()),
															fgseg.paintBlobImage(frame_measurment, fgseg.getBloblist()),
															fgseg.paintBlobImage(frame_common, fgseg.getBloblist()),
															fgseg.paintBlobImage(frame_trajectory, fgseg.getBloblist())
              	 );

				  if(waitKey(30) == 27) break;
				  int_counter++;
             }

	        cap.release();
	     	destroyAllWindows();
	     	waitKey(0);
	     	return 0;
	    }
